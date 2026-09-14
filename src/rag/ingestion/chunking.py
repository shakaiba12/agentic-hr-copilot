"""
Structure-Aware Markdown Filtering and Chunking Layer for Enterprise RAG Pipeline.
Preserves document hierarchy, lists, FAQ pairs, definitions, and sentence boundaries.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DocumentChunk:
    """Internal chunk representation containing text, hierarchy, and metadata."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    heading_path: list[str] = field(default_factory=list)
    content_type: str = "section"  # "section", "paragraph", "list", "faq", "definition", "notice"
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration parameters for structure-aware chunking."""

    max_chunk_size: int = 1200
    min_chunk_size: int = 150  
    overlap: int = 120

    def __post_init__(self) -> None:
        if self.max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {self.max_chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap cannot be negative, got {self.overlap}")
        if self.overlap >= self.max_chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly less than max_chunk_size ({self.max_chunk_size})"
            )
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size cannot be negative, got {self.min_chunk_size}")


class MarkdownFilter:
    """Deterministic, meaning-preserving Markdown normalizer and cleaner."""

    _HTML_TAG_RE = re.compile(r"<span[^>]*>(.*?)</span>", re.IGNORECASE | re.DOTALL)
    _GENERIC_HTML_RE = re.compile(r"<(div|p|b|i|u|strong|em)[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
    _BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
    _TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)

    def normalize(self, text: str) -> str:
        """
        Normalize Markdown formatting without altering content, semantics, or structure.
        """
        if not text:
            return ""

        # 1. Unicode normalization (NFKC)
        cleaned = unicodedata.normalize("NFKC", text)

        # 2. Strip unprintable control characters
        cleaned = self._CONTROL_CHAR_RE.sub("", cleaned)

        # 3. Clean inline HTML tags while preserving inner text
        cleaned = self._BR_TAG_RE.sub("\n", cleaned)
        cleaned = self._HTML_TAG_RE.sub(r"\1", cleaned)
        cleaned = self._GENERIC_HTML_RE.sub(r"\2", cleaned)

        # 4. Normalize line breaks and trim trailing whitespace per line
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = self._TRAILING_WHITESPACE_RE.sub("", cleaned)

        # 5. Compress 3+ consecutive blank lines to 2
        cleaned = self._EXCESS_BLANK_LINES_RE.sub("\n\n", cleaned)

        return cleaned.strip()


class StructureAwareChunker:
    """
    Heading-aware, semantic-block-aware, size-constrained hierarchical chunker.
    """

    # CommonMark compliant heading regex
    _HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?$")

    # FAQ Question patterns
    _FAQ_Q_EXPLICIT_RE = re.compile(
        r"^(Q[:.]|\*\*Q[:.]?\*\*|\*\*Question[:.]?\*\*|Question[:.])\s+",
        re.IGNORECASE,
    )
    _FAQ_BOLD_FIELD_RE = re.compile(
        r"^\*\*(What|When|How|Why|Where|Who)[:.]?\*\*\s*",
        re.IGNORECASE,
    )

    # Notice pattern
    _NOTICE_RE = re.compile(
        r"^(\*{0,2}(Note|Warning|Important|Caution|Notice|Alert)\*{0,2}[:.]\s*|>+\s*\[!(NOTE|WARNING|IMPORTANT|CAUTION|NOTICE)\])",
        re.IGNORECASE,
    )

    # List item pattern
    _LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$")

    # Common abbreviations to protect during sentence splitting
    _KNOWN_ABBREVIATIONS: Set[str] = {
        "e.g", "i.e", "etc", "vs", "u.s", "u.k", "dr", "mr", "mrs", "ms",
        "prof", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept",
        "oct", "nov", "dec", "inc", "ltd", "corp", "co", "no", "dept",
        "gov", "sec", "ref", "approx", "fig", "al", "st", "ave"
    }

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        filter_layer: Optional[MarkdownFilter] = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.filter = filter_layer or MarkdownFilter()

    def chunk_document(self, text: str, document_id: str = "doc") -> List[DocumentChunk]:
        """
        Produce structure-aware chunks from raw Markdown text.
        """
        normalized_text = self.filter.normalize(text)
        if not normalized_text:
            return []

        # 1. Parse document into structured sections tracking heading hierarchy
        raw_sections = self._parse_heading_hierarchy(normalized_text)

        chunks: List[DocumentChunk] = []
        chunk_counter = 1

        for heading_path, section_text in raw_sections:
            if not section_text.strip():
                continue

            # 2. Extract semantic units (paragraphs, lists, FAQ Q/A pairs, definitions, notices)
            semantic_units = self._extract_semantic_units(section_text)

            # 3. Assemble units into size-constrained blocks under current heading
            assembled_blocks = self._assemble_blocks(semantic_units)

            for block_text, content_type, was_split in assembled_blocks:
                clean_block = block_text.strip()
                if not clean_block:
                    continue

                # Hard size limit guarantee
                guaranteed_blocks = self._enforce_hard_limit(clean_block, content_type)
                for gb_text in guaranteed_blocks:
                    text_digest = hashlib.sha256(gb_text.encode("utf-8")).hexdigest()[:8]
                    chunk_id = f"{document_id}_{chunk_counter:04d}_{text_digest}"

                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=gb_text,
                        chunk_index=chunk_counter,
                        heading_path=list(heading_path),
                        content_type=content_type,
                        start_offset=None,
                        end_offset=None,
                        metadata={
                            "section": heading_path[-1] if heading_path else None,
                            "parent_sections": heading_path[:-1] if len(heading_path) > 1 else [],
                            "char_count": len(gb_text),
                            "estimated_tokens": max(1, len(gb_text) // 4),
                            "was_split": was_split,
                        },
                    )
                    chunks.append(chunk)
                    chunk_counter += 1

        # 4. Merge small chunks safely (same heading path, compatible content type)
        merged_chunks = self._merge_small_chunks(chunks)

        # 5. Compute exact normalized document offsets and finalize chunk sequential IDs
        return self._finalize_chunks(merged_chunks, normalized_text, document_id)

    def _parse_heading_hierarchy(self, text: str) -> List[Tuple[List[str], str]]:
        """
        Split markdown text into sections tracking heading path stack.
        """
        lines = text.split("\n")
        sections: List[Tuple[List[str], str]] = []

        current_heading_stack: List[Tuple[int, str]] = []  # [(level, title)]
        current_lines: List[str] = []

        for line in lines:
            heading_match = self._HEADING_RE.match(line)

            if heading_match:
                # Flush previous section content
                if current_lines:
                    heading_path = [h[1] for h in current_heading_stack]
                    section_content = "\n".join(current_lines).strip()
                    if section_content:
                        sections.append((heading_path, section_content))
                    current_lines = []

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Clean markdown formatting inside heading title
                clean_title = re.sub(r"[*_`]", "", title)
                clean_title = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", clean_title).strip()

                # Pop headings of equal or greater level
                while current_heading_stack and current_heading_stack[-1][0] >= level:
                    current_heading_stack.pop()

                current_heading_stack.append((level, clean_title))
            else:
                current_lines.append(line)

        # Flush final section
        if current_lines:
            heading_path = [h[1] for h in current_heading_stack]
            section_content = "\n".join(current_lines).strip()
            if section_content:
                sections.append((heading_path, section_content))

        return sections

    def _extract_semantic_units(self, section_text: str) -> List[Tuple[str, str]]:
        """
        Extract semantic units: paragraphs, complete lists, FAQ Q/A pairs, definition blocks, notices.
        """
        paragraphs = re.split(r"\n\n+", section_text)
        units: List[Tuple[str, str]] = []

        i = 0
        while i < len(paragraphs):
            p = paragraphs[i].strip()
            if not p:
                i += 1
                continue

            # 1. FAQ Question / Answer detection: keep Q + A together
            is_faq_q = bool(self._FAQ_Q_EXPLICIT_RE.match(p) or self._FAQ_BOLD_FIELD_RE.match(p))
            if is_faq_q and i + 1 < len(paragraphs):
                next_p = paragraphs[i + 1].strip()
                if not (self._FAQ_Q_EXPLICIT_RE.match(next_p) or self._FAQ_BOLD_FIELD_RE.match(next_p)):
                    combined_faq = f"{p}\n\n{next_p}"
                    units.append((combined_faq, "faq"))
                    i += 2
                    continue
                else:
                    units.append((p, "faq"))
                    i += 1
                    continue
            elif is_faq_q:
                units.append((p, "faq"))
                i += 1
                continue

            # 2. Notice detection
            if self._NOTICE_RE.match(p):
                units.append((p, "notice"))
                i += 1
                continue

            # 3. List detection: keep consecutive list items together
            lines = p.split("\n")
            if any(self._LIST_ITEM_RE.match(line) for line in lines):
                units.append((p, "list"))
                i += 1
                continue

            # 4. Definition detection: Term: Definition
            first_line = lines[0]
            if ":" in first_line and len(first_line.split(":", 1)[0].strip()) < 50 and not first_line.startswith(("http://", "https://")):
                units.append((p, "definition"))
                i += 1
                continue

            # 5. Default: Paragraph prose
            units.append((p, "paragraph"))
            i += 1

        return units

    def _assemble_blocks(
        self,
        semantic_units: List[Tuple[str, str]],
    ) -> List[Tuple[str, str, bool]]:
        """
        Assemble semantic units into size-constrained chunks respecting max_chunk_size and boundaries.
        Returns: List of (block_text, content_type, was_split)
        """
        blocks: List[Tuple[str, str, bool]] = []
        current_prose_texts: List[str] = []
        current_prose_length = 0

        for unit_text, unit_type in semantic_units:
            unit_len = len(unit_text)

            # If unit is a distinct standalone structure (notice, faq, definition, list)
            if unit_type in ("notice", "faq", "definition", "list"):
                # Flush any accumulated prose
                if current_prose_texts:
                    joined = "\n\n".join(current_prose_texts)
                    blocks.append((joined, "paragraph", False))
                    current_prose_texts = []
                    current_prose_length = 0

                if unit_len > self.config.max_chunk_size:
                    sub_blocks = self._split_large_unit(unit_text, unit_type)
                    for sb_text, sb_type in sub_blocks:
                        blocks.append((sb_text, sb_type, True))
                else:
                    blocks.append((unit_text, unit_type, False))
                continue

            # Standard paragraph prose: assemble together up to max_chunk_size
            if unit_len > self.config.max_chunk_size:
                if current_prose_texts:
                    joined = "\n\n".join(current_prose_texts)
                    blocks.append((joined, "paragraph", False))
                    current_prose_texts = []
                    current_prose_length = 0

                sub_blocks = self._split_large_unit(unit_text, "paragraph")
                for sb_text, sb_type in sub_blocks:
                    blocks.append((sb_text, sb_type, True))
                continue

            if current_prose_length + unit_len + 2 > self.config.max_chunk_size and current_prose_texts:
                joined = "\n\n".join(current_prose_texts)
                blocks.append((joined, "paragraph", False))
                current_prose_texts = []
                current_prose_length = 0

            current_prose_texts.append(unit_text)
            current_prose_length += unit_len + 2

        if current_prose_texts:
            joined = "\n\n".join(current_prose_texts)
            blocks.append((joined, "paragraph", False))

        return blocks

    def _split_large_unit(
        self,
        text: str,
        unit_type: str,
    ) -> List[Tuple[str, str]]:
        """
        Split a large unit that exceeds max_chunk_size using list/sentence boundaries and overlap.
        """
        lines = text.split("\n")
        # If it's a list or contains list items, split at list item boundaries
        if len(lines) > 1 and (unit_type == "list" or any(self._LIST_ITEM_RE.match(l) for l in lines)):
            return self._split_lines(lines, unit_type)

        # Fallback to sentence boundary splitting
        sentences = self._split_into_sentences(text)
        sub_blocks: List[Tuple[str, str]] = []
        current_sentences: List[str] = []
        current_len = 0

        for s in sentences:
            s_len = len(s)

            if s_len > self.config.max_chunk_size:
                if current_sentences:
                    block_txt = " ".join(current_sentences)
                    sub_blocks.append((block_txt, unit_type))
                    current_sentences = []
                    current_len = 0
                word_chunks = self._split_words(s)
                for wc in word_chunks:
                    sub_blocks.append((wc, unit_type))
                continue

            if current_len + s_len + 1 > self.config.max_chunk_size and current_sentences:
                block_txt = " ".join(current_sentences)
                sub_blocks.append((block_txt, unit_type))

                # Calculate overlap: retain last sentence(s) up to overlap config
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev_s in reversed(current_sentences):
                    if overlap_len + len(prev_s) + 1 <= self.config.overlap:
                        overlap_sentences.insert(0, prev_s)
                        overlap_len += len(prev_s) + 1
                    else:
                        break

                current_sentences = list(overlap_sentences)
                current_len = overlap_len

            current_sentences.append(s)
            current_len += s_len + 1

        if current_sentences:
            block_txt = " ".join(current_sentences)
            sub_blocks.append((block_txt, unit_type))

        return sub_blocks

    def _split_lines(
        self,
        lines: List[str],
        unit_type: str,
    ) -> List[Tuple[str, str]]:
        """Split line-based units (e.g. lists) keeping items whole."""
        sub_blocks: List[Tuple[str, str]] = []
        current_lines: List[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1

            if line_len > self.config.max_chunk_size:
                if current_lines:
                    block_txt = "\n".join(current_lines)
                    sub_blocks.append((block_txt, unit_type))
                    current_lines = []
                    current_len = 0
                word_chunks = self._split_words(line)
                for wc in word_chunks:
                    sub_blocks.append((wc, unit_type))
                continue

            if current_len + line_len > self.config.max_chunk_size and current_lines:
                block_txt = "\n".join(current_lines)
                sub_blocks.append((block_txt, unit_type))
                current_lines = []
                current_len = 0

            current_lines.append(line)
            current_len += line_len

        if current_lines:
            block_txt = "\n".join(current_lines)
            sub_blocks.append((block_txt, unit_type))

        return sub_blocks

    def _split_words(self, text: str) -> List[str]:
        """Word-level splitting fallback for sentences/lines exceeding max_chunk_size."""
        words = text.split(" ")
        chunks: List[str] = []
        current_words: List[str] = []
        current_len = 0

        for w in words:
            w_len = len(w)

            if w_len > self.config.max_chunk_size:
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []
                    current_len = 0
                for pos in range(0, w_len, self.config.max_chunk_size):
                    chunks.append(w[pos : pos + self.config.max_chunk_size])
                continue

            if current_len + w_len + 1 > self.config.max_chunk_size and current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_len = 0

            current_words.append(w)
            current_len += w_len + 1

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def _enforce_hard_limit(self, text: str, unit_type: str) -> List[str]:
        """Final safety guarantee that no chunk text exceeds max_chunk_size."""
        if len(text) <= self.config.max_chunk_size:
            return [text]
        return self._split_words(text)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Sentence splitter with abbreviation, URL, decimal, and quote protection.
        """
        if not text:
            return []

        pattern = re.compile(r'([.!?]+["\')\]]*)(?:\s+|$)')
        tokens: List[str] = []
        last_pos = 0

        for match in pattern.finditer(text):
            end_pos = match.end()
            cand = text[last_pos:end_pos].strip()

            words = cand.split()
            if words:
                last_word = words[-1].lower().rstrip('.!?"\')]}')
                if last_word in self._KNOWN_ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                    continue
                if re.search(r'\d+[.]\d*$', words[-1]):
                    continue

            tokens.append(cand)
            last_pos = end_pos

        if last_pos < len(text):
            remainder = text[last_pos:].strip()
            if remainder:
                if tokens:
                    tokens[-1] = f"{tokens[-1]} {remainder}"
                else:
                    tokens.append(remainder)

        return tokens if tokens else [text]

    def _merge_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Merge small chunks (< min_chunk_size) with adjacent sibling chunks sharing the same heading path.
        """
        if len(chunks) <= 1:
            return chunks

        merged: List[DocumentChunk] = []
        skip_next = False

        for i in range(len(chunks)):
            if skip_next:
                skip_next = False
                continue

            current = chunks[i]

            can_merge_forward = (
                i + 1 < len(chunks)
                and len(current.text) < self.config.min_chunk_size
                and current.heading_path == chunks[i + 1].heading_path
                and self._is_semantically_compatible(current.content_type, chunks[i + 1].content_type)
                and len(current.text) + len(chunks[i + 1].text) + 2 <= self.config.max_chunk_size
            )

            if can_merge_forward:
                next_chunk = chunks[i + 1]
                combined_text = f"{current.text}\n\n{next_chunk.text}"
                merged_chunk = DocumentChunk(
                    chunk_id=current.chunk_id,
                    document_id=current.document_id,
                    text=combined_text,
                    chunk_index=current.chunk_index,
                    heading_path=current.heading_path,
                    content_type="paragraph" if current.content_type == "paragraph" and next_chunk.content_type == "paragraph" else "section",
                    start_offset=None,
                    end_offset=None,
                    metadata={
                        **current.metadata,
                        "char_count": len(combined_text),
                        "estimated_tokens": max(1, len(combined_text) // 4),
                    },
                )
                merged.append(merged_chunk)
                skip_next = True
                continue

            can_merge_backward = (
                len(current.text) < self.config.min_chunk_size
                and bool(merged)
                and merged[-1].heading_path == current.heading_path
                and self._is_semantically_compatible(merged[-1].content_type, current.content_type)
                and len(merged[-1].text) + len(current.text) + 2 <= self.config.max_chunk_size
            )

            if can_merge_backward:
                prev = merged.pop()
                combined_text = f"{prev.text}\n\n{current.text}"
                merged_chunk = DocumentChunk(
                    chunk_id=prev.chunk_id,
                    document_id=prev.document_id,
                    text=combined_text,
                    chunk_index=prev.chunk_index,
                    heading_path=prev.heading_path,
                    content_type="paragraph" if prev.content_type == "paragraph" and current.content_type == "paragraph" else "section",
                    start_offset=None,
                    end_offset=None,
                    metadata={
                        **prev.metadata,
                        "char_count": len(combined_text),
                        "estimated_tokens": max(1, len(combined_text) // 4),
                    },
                )
                merged.append(merged_chunk)
            else:
                merged.append(current)

        return merged

    def _finalize_chunks(
        self,
        chunks: List[DocumentChunk],
        normalized_text: str,
        document_id: str,
    ) -> List[DocumentChunk]:
        """
        Assign exact normalized document offsets and sequential IDs.
        """
        finalized: List[DocumentChunk] = []
        search_start = 0

        for idx, chunk in enumerate(chunks, start=1):
            text_digest = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
            new_id = f"{document_id}_{idx:04d}_{text_digest}"
            chunk.chunk_index = idx
            chunk.chunk_id = new_id

            found_pos = normalized_text.find(chunk.text, search_start)
            if found_pos != -1:
                chunk.start_offset = found_pos
                chunk.end_offset = found_pos + len(chunk.text)
                search_start = found_pos + 1
            else:
                fallback_pos = normalized_text.find(chunk.text)
                if fallback_pos != -1:
                    chunk.start_offset = fallback_pos
                    chunk.end_offset = fallback_pos + len(chunk.text)
                    search_start = fallback_pos + 1

            finalized.append(chunk)

        return finalized

    @staticmethod
    def _is_semantically_compatible(type_a: str, type_b: str) -> bool:
        """Verify if two semantic unit types can be merged safely."""
        if type_a in ("notice", "faq", "definition") or type_b in ("notice", "faq", "definition"):
            return False
        return True
