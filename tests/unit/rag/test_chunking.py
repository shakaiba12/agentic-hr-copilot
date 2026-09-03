"""Unit tests for StructureAwareChunker (Step 1).
Comprehensive test matrix covering all requirements, edge cases, and regressions.
"""

import pytest
from src.rag.chunking import ChunkingConfig, DocumentChunk, StructureAwareChunker


@pytest.fixture
def chunker() -> StructureAwareChunker:
    config = ChunkingConfig(max_chunk_size=1200, min_chunk_size=150, overlap=120)
    return StructureAwareChunker(config=config)


class TestStructureAwareChunkerBasics:
    def test_empty_document(self, chunker: StructureAwareChunker):
        assert chunker.chunk_document("") == []
        assert chunker.chunk_document("   \n\n\t  ") == []

    def test_one_paragraph(self, chunker: StructureAwareChunker):
        text = "All US-based full-time employees and their dependents are eligible for health benefits through Sourcegraph."
        chunks = chunker.chunk_document(text, "doc1")
        assert len(chunks) == 1
        assert "All US-based full-time employees" in chunks[0].text
        assert chunks[0].chunk_index == 1

    def test_multiple_paragraphs(self, chunker: StructureAwareChunker):
        text = (
            "Sourcegraph offers comprehensive health benefits to all teammates.\n\n"
            "Benefits become effective on your date of hire and coverage continues through your employment."
        )
        chunks = chunker.chunk_document(text, "doc2")
        assert len(chunks) >= 1
        assert "Sourcegraph offers" in chunks[0].text
        assert "Benefits become effective" in chunks[0].text


class TestHeadingHierarchy:
    def test_h1_h2_h3_hierarchy(self, chunker: StructureAwareChunker):
        text = (
            "# Health Benefits\n\n"
            "## Eligibility\n\n"
            "### Dependents\n\n"
            "Your unmarried dependent children up to age 26 are eligible."
        )
        chunks = chunker.chunk_document(text, "doc3")
        assert len(chunks) >= 1
        assert chunks[0].heading_path == ["Health Benefits", "Eligibility", "Dependents"]
        assert chunks[0].metadata["section"] == "Dependents"

    def test_complex_heading_transitions(self, chunker: StructureAwareChunker):
        text = (
            "# Company\n\n"
            "Company overview text.\n\n"
            "## Benefits\n\n"
            "Benefits intro.\n\n"
            "### Health Insurance\n\n"
            "Insurance details.\n\n"
            "## Leave Policy\n\n"
            "Leave details.\n\n"
            "# Security\n\n"
            "Security rules."
        )
        chunks = chunker.chunk_document(text, "doc_transitions")
        paths = [c.heading_path for c in chunks]
        assert ["Company"] in paths
        assert ["Company", "Benefits"] in paths
        assert ["Company", "Benefits", "Health Insurance"] in paths
        assert ["Company", "Leave Policy"] in paths
        assert ["Security"] in paths

    def test_jump_and_retreat_heading_levels(self, chunker: StructureAwareChunker):
        text = (
            "## Level 2\n\n"
            "L2 text.\n\n"
            "#### Level 4\n\n"
            "L4 text.\n\n"
            "## Back to Level 2\n\n"
            "Back text."
        )
        chunks = chunker.chunk_document(text, "doc_jumps")
        assert chunks[0].heading_path == ["Level 2"]
        assert chunks[1].heading_path == ["Level 2", "Level 4"]
        assert chunks[2].heading_path == ["Back to Level 2"]

    def test_heading_with_formatting_and_unicode(self, chunker: StructureAwareChunker):
        text = (
            "# **Code of Conduct** & [Ethics Guide](https://example.com)\n\n"
            "## Section 1: ☕ Café & Global Teammates\n\n"
            "Global guidelines."
        )
        chunks = chunker.chunk_document(text, "doc_fmt")
        assert chunks[0].heading_path == [
            "Code of Conduct & Ethics Guide",
            "Section 1: ☕ Café & Global Teammates",
        ]


class TestSentenceSplittingAndAbbreviations:
    def test_protects_abbreviations_and_decimals(self, chunker: StructureAwareChunker):
        text = (
            "We offer benefits in the U.S. and U.K. for all teammates. "
            "For example, e.g. medical coverage is 100% covered. "
            "Version 2.1 was released on 2024-01-01 for $1,200.50 monthly budget. "
            "Employees are eligible immediately."
        )
        sentences = chunker._split_into_sentences(text)
        # Verify U.S., e.g., Version 2.1, $1,200.50 were not prematurely split
        assert not any(s.strip() in ("U.S.", "e.g.", "Version 2.1", "$1,200.50") for s in sentences)
        assert len(sentences) >= 2


class TestChunkSizeHardGuarantee:
    def test_huge_single_sentence_strictly_respects_max_chunk_size(self):
        small_config = ChunkingConfig(max_chunk_size=200, min_chunk_size=50, overlap=30)
        chunker = StructureAwareChunker(config=small_config)
        huge_sentence = (
            "This is a single excessively long policy sentence that contains no punctuation whatsoever "
            "and just goes on and on with continuous words to test word level splitting fallback behavior thoroughly "
            "across multiple size-constrained boundaries in the structure-aware chunking engine without ever failing."
        )
        chunks = chunker.chunk_document(huge_sentence, "doc_huge_sentence")
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 200

    def test_pathological_single_word_unbreakable(self):
        small_config = ChunkingConfig(max_chunk_size=100, min_chunk_size=20, overlap=10)
        chunker = StructureAwareChunker(config=small_config)
        unbreakable_word = "A" * 350
        chunks = chunker.chunk_document(unbreakable_word, "doc_unbreakable")
        assert len(chunks) == 4
        for c in chunks:
            assert len(c.text) <= 100


class TestSemanticUnitsAndFAQ:
    def test_faq_question_and_answer_kept_together(self, chunker: StructureAwareChunker):
        text = (
            "## FAQ\n\n"
            "**What:** Open Enrollment is your once-a-year opportunity.\n\n"
            "Outside open enrollment, changes require a qualifying life event."
        )
        chunks = chunker.chunk_document(text, "doc_faq")
        assert len(chunks) == 1
        assert "**What:**" in chunks[0].text
        assert "qualifying life event" in chunks[0].text
        assert chunks[0].content_type == "faq"

    def test_prose_not_misclassified_as_faq(self, chunker: StructureAwareChunker):
        text = (
            "## Project Planning\n\n"
            "What we need to achieve is high reliability across all services.\n\n"
            "Next steps involve testing the system thoroughly."
        )
        chunks = chunker.chunk_document(text, "doc_prose")
        # Should be classified as paragraph, not faq
        assert chunks[0].content_type == "paragraph"

    def test_notices_and_definitions(self, chunker: StructureAwareChunker):
        text = (
            "## Important Notice\n\n"
            "**Important:** Employees must submit requests 30 days in advance.\n\n"
            "QLE: Qualifying Life Event."
        )
        chunks = chunker.chunk_document(text, "doc_notice")
        types = [c.content_type for c in chunks]
        assert "notice" in types or "definition" in types


class TestSmallChunkMergeRestraint:
    def test_merges_paragraphs_under_same_heading(self, chunker: StructureAwareChunker):
        text = (
            "## Overview\n\n"
            "Short paragraph 1.\n\n"
            "Short paragraph 2."
        )
        chunks = chunker.chunk_document(text, "doc_merge")
        assert len(chunks) == 1
        assert "Short paragraph 1." in chunks[0].text
        assert "Short paragraph 2." in chunks[0].text

    def test_does_not_merge_notice_with_unrelated_list(self, chunker: StructureAwareChunker):
        text = (
            "## Rules\n\n"
            "**Warning:** Do not bypass safety controls.\n\n"
            "- Rule 1\n"
            "- Rule 2\n"
            "- Rule 3"
        )
        chunks = chunker.chunk_document(text, "doc_no_merge")
        assert len(chunks) >= 2


class TestOffsetsAndDeterminism:
    def test_offsets_are_exact_in_normalized_text(self, chunker: StructureAwareChunker):
        text = (
            "# Benefits\n\n"
            "Sourcegraph covers 100% of healthcare premiums.\n\n"
            "## Dental\n\n"
            "Dental care is provided through Cigna DPPO."
        )
        norm = chunker.filter.normalize(text)
        chunks = chunker.chunk_document(text, "doc_offsets")
        for c in chunks:
            assert c.start_offset is not None
            assert c.end_offset is not None
            assert c.start_offset <= c.end_offset
            assert norm[c.start_offset : c.end_offset] == c.text

    def test_duplicate_paragraphs_have_distinct_offsets(self, chunker: StructureAwareChunker):
        text = (
            "# HR Contacts\n\n"
            "For questions, contact hr@sourcegraph.com.\n\n"
            "## Benefits\n\n"
            "Review your benefits annually.\n\n"
            "## Expenses\n\n"
            "For questions, contact hr@sourcegraph.com."
        )
        norm = chunker.filter.normalize(text)
        chunks = chunker.chunk_document(text, "doc_dup")
        hr_chunks = [c for c in chunks if "contact hr@sourcegraph.com" in c.text]
        assert len(hr_chunks) == 2
        assert hr_chunks[0].start_offset != hr_chunks[1].start_offset
        assert norm[hr_chunks[0].start_offset : hr_chunks[0].end_offset] == hr_chunks[0].text
        assert norm[hr_chunks[1].start_offset : hr_chunks[1].end_offset] == hr_chunks[1].text

    def test_determinism_across_multiple_runs(self, chunker: StructureAwareChunker):
        text = (
            "# Policies\n\n"
            "Paragraph one is descriptive.\n\n"
            "- Bullet A\n"
            "- Bullet B\n\n"
            "**Note:** Final policy note."
        )
        chunks1 = chunker.chunk_document(text, "doc_det")
        chunks2 = chunker.chunk_document(text, "doc_det")
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text
            assert c1.heading_path == c2.heading_path
            assert c1.start_offset == c2.start_offset
            assert c1.end_offset == c2.end_offset


class TestEmptyHeadingsAndTransitions:
    def test_empty_heading_does_not_create_useless_chunk_or_contaminate(self, chunker: StructureAwareChunker):
        text = (
            "# Spending company money\n\n"
            "## Desk set-up\n\n"
            "### Full-time teammates\n\n"
            "We expect new teammates to spend up to $2,000 for desk setup.\n\n"
            "#### Team-specific totals\n\n"
            "## Travel and meals\n\n"
            "Travel rules apply to all business trips."
        )
        chunks = chunker.chunk_document(text, "doc_empty_heading")
        # Ensure no chunk contains only "#### Team-specific totals"
        assert not any(c.text.strip() == "#### Team-specific totals" for c in chunks)
        # Check that Travel section is not contaminated with Team-specific totals
        travel_chunk = next(c for c in chunks if "Travel rules" in c.text)
        assert travel_chunk.heading_path == ["Spending company money", "Travel and meals"]


class TestRealWorldHREdgeCases:
    def test_long_paragraph_exceeding_max_chunk_size(self, chunker: StructureAwareChunker):
        """Case 1: Long paragraph > 1200 chars splitting safely across sentence boundaries."""
        sentences = [
            f"Sentence number {i} describes policy provision regarding employee eligibility, coverage details, and compliance requirements in detail."
            for i in range(1, 16)
        ]
        long_paragraph = " ".join(sentences)
        assert len(long_paragraph) > 1500

        chunks = chunker.chunk_document(long_paragraph, "doc_long_p")
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 1200
            assert c.content_type == "paragraph"

    def test_long_list_exceeding_max_chunk_size(self, chunker: StructureAwareChunker):
        """Case 2: Long list > 1200 chars splitting safely between list items."""
        items = [
            f"- Rule item {i}: Teammates must adhere strictly to compliance standard {i} when handling expense reports, receipts, and invoices."
            for i in range(1, 25)
        ]
        long_list = "\n".join(items)
        assert len(long_list) > 2000

        chunks = chunker.chunk_document(long_list, "doc_long_list")
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 1200
            assert c.content_type == "list"
            # Verify list item boundaries preserved
            assert c.text.startswith("- Rule item")

    def test_faq_with_long_answer(self, chunker: StructureAwareChunker):
        """Case 3: FAQ Question + long answer exceeding max_chunk_size."""
        question = "**What:** How does the international parental leave policy work for remote employees?"
        answer_sentences = [
            f"Section {i} explains statutory requirements, local social security offsets, and base salary top-ups for non-US teammates."
            for i in range(1, 15)
        ]
        faq_text = f"## FAQ\n\n{question}\n\n" + " ".join(answer_sentences)
        chunks = chunker.chunk_document(faq_text, "doc_faq_long")
        assert len(chunks) >= 2
        assert "**What:**" in chunks[0].text
        assert chunks[0].heading_path == ["FAQ"]

    def test_document_with_many_headings_and_nested_lists(self, chunker: StructureAwareChunker):
        """Case 5: Document with many headings + nested lists."""
        doc = (
            "# Engineering Handbook\n\n"
            "Welcome to the team.\n\n"
            "## Onboarding\n\n"
            "### Hardware & Equipment\n\n"
            "- Laptop Setup\n"
            "  - Request MacBook Pro 16\n"
            "  - Setup YubiKey\n"
            "- Home Office Setup\n"
            "  - Desk and ergonomic chair\n"
            "  - External 4K monitor\n\n"
            "### Software Access\n\n"
            "- GitHub Enterprise\n"
            "- AWS Staging & Prod\n"
            "- Slack workspace\n\n"
            "## Work Guidelines\n\n"
            "### Code Review\n\n"
            "All PRs require at least one approving review."
        )
        chunks = chunker.chunk_document(doc, "doc_complex")
        assert len(chunks) >= 3
        # Verify nested heading paths
        paths = [c.heading_path for c in chunks]
        assert ["Engineering Handbook", "Onboarding", "Hardware & Equipment"] in paths
        assert ["Engineering Handbook", "Onboarding", "Software Access"] in paths
        assert ["Engineering Handbook", "Work Guidelines", "Code Review"] in paths

    def test_home_office_and_stipends_chunking(self, chunker: StructureAwareChunker):
        sample_doc = (
            "# Spending company money\n\n"
            "We want to make sure you have what you need to be productive and happy in your role at Sourcegraph.\n\n"
            "- You are free to purchase items and expense them without asking permission if it is in company interest.\n"
            "- If you spend more than $1,000 on any physical item, it will be considered property of the company.\n"
            "- Please file an expense report within 60 days of an eligible purchase.\n\n"
            "## Desk set-up\n\n"
            "### Full-time teammates\n\n"
            "We expect new teammates to spend up to **$2,000** for desk setup.\n\n"
            "**Note:** Contact people-ops@sourcegraph.com for approval on items exceeding $2,000."
        )
        chunks = chunker.chunk_document(sample_doc, "stipends_doc")
        assert len(chunks) >= 3
    def test_heading_only_document_produces_no_meaningless_chunks(self, chunker: StructureAwareChunker):
        """Case 3: Heading-only document should not produce empty chunks."""
        text = "# Benefits\n\n## Healthcare\n\n### Dental"
        chunks = chunker.chunk_document(text, "doc_headings_only")
        assert chunks == []


class TestContentPreservationAndEntities:
    """Verifies that URLs, emails, dates, policy numbers, currency, and Unicode entities are preserved."""

    def test_preserves_urls_and_emails(self, chunker: StructureAwareChunker):
        text = (
            "## Contact Support\n\n"
            "For inquiries, visit https://sourcegraph.com/handbook/people-ops and email people-ops@sourcegraph.com directly."
        )
        chunks = chunker.chunk_document(text, "doc_entities")
        assert len(chunks) == 1
        assert "https://sourcegraph.com/handbook/people-ops" in chunks[0].text
        assert "people-ops@sourcegraph.com" in chunks[0].text

    def test_preserves_dates_currency_policy_numbers(self, chunker: StructureAwareChunker):
        text = (
            "## Policy Revision\n\n"
            "Under Policy HR-042, effective January 15, 2026, teammates receive a $1,500 annual stipend and 20 days PTO."
        )
        chunks = chunker.chunk_document(text, "doc_dates")
        assert len(chunks) == 1
        assert "Policy HR-042" in chunks[0].text
        assert "January 15, 2026" in chunks[0].text
        assert "$1,500" in chunks[0].text
        assert "20 days" in chunks[0].text

    def test_unicode_accents_emojis_and_non_latin(self, chunker: StructureAwareChunker):
        text = (
            "## International Teammates 🌍\n\n"
            "We support teammates in São Paulo, München, and Tokyo (東京). "
            "Café stipends are provided: €500 / ¥75,000 monthly."
        )
        chunks = chunker.chunk_document(text, "doc_unicode")
        assert len(chunks) == 1
        assert "São Paulo" in chunks[0].text
        assert "München" in chunks[0].text
        assert "東京" in chunks[0].text
        assert "€500" in chunks[0].text
        assert "¥75,000" in chunks[0].text


class TestMalformedMarkdownAndEdgeFallbacks:
    """Verifies safe handling of malformed formatting, oversized list items, and configuration validation."""

    def test_malformed_markdown_does_not_crash(self, chunker: StructureAwareChunker):
        text = (
            "# Benefits\n\n"
            "- item one\n"
            "- item two\n\n"
            "**unfinished formatting and unclosed bold tags"
        )
        chunks = chunker.chunk_document(text, "doc_malformed")
        assert len(chunks) >= 1
        assert "item one" in chunks[0].text

    def test_oversized_individual_list_item_splits_safely(self, chunker: StructureAwareChunker):
        """Case 7: Single list item > 1200 chars."""
        single_item = "- " + "Extremely long compliance requirement without list break. " * 35
        assert len(single_item) > 1500

        chunks = chunker.chunk_document(single_item, "doc_huge_item")
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 1200

    def test_invalid_config_raises_value_error(self):
        """Validates that invalid overlap or chunk size configurations raise ValueError."""
        with pytest.raises(ValueError):
            ChunkingConfig(max_chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            ChunkingConfig(max_chunk_size=100, overlap=150)
        with pytest.raises(ValueError):
            ChunkingConfig(max_chunk_size=0)



