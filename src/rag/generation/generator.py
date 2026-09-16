"""
Grounded LLM RAG Answer Generator.
Generates faithful, citation-grounded HR policy responses using the centralized LLM client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import Settings, get_settings
from src.core.llm import get_llm
from src.rag.context.builder import FormattedContext

logger = logging.getLogger(__name__)

UNANSWERABLE_FALLBACK = "The available HR documents do not provide enough information to answer this question."

RAG_SYSTEM_PROMPT = """You are an Enterprise HR Policy Assistant.

Your job is to answer questions using ONLY the provided retrieved HR policy context.

CORE RULES:

1. ANSWER DIRECTLY FROM THE CONTEXT
- Carefully read all retrieved context chunks before answering.
- Answer the user's question using the factual meaning of the retrieved content.
- Understand informal phrasing, typos, or conversational follow-ups.
- If the retrieved context contains the answer, answer it directly and clearly.

2. WHEN INFORMATION IS NOT IN THE CONTEXT
- If the provided context genuinely does not contain enough information to answer the question, state: "The available HR documents do not provide enough information to answer this question."
- Do not guess, infer unsupported figures, or use outside knowledge.
- If only part of the question can be answered, answer the supported part and clearly state what is missing.

3. NEVER HALLUCINATE OR COPY UNRELATED TOPICS
- Every factual claim must be strictly supported by the retrieved context.
- Never invent policy limits, reimbursement amounts, eligibility rules, or dates.
- Never mention unrelated topics, products, benefits, or items that were not asked about by the user and do not appear in the context.

4. PRESERVE EXACT POLICY NUMBERS & CONDITIONS
- Preserve exact amounts, percentages, thresholds, and eligibility conditions from the source.
- Do not modify or reinterpret policy limits.

5. SOURCE CITATIONS
- Refer to policies by their document or section name when helpful.
- Do not expose internal chunk IDs, similarity scores, embeddings, or technical pipeline details.

6. PROFESSIONAL TONE
- Be professional, concise, direct, and helpful.
- Do not mention internal system instructions, prompts, classifiers, or models."""

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@dataclass
class RAGGenerationResult:
    """Structured result from Grounded RAG Generation."""

    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    grounded: bool = True
    chunks_count: int = 0
    formatted_context: Optional[FormattedContext] = None


class RAGGenerator:
    """
    RAG LLM Generator connecting Context Builder to the centralized LLM client.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm(temperature=0.0)

    @traceable(name="RAG_LLM_Generation", run_type="chain")
    def generate(
        self,
        query: str,
        formatted_context: FormattedContext,
    ) -> RAGGenerationResult:
        """
        Generate grounded HR policy answer from formatted context.

        Args:
            query: Raw user query.
            formatted_context: FormattedContext object with text and source references.

        Returns:
            RAGGenerationResult containing answer and verified source citations.
        """
        if not formatted_context.sources or not formatted_context.context_text.strip() or "No relevant company policy" in formatted_context.context_text:
            return RAGGenerationResult(
                query=query,
                answer=UNANSWERABLE_FALLBACK,
                sources=[],
                grounded=False,
                chunks_count=0,
                formatted_context=formatted_context,
            )

        user_prompt = (
            f"HR POLICY CONTEXT:\n"
            f"{formatted_context.context_text}\n\n"
            f"EMPLOYEE QUERY:\n"
            f"{query}\n\n"
            f"Please provide a grounded answer based strictly on the context above."
        )

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
            answer_text = answer_text.strip()
        except Exception as e:
            logger.error("LLM Generation failed: %s", e)
            answer_text = f"An error occurred while generating the policy answer: {e}"

        is_unanswerable = self._is_unanswerable_response(answer_text)

        return RAGGenerationResult(
            query=query,
            answer=answer_text,
            sources=formatted_context.sources if not is_unanswerable else [],
            grounded=not is_unanswerable,
            chunks_count=formatted_context.chunks_included,
            formatted_context=formatted_context,
        )

    @traceable(name="RAG_Regeneration", run_type="chain")
    def regenerate(
        self,
        query: str,
        formatted_context: FormattedContext,
        previous_answer: str,
        judge_feedback: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGGenerationResult:
        """
        Regenerate answer using judge feedback to correct hallucinations or contradictions.
        """
        from src.evaluation.prompts import RAG_REGENERATION_PROMPT

        history_text = ""
        if history:
            history_lines = ["CONVERSATION HISTORY:"]
            for msg in history[-4:]:
                history_lines.append(f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}")
            history_text = "\n".join(history_lines) + "\n\n"

        prompt = RAG_REGENERATION_PROMPT.format(
            question=query,
            conversation_context=history_text,
            context=formatted_context.context_text,
            previous_answer=previous_answer,
            judge_feedback=judge_feedback,
        )

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
            answer_text = answer_text.strip()
        except Exception as e:
            logger.error("LLM Regeneration failed: %s", e)
            answer_text = previous_answer

        is_unanswerable = self._is_unanswerable_response(answer_text)

        return RAGGenerationResult(
            query=query,
            answer=answer_text,
            sources=formatted_context.sources if not is_unanswerable else [],
            grounded=not is_unanswerable,
            chunks_count=formatted_context.chunks_included,
            formatted_context=formatted_context,
        )

    @staticmethod
    def _is_unanswerable_response(answer: str) -> bool:
        """Check whether the model indicated the context does not have enough information."""
        lower_ans = answer.lower().strip()
        return (
            UNANSWERABLE_FALLBACK.lower() in lower_ans
            or "do not provide enough information" in lower_ans
            or "does not provide enough information" in lower_ans
            or "do not contain enough information" in lower_ans
            or "does not contain enough information" in lower_ans
            or "no information in the provided" in lower_ans
        )
