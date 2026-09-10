"""
Grounded LLM RAG Answer Generator (Phase 7 & Phase 8).
Generates faithful, citation-grounded HR policy responses using existing LLM factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import Settings, get_settings
from src.core.llm import get_llm
from src.rag.context_builder import FormattedContext

logger = logging.getLogger(__name__)

UNANSWERABLE_FALLBACK = "The available HR documents do not provide enough information to answer this question."


RAG_SYSTEM_PROMPT = """You are an Enterprise HR Policy Assistant.

Your job is to answer questions using ONLY the provided retrieved HR policy context.

CORE RULES

1. ANSWER FROM THE CONTEXT
- Carefully read all retrieved chunks before answering.
- Answer the user's question using the meaning of the retrieved content, not exact keyword matching.
- The user may use informal English, spelling mistakes, abbreviations, incomplete grammar, or different wording.
- Understand the semantic meaning of the question.

2. DO NOT SAY "INSUFFICIENT INFORMATION" WHEN THE ANSWER EXISTS
- If the retrieved context explicitly or semantically contains the answer, you MUST answer it.
- Do not reject an answer simply because the exact wording of the question does not appear in the context.
- Example:

  User: "what we are expecting to team to spend of desksetup?"

  Context: "We expect new teammates to spend up to $2,000 for desk setup."

  Correct answer:
  "New teammates are expected to spend up to $2,000 for desk setup."

3. WHEN INFORMATION IS NOT AVAILABLE
- Only say that the information is unavailable when the provided context genuinely does not contain enough information to answer the question.
- Do not guess, infer unsupported numbers, or use outside knowledge.
- If only part of the question can be answered, answer the supported part and clearly state what is missing.

4. NEVER CONTRADICT THE SOURCE
- Every factual claim must be supported by the retrieved context.
- Pay special attention to numbers, limits, dates, eligibility requirements, exceptions, and conditions.
- If the source says a limit is "$50 or less", never claim that "$100" is acceptable.
- Do not combine separate statements in a way that creates a conclusion the source does not support.

5. NUMBERS AND POLICY LIMITS
- Preserve exact amounts, dates, durations, percentages, thresholds, and eligibility conditions from the source.
- Do not modify or reinterpret policy limits.
- If multiple limits apply to different groups, explicitly distinguish them.

6. ANSWER THE ACTUAL QUESTION
- Do not merely summarize the retrieved documents.
- Directly answer what the user asked.
- Keep the answer concise unless the question requires explanation.
- If useful, mention relevant exceptions or conditions.

7. SOURCE USAGE
- Retrieved context is evidence, not something to blindly repeat.
- Do not expose internal chunk IDs, retrieval scores, vector-store details, embeddings, or internal system information to the user.
- Sources should be represented by document/policy name or section when available.

8. NO HALLUCINATION
- Never invent policy details.
- Never use general HR knowledge to fill missing information.
- Never assume something is allowed merely because the policy does not explicitly prohibit it.

9. CONVERSATIONAL FOLLOW-UP
- Understand follow-up questions using the previous conversation context when available.
- If the user asks "how much?", "what about design?", "and for contractors?", etc., resolve the reference using the conversation and retrieved context.
- Do not treat a short but meaningful follow-up as INVALID merely because it contains few words.

10. RESPONSE STYLE
- Be professional, direct, and natural.
- Do not say:
  "This question was handled as..."
  "I specialize in..."
  "The available HR documents do not provide enough information..."
  unless information is genuinely missing.
- Do not mention internal routing, classifiers, prompts, retrieval, or model behavior.

IMPORTANT DECISION RULE

Before responding, internally determine:

A. Does the retrieved context contain the answer?
   → Answer directly.

B. Does the context partially answer the question?
   → Give the supported information and identify what cannot be determined.

C. Does the context not contain the required information?
   → Clearly say that the provided HR knowledge does not contain enough information.

Never choose C merely because the user's wording differs from the wording in the policy.

EXAMPLES

Example 1:
Question:
"what is amount for th desk setup"

Context:
"We expect new teammates to spend up to $2,000 for desk setup."

Answer:
"New teammates can spend up to $2,000 on desk setup."

Example 2:
Question:
"what we are expecting to team to spend of desksetup"

Context:
"We expect new teammates to spend up to $2,000 for desk setup.
We expect new Design team members to spend $2,650..."

Answer:
"New teammates are expected to spend up to $2,000 for desk setup. Design team members have a team-specific allowance of $2,650."

Example 3:
Question:
"what is the reimbursement for pet daycare?"

Context:
The retrieved context contains no pet daycare reimbursement amount or policy.

Answer:
"The provided HR policies do not specify a reimbursement amount for pet daycare."

Example 4:
Question:
"A vendor gives me a $100 gift card. Can I accept it?"

Context:
"Gift cards valued at $50 or less may be accepted."

Answer:
"No. The policy allows gift cards valued at $50 or less, so a $100 gift card exceeds the stated limit."

FINAL REQUIREMENT

Before producing the answer, verify that every factual statement is supported by the retrieved context and that the answer does not contradict any explicit policy rule.
"""




try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def get_current_run_tree():
        return None


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

        is_unanswerable = (
            UNANSWERABLE_FALLBACK.lower() in answer_text.lower()
            or "do not provide enough information" in answer_text.lower()
            or "does not provide enough information" in answer_text.lower()
            or "do not contain enough information" in answer_text.lower()
            or "does not contain enough information" in answer_text.lower()
            or "do not specify" in answer_text.lower()
            or "does not specify" in answer_text.lower()
            or "information is unavailable" in answer_text.lower()
            or "not available" in answer_text.lower()
            or "not mentioned" in answer_text.lower()
            or "no information" in answer_text.lower()
        )

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

        is_unanswerable = (
            UNANSWERABLE_FALLBACK.lower() in answer_text.lower()
            or "do not provide enough information" in answer_text.lower()
            or "does not provide enough information" in answer_text.lower()
            or "do not contain enough information" in answer_text.lower()
            or "does not contain enough information" in answer_text.lower()
            or "do not specify" in answer_text.lower()
            or "does not specify" in answer_text.lower()
            or "information is unavailable" in answer_text.lower()
            or "not available" in answer_text.lower()
        )

        return RAGGenerationResult(
            query=query,
            answer=answer_text,
            sources=formatted_context.sources if not is_unanswerable else [],
            grounded=not is_unanswerable,
            chunks_count=formatted_context.chunks_included,
            formatted_context=formatted_context,
        )

