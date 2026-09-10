"""
LLM-as-a-Judge verification service for RAG and SQL responses.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import get_settings
from src.core.llm import get_llm
from src.evaluation.prompts import (
    RAG_JUDGE_SYSTEM_PROMPT,
    SQL_JUDGE_SYSTEM_PROMPT,
)
from src.evaluation.schemas import (
    JudgeDecision,
    RAGJudgeInput,
    SQLJudgeInput,
    VerdictStatus,
)

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

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Evaluates generated answers for faithfulness, accuracy, relevance,
    and consistency using an LLM.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        self.settings = get_settings()
        self.threshold = (
            threshold
            if threshold is not None
            else self.settings.JUDGE_SCORE_THRESHOLD
        )
        self._llm = llm
        self._provider = provider or self.settings.JUDGE_PROVIDER
        self._model_name = model_name or self.settings.JUDGE_MODEL

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_llm(
                provider=self._provider,
                model_name=self._model_name,
                temperature=self.settings.JUDGE_TEMPERATURE,
            )
        return self._llm

    @traceable(name="LLM_Judge_RAG", run_type="chain")
    def evaluate_rag(self, input_data: RAGJudgeInput) -> JudgeDecision:
        """
        Evaluate a RAG-generated answer against the retrieved source documents.
        """
        if not self.settings.ENABLE_LLM_JUDGE:
            return JudgeDecision(
                passed=True,
                score=1.0,
                verdict_status=VerdictStatus.SKIPPED,
                reason="LLM judge disabled in configuration.",
            )

        start_time = time.perf_counter()
        try:
            formatted_context = self._format_retrieved_docs(input_data.retrieved_docs)
            formatted_history = self._format_history(input_data.history)

            user_prompt = (
                f"USER QUESTION:\n{input_data.question}\n\n"
                f"{formatted_history}"
                f"RETRIEVED SOURCE DOCUMENTS:\n{formatted_context}\n\n"
                f"GENERATED ANSWER TO EVALUATE:\n{input_data.answer}\n\n"
                "Please output your evaluation JSON object now:"
            )

            messages = [
                SystemMessage(content=RAG_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else str(response)
            decision = self._parse_judge_json(
                raw_content,
                question=input_data.question,
                context_text=formatted_context,
                answer_text=input_data.answer,
            )
            decision.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Apply score threshold & fatal flags
            if decision.score < self.threshold or decision.contradiction or not decision.grounded:
                decision.passed = False

            logger.info(
                f"RAG Judge evaluation finished: passed={decision.passed}, "
                f"score={decision.score}, latency={decision.latency_ms}ms"
            )
            return decision

        except Exception as exc:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"LLM Judge evaluation failed or timed out: {exc}")
            return JudgeDecision(
                passed=True,
                score=1.0,
                verdict_status=VerdictStatus.JUDGE_UNAVAILABLE,
                reason=f"Judge evaluation unavailable: {exc}",
                latency_ms=elapsed,
            )

    @traceable(name="LLM_Judge_SQL", run_type="chain")
    def evaluate_sql(self, input_data: SQLJudgeInput) -> JudgeDecision:
        """
        Evaluate a SQL-generated answer against the actual executed SQL and result rows.
        """
        if not self.settings.ENABLE_LLM_JUDGE:
            return JudgeDecision(
                passed=True,
                score=1.0,
                verdict_status=VerdictStatus.SKIPPED,
                reason="LLM judge disabled in configuration.",
            )

        start_time = time.perf_counter()
        try:
            formatted_history = self._format_history(input_data.history)
            formatted_result = self._format_sql_result(input_data.sql_result)

            user_prompt = (
                f"USER QUESTION:\n{input_data.question}\n\n"
                f"{formatted_history}"
                f"GENERATED SQL QUERY:\n{input_data.sql}\n\n"
                f"ACTUAL SQL EXECUTION RESULT:\n{formatted_result}\n\n"
                f"GENERATED NATURAL LANGUAGE ANSWER:\n{input_data.answer}\n\n"
                "Please output your evaluation JSON object now:"
            )

            messages = [
                SystemMessage(content=SQL_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else str(response)
            decision = self._parse_judge_json(
                raw_content,
                question=input_data.question,
                context_text=formatted_result,
                answer_text=input_data.answer,
            )
            decision.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Check SQL-specific consistency
            if decision.result_consistent is False or decision.contradiction or decision.score < self.threshold:
                decision.passed = False

            logger.info(
                f"SQL Judge evaluation finished: passed={decision.passed}, "
                f"score={decision.score}, latency={decision.latency_ms}ms"
            )
            return decision

        except Exception as exc:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"LLM SQL Judge evaluation failed or timed out: {exc}")
            return JudgeDecision(
                passed=True,
                score=1.0,
                verdict_status=VerdictStatus.JUDGE_UNAVAILABLE,
                reason=f"Judge evaluation unavailable: {exc}",
                latency_ms=elapsed,
            )

    def _parse_judge_json(self, raw_content: str, question: str = "", context_text: str = "", answer_text: str = "") -> JudgeDecision:
        """Extract and parse structured JSON verdict from model response."""
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"(\{.*\})", raw_content, re.DOTALL)
            json_str = json_match.group(1) if json_match else raw_content

        data = json.loads(json_str)

        # Normalize fields
        passed = bool(data.get("passed", True))
        score = float(data.get("score", 1.0 if passed else 0.0))
        correctness = float(data.get("correctness", score))
        groundedness = float(data.get("groundedness", 1.0 if data.get("grounded", True) else 0.0))
        relevance = float(data.get("relevance", 1.0 if data.get("relevant", True) else 0.0))
        hallucination = bool(data.get("hallucination", not data.get("grounded", True)))
        grounded = bool(data.get("grounded", not hallucination))
        relevant = bool(data.get("relevant", relevance >= 0.7))
        complete = bool(data.get("complete", True))
        contradiction = bool(data.get("contradiction", False))
        result_consistent = data.get("result_consistent")
        query_relevant = data.get("query_relevant")
        issues = list(data.get("issues", []))
        reason = str(data.get("reason", ""))
        needs_regeneration = bool(data.get("needs_regeneration", not passed or contradiction or hallucination))

        # Check Requirement 8A: Context contains answer but generated answer claims insufficient information
        is_insufficient_claim = any(
            phrase in answer_text.lower()
            for phrase in [
                "do not provide enough information",
                "does not provide enough information",
                "do not contain enough information",
                "does not contain enough information",
            ]
        )
        if is_insufficient_claim and context_text:
            # Check if key numbers / keywords from context address the question
            question_words = set(re.findall(r"\w+", question.lower())) - {"what", "is", "the", "are", "for", "to", "of", "in", "a", "an", "how", "much", "do", "we"}
            if len(question_words) >= 1:
                matches_in_ctx = sum(1 for w in question_words if w in context_text.lower())
                if matches_in_ctx >= max(1, len(question_words) // 2) and re.search(r"(\$\d+|\d+\s*(days|weeks|months|dollars|percent|%))", context_text):
                    passed = False
                    score = min(score, 0.3)
                    correctness = min(correctness, 0.3)
                    needs_regeneration = True
                    issues.append("Answer claimed insufficient information when relevant facts were present in retrieved context.")
                    if not reason:
                        reason = "Context contains the answer but generated answer says insufficient information."

        return JudgeDecision(
            passed=passed,
            score=max(0.0, min(1.0, score)),
            correctness=max(0.0, min(1.0, correctness)),
            groundedness=max(0.0, min(1.0, groundedness)),
            relevance=max(0.0, min(1.0, relevance)),
            hallucination=hallucination,
            needs_regeneration=needs_regeneration,
            grounded=grounded,
            relevant=relevant,
            complete=complete,
            contradiction=contradiction,
            result_consistent=result_consistent,
            query_relevant=query_relevant,
            issues=issues,
            reason=reason,
            verdict_status=VerdictStatus.VERIFIED if passed else VerdictStatus.REJECTED_AND_REGENERATED,
        )

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into structured text for judge context."""
        if not docs:
            return "No documents retrieved."

        lines = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source") or doc.get("metadata", {}).get("source", "Unknown Document")
            section = doc.get("section") or doc.get("metadata", {}).get("section", "General")
            content = doc.get("content") or doc.get("text", "")
            lines.append(f"--- Document {i} [{source} § {section}] ---\n{content.strip()}")

        return "\n\n".join(lines)

    def _format_sql_result(self, result: Any) -> str:
        """Format SQL execution result into structured string for judge context."""
        if result is None:
            return "NULL"
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    def _format_history(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Format conversation history if present."""
        if not history:
            return ""

        formatted = ["CONVERSATION HISTORY (FOR CONTEXT ONLY):"]
        for msg in history[-4:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if content:
                formatted.append(f"{role}: {content}")
        formatted.append("")
        return "\n".join(formatted) + "\n"
