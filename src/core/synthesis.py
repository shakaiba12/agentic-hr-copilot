"""
Answer Synthesis and Conversational Response Generation Module for PeopleQuery AI.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.llm import get_llm
from src.core.router import RouteCategory
from src.core.routing import DEFAULT_FALLBACK_ANSWER, STATIC_CATEGORY_RESPONSES

logger = logging.getLogger(__name__)


class HybridSynthesizer:
    """Synthesizes structured SQL database facts with unstructured RAG policy context."""

    def __init__(self, temperature: float = 0.0) -> None:
        self.temperature = temperature

    def _handle_both_failed(
        self,
        query: str,
        data: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        sql_msg: str,
        rag_msg: str,
    ) -> str:
        return (
            "Unable to complete the hybrid query: both the database query and policy documentation "
            "search encountered an error. Please refine your query or try again later."
        )

    def _handle_sql_failed(
        self,
        query: str,
        data: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        sql_msg: str,
        rag_msg: str,
    ) -> str:
        fallback = rag_msg or (
            "\n".join(c.get("text", "") for c in chunks[:2])
            if chunks
            else "Policy documentation was searched, but no specific matching sections were found."
        )
        return f"{fallback}\n\n*(Note: Employee database records could not be retrieved due to a query error.)*"

    def _handle_rag_failed(
        self,
        query: str,
        data: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        sql_msg: str,
        rag_msg: str,
    ) -> str:
        sql_summary = sql_msg or (
            f"Retrieved {len(data)} matching employee records from the database."
            if data
            else "No matching employee records were found in the database."
        )
        return f"{sql_summary}\n\n*(Note: Company policy documentation search encountered an error.)*"

    def synthesize(
        self,
        query: str,
        sql_data: Optional[List[Dict[str, Any]]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        sql_success: bool = True,
        rag_success: bool = True,
        sql_msg: str = "",
        rag_msg: str = "",
        judge_feedback: Optional[str] = None,
    ) -> str:
        """
        Combine SQL tabular data and RAG context using LLM or structured fallback.
        Handles partial failure modes gracefully and distinguishes zero-row results from query failures.
        """
        data = sql_data or []
        chunks = retrieved_chunks or []

        # Partial failure dispatch handlers: (sql_success, rag_success)
        partial_failure_handlers = {
            (False, False): self._handle_both_failed,
            (False, True): self._handle_sql_failed,
            (True, False): self._handle_rag_failed,
        }

        failure_handler = partial_failure_handlers.get((sql_success, rag_success))
        if failure_handler is not None:
            return failure_handler(query, data, chunks, sql_msg, rag_msg)

        # Both succeeded: Full Synthesis
        if chunks or data:
            try:
                llm_client = get_llm(temperature=self.temperature)
                chunks_text = "\n".join(
                    f"[{c.get('source', 'Doc')} § {c.get('section', 'General')}]: {c.get('text', '')}"
                    for c in chunks[:4]
                ) if chunks else "No qualitative policy documents retrieved."

                sql_context_str = (
                    f"DATABASE RESULTS ({len(data)} records):\n{data[:20]}"
                    if data
                    else "DATABASE RESULTS: 0 rows returned (Query executed successfully with no matching records)."
                )

                feedback_instruction = (
                    f"\nPREVIOUS EVALUATION FEEDBACK TO ADDRESS:\n{judge_feedback}\n"
                    if judge_feedback
                    else ""
                )

                prompt = (
                    "You are PeopleQuery AI, an Enterprise HR Intelligence Copilot.\n"
                    "Answer the user's question by combining the structured DATABASE FACTS and qualitative POLICY EVIDENCE below.\n\n"
                    "Strict Rules:\n"
                    "1. Ground all numbers, counts, salaries, dates, and employee names strictly in the DATABASE RESULTS. Do not invent figures or names.\n"
                    "2. Ground all rules, eligibility criteria, and procedures strictly in the POLICY EVIDENCE.\n"
                    "3. If database results returned 0 rows, clearly state that no matching records exist.\n"
                    "4. If policy evidence does not cover the question, clearly state that.\n"
                    "5. Keep citations [source#section] intact where applicable.\n"
                    f"{feedback_instruction}\n"
                    f"User Question: \"{query}\"\n\n"
                    f"{sql_context_str}\n\n"
                    f"POLICY EVIDENCE:\n{chunks_text}\n\n"
                    "Synthesized Answer:"
                )
                res = llm_client.invoke(prompt)
                raw_c = res.content if hasattr(res, "content") else str(res)
                synth_text = "".join(
                    chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk)
                    for chunk in raw_c
                ) if isinstance(raw_c, list) else str(raw_c)

                if synth_text.strip():
                    return synth_text.strip()
            except Exception as e:
                logger.debug("Hybrid LLM synthesis fallback: %s", e)

        # Deterministic combination fallback if LLM synthesis is unavailable
        if sql_msg and rag_msg:
            return f"{sql_msg}\n\n{rag_msg}"
        return sql_msg or rag_msg or "No matching database records or policy documents were found for this query."

    def regenerate(
        self,
        query: str,
        sql_data: Optional[List[Dict[str, Any]]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        sql_success: bool = True,
        rag_success: bool = True,
        sql_msg: str = "",
        rag_msg: str = "",
        judge_feedback: str = "",
    ) -> str:
        """Regenerate hybrid response with judge critique guidance."""
        return self.synthesize(
            query=query,
            sql_data=sql_data,
            retrieved_chunks=retrieved_chunks,
            sql_success=sql_success,
            rag_success=rag_success,
            sql_msg=sql_msg,
            rag_msg=rag_msg,
            judge_feedback=judge_feedback,
        )


class ConversationalResponseHandler:
    """Generates polite, concise conversational greetings, out-of-scope, and general responses."""

    def __init__(self, temperature: float = 0.3) -> None:
        self.temperature = temperature

    def respond(self, query: str, category: Optional[RouteCategory] = None) -> str:
        """Generate dynamic or static conversational responses."""
        cat = category or RouteCategory.CASUAL

        # Attempt dynamic warm conversational response
        try:
            llm_client = get_llm(temperature=self.temperature)
            prompt = (
                f"You are PeopleQuery AI, an Enterprise HR Intelligence Copilot. "
                f"You specialize in company HR policies, employee records, benefits, and workplace guidelines.\n\n"
                f"User Query: \"{query}\"\n\n"
                f"Please provide a warm, helpful, and concise response to the user."
            )
            llm_res = llm_client.invoke(prompt)
            answer_text = (llm_res.content if hasattr(llm_res, "content") else str(llm_res)).strip()
            if answer_text:
                return answer_text
        except Exception as e:
            logger.debug("Conversational response generation fallback: %s", e)

        if cat in STATIC_CATEGORY_RESPONSES:
            return STATIC_CATEGORY_RESPONSES[cat]

        if cat == RouteCategory.GENERAL:
            return (
                f"This question was handled as a general inquiry: '{query}'. "
                "I specialize in enterprise HR policies and company data."
            )

        return DEFAULT_FALLBACK_ANSWER
