"""
Unit and E2E Tests for Context Builder, Grounded RAG Generator, Faithfulness, and Edge Cases (Phases 6-10).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.rag.context_builder import ContextBuilder, FormattedContext
from src.rag.generator import RAGGenerationResult, RAGGenerator, UNANSWERABLE_FALLBACK
from src.rag.pipeline import RAGPipeline, RAGPipelineResult
from src.rag.reranking.reranker import RerankedResult


class TestContextBuilder:
    """Unit tests for Phase 6 ContextBuilder."""

    def test_build_context_with_valid_chunks(self):
        builder = ContextBuilder(max_chars=4000, max_chunks=3)
        chunks = [
            RerankedResult(
                id="doc1_c1",
                text="Employees receive 20 days PTO.",
                metadata={"document_id": "time-off", "heading_path": ["Time Off", "PTO"], "section": "PTO", "content_type": "paragraph"},
                retrieval_distance=0.2,
                reranker_score=0.9,
                final_rank=1,
            ),
            RerankedResult(
                id="doc1_c2",
                text="Submit PTO in Deel.",
                metadata={"document_id": "time-off", "heading_path": ["Time Off", "Deel"], "section": "Deel", "content_type": "paragraph"},
                retrieval_distance=0.3,
                reranker_score=0.8,
                final_rank=2,
            ),
        ]

        formatted = builder.build_context(chunks)

        assert formatted.chunks_included == 2
        assert "Time Off > PTO" in formatted.context_text
        assert "Employees receive 20 days PTO" in formatted.context_text
        assert len(formatted.sources) == 2
        assert formatted.sources[0]["chunk_id"] == "doc1_c1"
        assert formatted.sources[1]["chunk_id"] == "doc1_c2"

    def test_build_context_deduplication(self):
        builder = ContextBuilder(max_chars=4000, max_chunks=5)
        chunks = [
            RerankedResult(id="dup_1", text="Duplicate text", metadata={}, retrieval_distance=0.1, reranker_score=0.9, final_rank=1),
            RerankedResult(id="dup_1", text="Duplicate text", metadata={}, retrieval_distance=0.1, reranker_score=0.8, final_rank=2),
            RerankedResult(id="unique_1", text="Unique text", metadata={}, retrieval_distance=0.2, reranker_score=0.7, final_rank=3),
        ]

        formatted = builder.build_context(chunks)
        assert formatted.chunks_included == 2
        assert [s["chunk_id"] for s in formatted.sources] == ["dup_1", "unique_1"]

    def test_build_context_empty_chunks(self):
        builder = ContextBuilder()
        formatted = builder.build_context([])
        assert formatted.chunks_included == 0
        assert formatted.sources == []
        assert "No relevant company policy" in formatted.context_text


class TestRAGGenerator:
    """Unit tests for Phase 7 & 8 Grounded Generator & Citations."""

    def test_generate_answerable_query(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Employees receive 20 days of paid annual leave per year.\n\nSources:\n- time-off: Annual Leave"
        mock_llm.invoke.return_value = mock_response

        generator = RAGGenerator(llm=mock_llm)
        context = FormattedContext(
            context_text="--- [SOURCE 1] ---\nDocument: time-off\nSection: Annual Leave\n\nEmployees receive 20 days of annual leave.",
            sources=[{"document_id": "time-off", "section": "Annual Leave", "chunk_id": "c1"}],
            chunks_included=1,
        )

        result = generator.generate("How many annual leaves do employees receive?", formatted_context=context)

        assert result.grounded is True
        assert "20 days of paid annual leave" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0]["document_id"] == "time-off"

    def test_generate_unanswerable_query(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = UNANSWERABLE_FALLBACK
        mock_llm.invoke.return_value = mock_response

        generator = RAGGenerator(llm=mock_llm)
        context = FormattedContext(
            context_text="--- [SOURCE 1] ---\nDocument: time-off\n\nSome unrelated time off text.",
            sources=[{"document_id": "time-off", "section": "General", "chunk_id": "c1"}],
            chunks_included=1,
        )

        result = generator.generate("What is the company stock purchase plan discount in Antarctica?", formatted_context=context)

        assert result.grounded is False
        assert UNANSWERABLE_FALLBACK in result.answer
        assert result.sources == []

    def test_generate_empty_context_fallback(self):
        generator = RAGGenerator()
        empty_context = FormattedContext(context_text="", sources=[], chunks_included=0)
        result = generator.generate("Any question", formatted_context=empty_context)

        assert result.grounded is False
        assert UNANSWERABLE_FALLBACK in result.answer


class TestEndToEndRAGPipeline:
    """End-to-End integration tests for RAG pipeline across various queries."""

    def test_e2e_rag_pipeline_answerable_query(self):
        pipeline = RAGPipeline()
        query = "Who can be included as a dependent for health benefits?"
        result: RAGPipelineResult = pipeline.handle(query)

        assert result.success is True
        assert len(result.response) > 20
        # Must mention spouse, partner, or children
        assert any(w in result.response.lower() for w in ["spouse", "partner", "children", "dependent"])
        assert result.chunks_count > 0

    def test_e2e_rag_pipeline_unanswerable_query(self):
        pipeline = RAGPipeline()
        query = "What is the pet insurance policy reimbursement rate for penguins in Antarctica?"
        result: RAGPipelineResult = pipeline.handle(query)

        assert result.success is True
        # Must refuse or acknowledge insufficient info
        assert any(phrase in result.response.lower() for phrase in [
            "do not provide enough information",
            "not mentioned",
            "not provide",
            "not available",
            "no information",
        ])


class TestRAGPipelineOrchestrationMocked:
    """Unit tests for complete RAG orchestration and error handling with mocks."""

    def test_pipeline_empty_query_returns_error_result(self):
        pipeline = RAGPipeline()
        res = pipeline.handle("   ")
        assert res.success is False
        assert res.error == "Empty query"
        assert res.response == UNANSWERABLE_FALLBACK

    def test_pipeline_retriever_failure_handled_gracefully(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("Database connection timed out")

        pipeline = RAGPipeline(retriever=mock_retriever)
        res = pipeline.handle("What is the leave policy?")

        assert res.success is False
        assert "Database connection timed out" in (res.error or "")
        assert "error retrieving HR policy" in res.response

    def test_pipeline_full_mocked_success_flow(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RerankedResult(
                id="doc_1",
                text="16 weeks of paid parental leave.",
                metadata={"document_id": "parental-leave", "section": "Leave Duration"},
                retrieval_distance=0.1,
                reranker_score=0.95,
                final_rank=1,
            )
        ]

        mock_generator = MagicMock()
        mock_generator.generate.return_value = RAGGenerationResult(
            query="parental leave",
            answer="Employees receive 16 weeks of paid parental leave.\n\nSources:\n- parental-leave",
            sources=[{"document_id": "parental-leave", "section": "Leave Duration"}],
            grounded=True,
            chunks_count=1,
        )

        pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
        res = pipeline.handle("How much parental leave do employees receive?")

        assert res.success is True
        assert "16 weeks of paid parental leave" in res.response
        assert res.chunks_count == 1
        assert res.grounded is True
        assert len(res.sources) == 1

