from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm import OpenRouterClient, PromptTemplates
from src.rag.retriever import RAGRetriever, RetrievedChunk
from src.rag.response_builder import RAGAnswer, RAGResponseBuilder
from src.rag.source_formatter import SourceFormatter


@dataclass(frozen=True)
class RAGQueryConfig:
    """Runtime settings for one RAG query."""

    top_k: int = 3
    temperature: float = 0.2
    max_tokens: int = 800


@dataclass(frozen=True)
class RAGQueryInput:
    """Input for the RAG query engine."""

    question: str
    config: RAGQueryConfig = RAGQueryConfig()


class RAGQueryEngine:
    """Runs retrieval and answer generation."""

    def __init__(
        self,
        retriever: RAGRetriever,
        llm_client: OpenRouterClient,
        response_builder: RAGResponseBuilder | None = None,
        source_formatter: SourceFormatter | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.response_builder = response_builder or RAGResponseBuilder()
        self.source_formatter = source_formatter or SourceFormatter()

    def query(self, query_input: RAGQueryInput) -> RAGAnswer:
        question = query_input.question.strip()

        if not question:
            return self.response_builder.empty_answer(
                question=query_input.question,
                message="Question is empty.",
            )

        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=query_input.config.top_k,
        )

        if not retrieval_result.chunks:
            return self.response_builder.empty_answer(
                question=question,
                message="No relevant context was found.",
            )

        context = self.source_formatter.format_context(
            chunks=retrieval_result.chunks,
        )

        prompt = PromptTemplates.build_rag_prompt(
            question=question,
            context=context,
        )

        llm_response = self.llm_client.generate_text(
            prompt=prompt.user_prompt,
            system_prompt=prompt.system_prompt,
            temperature=query_input.config.temperature,
            max_tokens=query_input.config.max_tokens,
        )

        return self.response_builder.build(
            question=question,
            answer=llm_response.text,
            chunks=retrieval_result.chunks,
            model=llm_response.model,
            usage=llm_response.usage,
            collection_name=retrieval_result.collection_name,
            embedding_model=retrieval_result.embedding_model,
        )
