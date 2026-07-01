from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class RAGSource:
    """Source used for the generated answer."""

    text: str
    score: float
    point_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RAGAnswer:
    """Final RAG answer."""

    question: str
    answer: str
    sources: list[RAGSource] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    collection_name: str = ""
    embedding_model: str = ""


class RAGResponseBuilder:
    """Builds final RAG response objects."""

    def build(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
        model: str,
        usage: dict[str, Any],
        collection_name: str,
        embedding_model: str,
    ) -> RAGAnswer:
        sources = [
            self._chunk_to_source(chunk)
            for chunk in chunks
        ]

        return RAGAnswer(
            question=question,
            answer=answer,
            sources=sources,
            model=model,
            usage=usage,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

    def empty_answer(
        self,
        question: str,
        message: str,
    ) -> RAGAnswer:
        return RAGAnswer(
            question=question,
            answer=message,
            sources=[],
        )

    def _chunk_to_source(self, chunk: RetrievedChunk) -> RAGSource:
        return RAGSource(
            text=chunk.text,
            score=chunk.score,
            point_id=chunk.point_id,
            metadata=chunk.metadata,
        )
