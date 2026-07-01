from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embeddings import SentenceTransformerEmbedder
from src.vector_store import QdrantClientFactory, QdrantIndex, SearchResult


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved chunk returned by the RAG retriever."""

    text: str
    score: float
    metadata: dict[str, Any]
    point_id: str


@dataclass(frozen=True)
class RetrievalResult:
    """Full retrieval result for one user query."""

    query: str
    chunks: list[RetrievedChunk]
    collection_name: str
    embedding_model: str


class RAGRetriever:
    """Retrieves relevant chunks from Qdrant."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str | None = None,
        collection_name: str = "rag_markdown_docs",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_device: str | None = None,
        embedding_batch_size: int = 32,
        vector_size: int | None = None,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.embedding_batch_size = embedding_batch_size

        self.embedder = SentenceTransformerEmbedder(
            model_name=self.embedding_model,
            device=self.embedding_device,
            batch_size=self.embedding_batch_size,
            normalize_embeddings=True,
        )

        self.vector_size = vector_size or self.embedder.get_embedding_dimension()

        self.qdrant_index = self._create_qdrant_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult(
                query=query,
                chunks=[],
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
            )

        query_vector = self.embedder.embed_query(query)

        search_results = self.qdrant_index.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        chunks = [
            self._to_retrieved_chunk(result)
            for result in search_results
        ]

        return RetrievalResult(
            query=query,
            chunks=chunks,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )

    def _create_qdrant_index(self) -> QdrantIndex:
        client = QdrantClientFactory(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        ).create()

        return QdrantIndex(
            client=client,
            collection_name=self.collection_name,
            vector_size=self.vector_size,
        )

    def _to_retrieved_chunk(
        self,
        result: SearchResult,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            text=result.text,
            score=result.score,
            metadata=result.metadata,
            point_id=result.point_id,
        )
