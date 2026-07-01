from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from uuid import uuid5, NAMESPACE_URL

from llama_index.core.schema import BaseNode
from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class SearchResult:
    """Single retrieved source chunk."""

    point_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class QdrantIndex:
    """Manages a Qdrant collection for RAG chunks."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "rag_markdown_docs",
        vector_size: int = 384,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

    def collection_exists(self) -> bool:
        return self.client.collection_exists(
            collection_name=self.collection_name,
        )

    def create_collection(self) -> None:
        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )

    def recreate_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(
                collection_name=self.collection_name,
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )

    def upsert_nodes(
        self,
        nodes: Sequence[BaseNode],
        batch_size: int = 64,
    ) -> int:
        if not nodes:
            return 0

        self.create_collection()

        total_count = 0

        for start_index in range(0, len(nodes), batch_size):
            batch = nodes[start_index : start_index + batch_size]
            points = [self._node_to_point(node) for node in batch]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            total_count += len(points)

        return total_count

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
    ) -> list[SearchResult]:
        if not query_vector:
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self._scored_point_to_result(point)
            for point in response.points
        ]

    def count_points(self) -> int:
        response = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return int(response.count)

    def _node_to_point(self, node: BaseNode) -> models.PointStruct:
        if node.embedding is None:
            raise ValueError("Node has no embedding.")

        payload = self._build_payload(node)
        point_id = self._build_point_id(payload)

        return models.PointStruct(
            id=point_id,
            vector=node.embedding,
            payload=payload,
        )

    def _build_payload(self, node: BaseNode) -> dict[str, Any]:
        payload = dict(node.metadata)
        payload["text"] = node.get_content()

        return payload

    def _build_point_id(self, payload: dict[str, Any]) -> str:
        chunk_id = payload.get("chunk_id")

        if chunk_id:
            return str(uuid5(NAMESPACE_URL, str(chunk_id)))

        text = payload.get("text", "")
        file_path = payload.get("file_path", "")
        fallback_key = f"{file_path}:{text[:200]}"

        return str(uuid5(NAMESPACE_URL, fallback_key))

    def _scored_point_to_result(self, point: Any) -> SearchResult:
        payload = point.payload or {}

        text = str(payload.get("text", ""))
        metadata = {
            key: value
            for key, value in payload.items()
            if key != "text"
        }

        return SearchResult(
            point_id=str(point.id),
            score=float(point.score),
            text=text,
            metadata=metadata,
        )
