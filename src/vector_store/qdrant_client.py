from __future__ import annotations

from qdrant_client import QdrantClient


class QdrantClientFactory:
    """Creates Qdrant client instances."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.timeout = timeout

    def create(self) -> QdrantClient:
        return QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
