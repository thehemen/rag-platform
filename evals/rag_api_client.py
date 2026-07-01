from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass(frozen=True)
class RAGAPIResponse:
    """Response returned by the RAG API."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    collection_name: str = ""
    embedding_model: str = ""


class RAGAPIClientError(Exception):
    """Raised when the RAG API request fails."""


class RAGAPIClient:
    """Small client for the FastAPI RAG service."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        verbose: bool = True,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose

    def ask(
        self,
        question: str,
        top_k: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> RAGAPIResponse:
        """Calls full RAG generation endpoint."""

        payload = {
            "question": question,
            "top_k": top_k,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        endpoint = f"{self.api_url}/ask"

        self._print(f"[api] POST {endpoint}")
        self._print(f"[api] question={question!r}")

        data = self._post_json(
            endpoint=endpoint,
            payload=payload,
            error_prefix="RAG ask API returned an error",
        )

        self._print("[api] /ask response received")

        return RAGAPIResponse(
            question=str(data.get("question", question)),
            answer=str(data.get("answer", "")),
            sources=list(data.get("sources", [])),
            model=str(data.get("model", "")),
            usage=dict(data.get("usage", {})),
            collection_name=str(data.get("collection_name", "")),
            embedding_model=str(data.get("embedding_model", "")),
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> RAGAPIResponse:
        """Calls retrieval-only endpoint."""

        payload = {
            "question": question,
            "top_k": top_k,
        }

        endpoint = f"{self.api_url}/retrieve"

        self._print(f"[api] POST {endpoint}")
        self._print(f"[api] question={question!r}")

        data = self._post_json(
            endpoint=endpoint,
            payload=payload,
            error_prefix="RAG retrieve API returned an error",
        )

        self._print("[api] /retrieve response received")

        return RAGAPIResponse(
            question=str(data.get("question", question)),
            answer="",
            sources=list(data.get("sources", [])),
            model="",
            usage={},
            collection_name=str(data.get("collection_name", "")),
            embedding_model=str(data.get("embedding_model", "")),
        )

    def health(self) -> bool:
        endpoint = f"{self.api_url}/health"

        self._print(f"[api] GET {endpoint}")

        try:
            response = requests.get(
                endpoint,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RAGAPIClientError(
                f"RAG API health check failed: {error}"
            ) from error

        self._print("[api] health check ok")
        return True

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        error_prefix: str,
    ) -> dict[str, Any]:
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as error:
            raise RAGAPIClientError(
                f"{error_prefix}. Request timed out after {self.timeout} seconds."
            ) from error
        except requests.RequestException as error:
            raise RAGAPIClientError(
                f"{error_prefix}. Request failed: {error}"
            ) from error

        if response.status_code >= 400:
            raise RAGAPIClientError(
                f"{error_prefix}. "
                f"Status: {response.status_code}. "
                f"Body: {self._safe_response_text(response)}"
            )

        try:
            return response.json()
        except ValueError as error:
            raise RAGAPIClientError(
                f"{error_prefix}. Response is not valid JSON. "
                f"Body: {self._safe_response_text(response)}"
            ) from error

    def _safe_response_text(self, response: requests.Response) -> str:
        try:
            return response.text
        except Exception:
            return "<unreadable response>"

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)
