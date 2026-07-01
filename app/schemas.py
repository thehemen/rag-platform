from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request schema for RAG questions."""

    question: str = Field(default="")
    top_k: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=1, le=4096)


class SourceChunk(BaseModel):
    """Retrieved source chunk."""

    text: str = Field(default="")
    score: float = Field(default=0.0)
    point_id: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Response schema for RAG answers."""

    question: str = Field(default="")
    answer: str = Field(default="")
    sources: list[SourceChunk] = Field(default_factory=list)
    model: str = Field(default="")
    usage: dict[str, Any] = Field(default_factory=dict)
    collection_name: str = Field(default="")
    embedding_model: str = Field(default="")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class RetrieveRequest(BaseModel):
    """Request schema for retrieval-only search."""

    question: str = Field(default="")
    top_k: int = Field(default=3, ge=1, le=20)


class RetrieveResponse(BaseModel):
    """Response schema for retrieval-only search."""

    question: str = Field(default="")
    sources: list[SourceChunk] = Field(default_factory=list)
    collection_name: str = Field(default="")
    embedding_model: str = Field(default="")
