from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request schema for user questions."""

    question: str = Field(default="")
    top_k: int = Field(default=3)


class SourceChunk(BaseModel):
    """Retrieved source chunk metadata."""

    file_name: str = Field(default="")
    text: str = Field(default="")
    score: float = Field(default=0.0)


class AskResponse(BaseModel):
    """Response schema for generated answers."""

    answer: str = Field(default="")
    sources: list[SourceChunk] = Field(default_factory=list)
