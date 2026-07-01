from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_rag_query_engine
from app.schemas import AskRequest, AskResponse, HealthResponse, SourceChunk
from src.rag import RAGAnswer, RAGQueryConfig, RAGQueryEngine, RAGQueryInput


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    query_engine: RAGQueryEngine = Depends(get_rag_query_engine),
) -> AskResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    result = query_engine.query(
        RAGQueryInput(
            question=question,
            config=RAGQueryConfig(
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
        )
    )

    return _to_ask_response(result)


def _to_ask_response(result: RAGAnswer) -> AskResponse:
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceChunk(
                text=source.text,
                score=source.score,
                point_id=source.point_id,
                metadata=source.metadata,
            )
            for source in result.sources
        ],
        model=result.model,
        usage=result.usage,
        collection_name=result.collection_name,
        embedding_model=result.embedding_model,
    )
