from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_rag_query_engine, get_rag_retriever
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceChunk,
)
from src.llm import GroqGenerationError, GroqRateLimitError
from src.rag import RAGAnswer, RAGQueryConfig, RAGQueryEngine, RAGQueryInput, RAGRetriever


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

    try:
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

    except GroqRateLimitError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "groq_rate_limited",
                "message": str(error),
            },
        ) from error

    except GroqGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "groq_generation_failed",
                "message": str(error),
            },
        ) from error

    return _to_ask_response(result)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(
    request: RetrieveRequest,
    retriever: RAGRetriever = Depends(get_rag_retriever),
) -> RetrieveResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    result = retriever.retrieve(
        query=question,
        top_k=request.top_k,
    )

    return RetrieveResponse(
        question=result.query,
        sources=[
            SourceChunk(
                text=chunk.text,
                score=chunk.score,
                point_id=chunk.point_id,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ],
        collection_name=result.collection_name,
        embedding_model=result.embedding_model,
    )


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
