from fastapi import APIRouter

from app.schemas import AskRequest, AskResponse


router = APIRouter()


class RAGRoutes:
    """API routes for the RAG service."""

    pass


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    return AskResponse()
