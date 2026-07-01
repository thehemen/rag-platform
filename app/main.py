from fastapi import FastAPI

from app.api_routes import router


class RAGApplication:
    """FastAPI application container."""

    pass


app = FastAPI(title="RAG OpenRouter Project")

app.include_router(router)
