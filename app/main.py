from __future__ import annotations

from fastapi import FastAPI

from app.api_routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Grok Project",
        description="Small RAG API using Qdrant, SentenceTransformers, and Grok.",
        version="0.1.0",
    )

    app.include_router(router)

    return app


app = create_app()
