from pydantic import BaseModel


class AppSettings(BaseModel):
    """Application settings."""

    app_env: str = "local"


class ModelSettings(BaseModel):
    """LLM and embedding model settings."""

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"


class QdrantSettings(BaseModel):
    """Qdrant connection settings."""

    url: str = "http://localhost:6333"
    collection_name: str = "rag_markdown_docs"


class RAGSettings(BaseModel):
    """RAG pipeline settings."""

    chunk_size: int = 512
    chunk_overlap: int = 80
    top_k: int = 3


class Settings(BaseModel):
    """Root project settings."""

    app: AppSettings = AppSettings()
    models: ModelSettings = ModelSettings()
    qdrant: QdrantSettings = QdrantSettings()
    rag: RAGSettings = RAGSettings()
