from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.llm import OpenRouterClient
from src.rag import RAGQueryEngine, RAGRetriever


load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_rag_query_engine() -> RAGQueryEngine:
    """Creates and caches the RAG query engine."""

    retriever = RAGRetriever(
        qdrant_url=os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        ),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        collection_name=os.getenv(
            "QDRANT_COLLECTION",
            "rag_markdown_docs",
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5",
        ),
        embedding_device=os.getenv("EMBEDDING_DEVICE") or None,
        embedding_batch_size=int(
            os.getenv("EMBEDDING_BATCH_SIZE", "32"),
        ),
    )

    llm_client = OpenRouterClient(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        model=os.getenv("OPENROUTER_MODEL", ""),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
    )

    return RAGQueryEngine(
        retriever=retriever,
        llm_client=llm_client,
    )
