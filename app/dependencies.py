from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.llm import GroqLLMClient
from src.rag import RAGQueryEngine, RAGRetriever


load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_rag_retriever() -> RAGRetriever:
    """Creates and caches the RAG retriever."""

    return RAGRetriever(
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


@lru_cache(maxsize=1)
def get_rag_query_engine() -> RAGQueryEngine:
    """Creates and caches the RAG query engine."""

    retriever = get_rag_retriever()

    llm_client = GroqLLMClient(
        api_key=os.getenv("GROQ_API_KEY") or None,
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
        fallback_models=_parse_models(os.getenv("GROQ_FALLBACK_MODELS", "")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.6")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        top_p=float(os.getenv("LLM_TOP_P", "0.95")),
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "default"),
        stream=os.getenv("LLM_STREAM", "false").lower()
        in {"1", "true", "yes", "y"},
        strip_reasoning=True,
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        retry_sleep_seconds=float(
            os.getenv("LLM_RETRY_SLEEP_SECONDS", "2.0")
        ),
        timeout=float(os.getenv("LLM_TIMEOUT", "180")),
    )

    return RAGQueryEngine(
        retriever=retriever,
        llm_client=llm_client,
    )


def _parse_models(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]
