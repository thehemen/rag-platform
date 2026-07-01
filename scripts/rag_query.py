from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.llm import OpenRouterClient
from src.rag import RAGQueryConfig, RAGQueryEngine, RAGQueryInput, RAGRetriever


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    question = _read_question()

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

    query_engine = RAGQueryEngine(
        retriever=retriever,
        llm_client=llm_client,
    )

    result = query_engine.query(
        RAGQueryInput(
            question=question,
            config=RAGQueryConfig(
                top_k=int(os.getenv("TOP_K", "3")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
            ),
        )
    )

    _print_result(result)


def _read_question() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    return input("Question: ").strip()


def _print_result(result) -> None:
    print()
    print("RAG answer")
    print("-" * 80)
    print(f"Question: {result.question}")
    print(f"Model: {result.model}")
    print(f"Collection: {result.collection_name}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Usage: {result.usage}")
    print("-" * 80)
    print(result.answer)

    print()
    print("Sources")
    print("-" * 80)

    if not result.sources:
        print("No sources.")
        return

    for index, source in enumerate(result.sources, start=1):
        metadata = source.metadata

        print()
        print(f"Source #{index}")
        print(f"Score: {source.score:.4f}")
        print(f"Point ID: {source.point_id}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(f"Document ID: {metadata.get('document_id')}")
        print(f"Source type: {metadata.get('source_type')}")
        print(f"File: {metadata.get('file_path')}")
        print()
        print(source.text[:700])


if __name__ == "__main__":
    main()
