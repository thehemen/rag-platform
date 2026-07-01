from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion import IndexBuilder


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    builder = IndexBuilder(
        markdown_dir=os.getenv(
            "MARKDOWN_DIR",
            "data/raw/markdown",
        ),
        plaintext_dir=os.getenv(
            "PLAINTEXT_DIR",
            "data/raw/plaintext",
        ),
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
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        recreate_collection=os.getenv(
            "RECREATE_COLLECTION",
            "true",
        ).lower()
        in {"1", "true", "yes", "y"},
    )

    result = builder.rebuild()

    print("Index rebuilt successfully.")
    print("-" * 80)
    print(f"Documents: {result.document_count}.")
    print(f"Chunks: {result.chunk_count}.")
    print(f"Inserted points: {result.inserted_count}.")
    print(f"Stored points: {result.stored_count}.")
    print(f"Collection: {result.collection_name}.")
    print(f"Embedding model: {result.embedding_model}.")
    print(f"Embedding dimension: {result.embedding_dimension}.")


if __name__ == "__main__":
    main()
