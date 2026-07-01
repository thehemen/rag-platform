from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag import RAGRetriever


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    query = _read_query()

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

    top_k = int(os.getenv("TOP_K", "3"))

    result = retriever.retrieve(
        query=query,
        top_k=top_k,
    )

    print()
    print("Retrieval result")
    print("-" * 80)
    print(f"Query: {result.query}")
    print(f"Collection: {result.collection_name}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Chunks: {len(result.chunks)}")

    for index, chunk in enumerate(result.chunks, start=1):
        metadata = chunk.metadata

        print()
        print("=" * 80)
        print(f"Result #{index}")
        print(f"Score: {chunk.score:.4f}")
        print(f"Point ID: {chunk.point_id}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(f"Document ID: {metadata.get('document_id')}")
        print(f"Source type: {metadata.get('source_type')}")
        print(f"File: {metadata.get('file_path')}")
        print("-" * 80)
        print(chunk.text[:1200])


def _read_query() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    return input("Question: ").strip()


if __name__ == "__main__":
    main()
