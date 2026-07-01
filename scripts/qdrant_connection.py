from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.embeddings import SentenceTransformerEmbedder
from src.ingestion import DatasetLoader, DocumentChunker
from src.vector_store import QdrantClientFactory, QdrantIndex


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    collection_name = os.getenv("QDRANT_COLLECTION", "rag_markdown_docs")

    loader = DatasetLoader()
    documents = loader.load()

    if not documents:
        print("No documents found.")
        return

    chunker = DocumentChunker(
        chunk_size=512,
        chunk_overlap=80,
    )
    nodes = chunker.chunk(documents)

    if not nodes:
        print("No chunks created.")
        return

    embedder = SentenceTransformerEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        device=None,
        batch_size=32,
        normalize_embeddings=True,
    )

    embedded_nodes = embedder.embed_nodes(nodes)

    client = QdrantClientFactory(
        url=qdrant_url,
        api_key=qdrant_api_key,
    ).create()

    index = QdrantIndex(
        client=client,
        collection_name=collection_name,
        vector_size=embedder.get_embedding_dimension(),
    )

    index.recreate_collection()

    inserted_count = index.upsert_nodes(
        nodes=embedded_nodes,
        batch_size=64,
    )

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection: {collection_name}")
    print(f"Inserted chunks: {inserted_count}")
    print(f"Stored points: {index.count_points()}")

    query = "What is this project?"
    query_vector = embedder.embed_query(query)

    results = index.search(
        query_vector=query_vector,
        top_k=3,
    )

    print()
    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    for result in results:
        print("-" * 80)
        print(f"Point ID: {result.point_id}")
        print(f"Score: {result.score:.4f}")
        print(f"File: {result.metadata.get('file_path')}")
        print(f"Chunk ID: {result.metadata.get('chunk_id')}")
        print()
        print(result.text[:500])


if __name__ == "__main__":
    main()
