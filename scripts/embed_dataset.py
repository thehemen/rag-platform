from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.embeddings import SentenceTransformerEmbedder
from src.ingestion import DatasetLoader, DocumentChunker


def main() -> None:
    loader = DatasetLoader()
    documents = loader.load()

    if not documents:
        print("No documents found.")
        print("Expected folders:")
        print("- data/raw/markdown")
        print("- data/raw/plaintext")
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

    print(f"Loaded documents: {len(documents)}.")
    print(f"Created chunks: {len(nodes)}.")
    print(f"Embedded chunks: {len(embedded_nodes)}.")
    print(f"Embedding model: {embedder.model_name}.")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}.")

    for node in embedded_nodes[:5]:
        metadata = node.metadata
        embedding = node.embedding or []

        print("-" * 80)
        print(f"Chunk ID: {metadata.get('chunk_id')}.")
        print(f"Document ID: {metadata.get('document_id')}.")
        print(f"File: {metadata.get('file_path')}.")
        print(f"Embedding dimension: {len(embedding)}.")
        print(f"Embedding preview: {embedding[:5]}.")


if __name__ == "__main__":
    main()
