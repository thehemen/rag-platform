from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion import DatasetLoader, DocumentChunker


def main() -> None:
    loader = DatasetLoader()
    documents = loader.load()

    chunker = DocumentChunker(
        chunk_size=512,
        chunk_overlap=80,
    )

    nodes = chunker.chunk(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(nodes)}")

    for node in nodes[:10]:
        metadata = node.metadata
        text = node.get_content()

        print("-" * 80)
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(f"Document ID: {metadata.get('document_id')}")
        print(f"Source type: {metadata.get('source_type')}")
        print(f"File: {metadata.get('file_path')}")
        print(f"Chunk index: {metadata.get('chunk_index')}")
        print(f"Characters: {len(text)}")
        print()
        print(text[:500])


if __name__ == "__main__":
    main()
