from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion import DatasetLoader


def main() -> None:
    loader = DatasetLoader()
    documents = loader.load()

    print(f"Loaded documents: {len(documents)}")

    for document in documents:
        metadata = document.metadata

        print("-" * 80)
        print(f"ID: {metadata.get('document_id')}")
        print(f"Type: {metadata.get('source_type')}")
        print(f"File: {metadata.get('file_path')}")
        print(f"Characters: {len(document.text)}")


if __name__ == "__main__":
    main()
