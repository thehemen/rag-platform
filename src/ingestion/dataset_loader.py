from pathlib import Path
from typing import Iterable

from llama_index.core import Document


class DatasetLoader:
    """Loads source documents from the local dataset folders."""

    MARKDOWN_EXTENSIONS = {".md", ".markdown"}
    PLAINTEXT_EXTENSIONS = {".txt", ".text"}

    def __init__(
        self,
        markdown_dir: str | Path = "data/raw/markdown",
        plaintext_dir: str | Path = "data/raw/plaintext",
        encoding: str = "utf-8",
    ) -> None:
        self.markdown_dir = Path(markdown_dir)
        self.plaintext_dir = Path(plaintext_dir)
        self.encoding = encoding

    def load(self) -> list[Document]:
        documents: list[Document] = []

        documents.extend(self.load_markdown())
        documents.extend(self.load_plaintext())

        return documents

    def load_markdown(self) -> list[Document]:
        return self._load_from_directory(
            directory=self.markdown_dir,
            extensions=self.MARKDOWN_EXTENSIONS,
            source_type="markdown",
        )

    def load_plaintext(self) -> list[Document]:
        return self._load_from_directory(
            directory=self.plaintext_dir,
            extensions=self.PLAINTEXT_EXTENSIONS,
            source_type="plaintext",
        )

    def _load_from_directory(
        self,
        directory: Path,
        extensions: set[str],
        source_type: str,
    ) -> list[Document]:
        if not directory.exists():
            return []

        documents: list[Document] = []

        for file_path in self._iter_files(directory, extensions):
            text = self._read_text(file_path)

            if not text.strip():
                continue

            document = Document(
                text=text,
                metadata=self._build_metadata(
                    file_path=file_path,
                    root_dir=directory,
                    source_type=source_type,
                ),
            )

            documents.append(document)

        return documents

    def _iter_files(self, directory: Path, extensions: set[str]) -> Iterable[Path]:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                yield file_path

    def _read_text(self, file_path: Path) -> str:
        return file_path.read_text(encoding=self.encoding, errors="replace")

    def _build_metadata(
        self,
        file_path: Path,
        root_dir: Path,
        source_type: str,
    ) -> dict:
        relative_path = file_path.relative_to(root_dir)

        return {
            "source_type": source_type,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "relative_path": str(relative_path),
            "extension": file_path.suffix.lower(),
            "document_id": self._build_document_id(source_type, relative_path),
        }

    def _build_document_id(self, source_type: str, relative_path: Path) -> str:
        normalized_path = str(relative_path).replace("\\", "/")
        return f"{source_type}:{normalized_path}"
