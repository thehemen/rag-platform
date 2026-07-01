from __future__ import annotations

from dataclasses import dataclass

from src.embeddings import SentenceTransformerEmbedder
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.document_chunker import DocumentChunker
from src.vector_store import QdrantClientFactory, QdrantIndex


@dataclass(frozen=True)
class IndexBuildResult:
    """Result of the index rebuilding process."""

    document_count: int
    chunk_count: int
    inserted_count: int
    stored_count: int
    collection_name: str
    embedding_model: str
    embedding_dimension: int


class IndexBuilder:
    """Rebuilds the Qdrant index from local source documents."""

    def __init__(
        self,
        markdown_dir: str = "data/raw/markdown",
        plaintext_dir: str = "data/raw/plaintext",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str | None = None,
        collection_name: str = "rag_markdown_docs",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_device: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 80,
        batch_size: int = 32,
        recreate_collection: bool = True,
    ) -> None:
        self.markdown_dir = markdown_dir
        self.plaintext_dir = plaintext_dir
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.recreate_collection = recreate_collection

    def rebuild(self) -> IndexBuildResult:
        documents = self._load_documents()
        nodes = self._chunk_documents(documents)
        embedder = self._create_embedder()
        embedded_nodes = embedder.embed_nodes(nodes)
        qdrant_index = self._create_qdrant_index(embedder)

        if self.recreate_collection:
            qdrant_index.recreate_collection()
        else:
            qdrant_index.create_collection()

        inserted_count = qdrant_index.upsert_nodes(
            nodes=embedded_nodes,
            batch_size=self.batch_size,
        )

        stored_count = qdrant_index.count_points()

        return IndexBuildResult(
            document_count=len(documents),
            chunk_count=len(nodes),
            inserted_count=inserted_count,
            stored_count=stored_count,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            embedding_dimension=embedder.get_embedding_dimension(),
        )

    def _load_documents(self):
        loader = DatasetLoader(
            markdown_dir=self.markdown_dir,
            plaintext_dir=self.plaintext_dir,
        )

        return loader.load()

    def _chunk_documents(self, documents):
        chunker = DocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return chunker.chunk(documents)

    def _create_embedder(self) -> SentenceTransformerEmbedder:
        return SentenceTransformerEmbedder(
            model_name=self.embedding_model,
            device=self.embedding_device,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )

    def _create_qdrant_index(
        self,
        embedder: SentenceTransformerEmbedder,
    ) -> QdrantIndex:
        client = QdrantClientFactory(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        ).create()

        return QdrantIndex(
            client=client,
            collection_name=self.collection_name,
            vector_size=embedder.get_embedding_dimension(),
        )
