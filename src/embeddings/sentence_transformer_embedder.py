from __future__ import annotations

from typing import Sequence

from llama_index.core.schema import BaseNode
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Creates embeddings with a local SentenceTransformers model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(
            model_name_or_path=self.model_name,
            device=self.device,
        )

    def embed_text(self, text: str) -> list[float]:
        """Embeds a single text value."""

        if not text.strip():
            return []

        embedding = self.model.encode(
            text,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeds a list of text values."""

        valid_texts = [text if text.strip() else " " for text in texts]

        if not valid_texts:
            return []

        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embeds a user query."""

        return self.embed_text(query)

    def embed_nodes(self, nodes: Sequence[BaseNode]) -> list[BaseNode]:
        """Embeds nodes and stores vectors inside each node."""

        if not nodes:
            return []

        texts = [node.get_content() for node in nodes]
        embeddings = self.embed_texts(texts)

        embedded_nodes: list[BaseNode] = []

        for node, embedding in zip(nodes, embeddings, strict=True):
            node.embedding = embedding
            node.metadata["embedding_model"] = self.model_name
            node.metadata["embedding_dimension"] = len(embedding)
            embedded_nodes.append(node)

        return embedded_nodes

    def get_embedding_dimension(self) -> int:
        """Returns the output vector size of the embedding model."""

        return int(self.model.get_embedding_dimension())
