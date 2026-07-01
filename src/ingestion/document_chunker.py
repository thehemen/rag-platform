from collections import defaultdict
from typing import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


class DocumentChunker:
    """Splits loaded documents into smaller LlamaIndex nodes."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 80,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, documents: Sequence[Document]) -> list[BaseNode]:
        if not documents:
            return []

        nodes = self.node_parser.get_nodes_from_documents(
            documents=list(documents),
            show_progress=False,
        )

        return self._add_chunk_metadata(nodes)

    def _add_chunk_metadata(self, nodes: list[BaseNode]) -> list[BaseNode]:
        counters: dict[str, int] = defaultdict(int)

        for node in nodes:
            document_id = node.metadata.get("document_id", "unknown_document")
            chunk_index = counters[document_id]

            node.metadata["chunk_index"] = chunk_index
            node.metadata["chunk_id"] = self._build_chunk_id(
                document_id=document_id,
                chunk_index=chunk_index,
            )

            counters[document_id] += 1

        return nodes

    def _build_chunk_id(self, document_id: str, chunk_index: int) -> str:
        return f"{document_id}:chunk:{chunk_index}"
