from __future__ import annotations

from src.rag.retriever import RetrievedChunk


class SourceFormatter:
    """Formats retrieved chunks for prompts and output."""

    def format_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:
        if not chunks:
            return "No context was retrieved."

        formatted_chunks: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            formatted_chunks.append(
                self._format_chunk(
                    index=index,
                    chunk=chunk,
                )
            )

        return "\n\n".join(formatted_chunks)

    def _format_chunk(
        self,
        index: int,
        chunk: RetrievedChunk,
    ) -> str:
        file_name = chunk.metadata.get("file_name", "")
        file_path = chunk.metadata.get("file_path", "")
        chunk_id = chunk.metadata.get("chunk_id", "")

        return (
            f"[Source {index}]\n"
            f"File name: {file_name}\n"
            f"File path: {file_path}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Score: {chunk.score:.4f}\n"
            f"Text:\n{chunk.text}"
        )

    def format_sources_for_display(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:
        if not chunks:
            return "No sources."

        lines: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata

            lines.append(
                "\n".join(
                    [
                        f"{index}. {metadata.get('file_name', '')}",
                        f"   Score: {chunk.score:.4f}",
                        f"   Chunk ID: {metadata.get('chunk_id', '')}",
                        f"   Path: {metadata.get('file_path', '')}",
                    ]
                )
            )

        return "\n\n".join(lines)
