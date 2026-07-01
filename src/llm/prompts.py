from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGPrompt:
    """Final prompt data for answer generation."""

    system_prompt: str
    user_prompt: str


class PromptTemplates:
    """Prompt templates for RAG answer generation."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful RAG assistant. "
        "Answer using only the provided context. "
        "If the context does not contain the answer, say that the information was not found. "
        "Keep the answer concise and cite source file names when possible."
    )

    @classmethod
    def build_rag_prompt(
        cls,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> RAGPrompt:
        return RAGPrompt(
            system_prompt=system_prompt or cls.DEFAULT_SYSTEM_PROMPT,
            user_prompt=(
                "Context:\n"
                f"{context}\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Answer:"
            ),
        )

    @classmethod
    def build_context_from_chunks(
        cls,
        chunks: list[str],
    ) -> str:
        if not chunks:
            return "No context was retrieved."

        formatted_chunks = []

        for index, chunk in enumerate(chunks, start=1):
            formatted_chunks.append(
                f"[Chunk {index}]\n{chunk}"
            )

        return "\n\n".join(formatted_chunks)
