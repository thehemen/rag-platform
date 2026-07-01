from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr
import requests


@dataclass(frozen=True)
class GradioSettings:
    """Settings for the Gradio demo."""

    api_url: str = "http://localhost:8000"
    request_timeout: float = 120.0


class GradioRAGApp:
    """Gradio interface for testing the RAG API."""

    def __init__(self, settings: GradioSettings | None = None) -> None:
        self.settings = settings or GradioSettings()

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="RAG OpenRouter Demo") as demo:
            gr.Markdown("# RAG OpenRouter Demo")
            gr.Markdown(
                "Ask a question about the indexed Markdown and text documents."
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is this project?",
                    lines=3,
                )

            with gr.Row():
                top_k_input = gr.Slider(
                    label="Top K",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                )
                temperature_input = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.2,
                    step=0.1,
                )
                max_tokens_input = gr.Slider(
                    label="Max tokens",
                    minimum=128,
                    maximum=4096,
                    value=800,
                    step=128,
                )

            ask_button = gr.Button("Ask", variant="primary")

            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Textbox(
                label="Sources",
                lines=14,
                interactive=False,
            )
            raw_output = gr.JSON(label="Raw response")

            ask_button.click(
                fn=self.ask,
                inputs=[
                    question_input,
                    top_k_input,
                    temperature_input,
                    max_tokens_input,
                ],
                outputs=[
                    answer_output,
                    sources_output,
                    raw_output,
                ],
            )

        return demo

    def ask(
        self,
        question: str,
        top_k: int,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str, dict[str, Any]]:
        question = question.strip()

        if not question:
            return (
                "Please enter a question.",
                "No sources.",
                {},
            )

        payload = {
            "question": question,
            "top_k": int(top_k),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        try:
            response = requests.post(
                f"{self.settings.api_url.rstrip('/')}/ask",
                json=payload,
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            return (
                f"API request failed: `{error}`",
                "No sources.",
                {},
            )

        data = response.json()

        answer = data.get("answer", "")
        sources = self._format_sources(data.get("sources", []))

        return answer, sources, data

    def _format_sources(
        self,
        sources: list[dict[str, Any]],
    ) -> str:
        if not sources:
            return "No sources."

        lines: list[str] = []

        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})

            lines.extend(
                [
                    f"Source #{index}",
                    f"Score: {source.get('score', 0.0):.4f}",
                    f"Point ID: {source.get('point_id', '')}",
                    f"Chunk ID: {metadata.get('chunk_id', '')}",
                    f"Document ID: {metadata.get('document_id', '')}",
                    f"Source type: {metadata.get('source_type', '')}",
                    f"File: {metadata.get('file_path', '')}",
                    "",
                    str(source.get("text", ""))[:1000],
                    "",
                    "-" * 80,
                    "",
                ]
            )

        return "\n".join(lines)
