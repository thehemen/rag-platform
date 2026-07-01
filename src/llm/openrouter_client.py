from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class ChatMessage:
    """Single chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class OpenRouterResponse:
    """Generated LLM response."""

    text: str
    model: str
    usage: dict[str, Any]


class OpenRouterClient:
    """OpenAI-compatible client wrapper for OpenRouter."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required.")

        if not model:
            raise ValueError("OpenRouter model name is required.")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> OpenRouterResponse:
        payload_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload_messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

        text = response.choices[0].message.content or ""

        return OpenRouterResponse(
            text=text,
            model=response.model or self.model,
            usage=self._extract_usage(response),
        )

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> OpenRouterResponse:
        messages = [
            ChatMessage(
                role="system",
                content=system_prompt,
            ),
            ChatMessage(
                role="user",
                content=prompt,
            ),
        ]

        return self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)

        if usage is None:
            return {}

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
