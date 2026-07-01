from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from groq import Groq
from groq import APIConnectionError, APIError, RateLimitError

from src.utils import LLMTextUtils


@dataclass(frozen=True)
class ChatMessage:
    """Single chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class GroqResponse:
    """Generated Groq response."""

    text: str
    model: str
    usage: dict[str, Any]


class GroqClientError(Exception):
    """Base Groq client error."""


class GroqRateLimitError(GroqClientError):
    """Raised when all Groq models are rate-limited."""


class GroqGenerationError(GroqClientError):
    """Raised when Groq answer generation fails."""


class GroqLLMClient:
    """Groq client wrapper for chat completion generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen/qwen3.6-27b",
        fallback_models: list[str] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        reasoning_effort: str = "default",
        stream: bool = False,
        strip_reasoning: bool = True,
        max_retries: int = 2,
        retry_sleep_seconds: float = 2.0,
        timeout: float = 180.0,
    ) -> None:
        if not model:
            raise ValueError("Groq model name is required.")

        self.api_key = api_key
        self.model = model
        self.fallback_models = fallback_models or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.strip_reasoning = strip_reasoning
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.timeout = timeout

        client_kwargs: dict[str, Any] = {
            "timeout": self.timeout,
        }

        if self.api_key:
            client_kwargs["api_key"] = self.api_key

        self.client = Groq(**client_kwargs)

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GroqResponse:
        payload_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        models_to_try = self._get_models_to_try()
        rate_limit_errors: list[str] = []
        generation_errors: list[str] = []

        for model_name in models_to_try:
            try:
                return self._generate_with_model(
                    model_name=model_name,
                    messages=payload_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            except RateLimitError as error:
                rate_limit_errors.append(
                    f"{model_name}: {self._format_error(error)}"
                )

            except (APIConnectionError, APIError) as error:
                generation_errors.append(
                    f"{model_name}: {self._format_error(error)}"
                )

        if rate_limit_errors and not generation_errors:
            raise GroqRateLimitError(
                "All Groq models are currently rate-limited. "
                + " | ".join(rate_limit_errors)
            )

        raise GroqGenerationError(
            "Groq generation failed. "
            + " | ".join(rate_limit_errors + generation_errors)
        )

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GroqResponse:
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

    def _generate_with_model(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> GroqResponse:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_completion_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    top_p=self.top_p,
                    reasoning_effort=self.reasoning_effort,
                    stream=self.stream,
                    stop=None,
                )

                if self.stream:
                    text = self._collect_stream_text(completion)

                    if self.strip_reasoning:
                        text = LLMTextUtils.strip_reasoning(text)

                    return GroqResponse(
                        text=text,
                        model=model_name,
                        usage={},
                    )

                text = completion.choices[0].message.content or ""

                if self.strip_reasoning:
                    text = LLMTextUtils.strip_reasoning(text)

                return GroqResponse(
                    text=text,
                    model=getattr(completion, "model", model_name) or model_name,
                    usage=self._extract_usage(completion),
                )

            except RateLimitError as error:
                last_error = error

                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds)
                    continue

                raise

            except (APIConnectionError, APIError) as error:
                last_error = error

                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds)
                    continue

                raise

        raise GroqGenerationError(
            f"Groq generation failed for model {model_name}: {last_error}"
        )

    def _collect_stream_text(self, completion) -> str:
        parts: list[str] = []

        for chunk in completion:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if content:
                parts.append(content)

        return "".join(parts)

    def _get_models_to_try(self) -> list[str]:
        models = [self.model]

        for fallback_model in self.fallback_models:
            fallback_model = fallback_model.strip()

            if fallback_model and fallback_model not in models:
                models.append(fallback_model)

        return models

    def _extract_usage(self, completion: Any) -> dict[str, Any]:
        usage = getattr(completion, "usage", None)

        if usage is None:
            return {}

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def _format_error(self, error: Exception) -> str:
        response = getattr(error, "response", None)

        if response is not None:
            try:
                return str(response.json())
            except Exception:
                return str(error)

        return str(error)
