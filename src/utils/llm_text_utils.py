from __future__ import annotations

import json
import re


class LLMTextUtils:
    """Helpers for cleaning LLM outputs."""

    THINK_BLOCK_PATTERNS = [
        re.compile(
            r"<think>.*?</think>",
            flags=re.DOTALL | re.IGNORECASE,
        ),
        re.compile(
            r"<thinking>.*?</thinking>",
            flags=re.DOTALL | re.IGNORECASE,
        ),
        re.compile(
            r"\[think\].*?\[/think\]",
            flags=re.DOTALL | re.IGNORECASE,
        ),
        re.compile(
            r"\[thinking\].*?\[/thinking\]",
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ]

    CODE_BLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*(.*?)```",
        flags=re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Cleans model output for normal application use."""

        text = cls.strip_think_blocks(text)
        text = cls._normalize_spacing(text)

        return text

    @classmethod
    def strip_think_blocks(cls, text: str) -> str:
        """Removes visible thinking blocks from model output."""

        if not text:
            return ""

        cleaned = text

        for pattern in cls.THINK_BLOCK_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        return cleaned.strip()

    @classmethod
    def extract_json_text(cls, text: str) -> str:
        """Extracts a valid JSON object or array from a model response."""

        cleaned = cls.clean_text(text)
        cleaned = cls._strip_code_block(cleaned)

        if cls._is_valid_json(cleaned):
            return cleaned

        json_candidate = cls._find_json_candidate(cleaned)

        if json_candidate and cls._is_valid_json(json_candidate):
            return json_candidate

        return cleaned

    @classmethod
    def _normalize_spacing(cls, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]

        normalized_lines: list[str] = []
        previous_empty = False

        for line in lines:
            is_empty = not line.strip()

            if is_empty and previous_empty:
                continue

            normalized_lines.append(line)
            previous_empty = is_empty

        return "\n".join(normalized_lines).strip()

    @classmethod
    def _strip_code_block(cls, text: str) -> str:
        match = cls.CODE_BLOCK_PATTERN.search(text)

        if match:
            return match.group(1).strip()

        return text.strip()

    @classmethod
    def _find_json_candidate(cls, text: str) -> str:
        object_candidate = cls._extract_between(text, "{", "}")

        if object_candidate:
            return object_candidate

        array_candidate = cls._extract_between(text, "[", "]")

        if array_candidate:
            return array_candidate

        return ""

    @classmethod
    def _extract_between(
        cls,
        text: str,
        open_char: str,
        close_char: str,
    ) -> str:
        start = text.find(open_char)
        end = text.rfind(close_char)

        if start == -1 or end == -1 or end <= start:
            return ""

        return text[start : end + 1].strip()

    @classmethod
    def _is_valid_json(cls, text: str) -> bool:
        try:
            json.loads(text)
        except Exception:
            return False

        return True
