from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from deepeval.test_case import LLMTestCase


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "project",
    "goal",
    "purpose",
    "main",
    "feature",
    "features",
}


@dataclass
class SafeMetricResult:
    """Internal result for a safe metric."""

    score: float
    reason: str


class SafeBaseMetric:
    """Small DeepEval-compatible metric interface."""

    def __init__(
        self,
        threshold: float = 0.7,
        include_reason: bool = True,
    ) -> None:
        self.threshold = threshold
        self.include_reason = include_reason
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str = ""

    def measure(self, test_case: LLMTestCase) -> float:
        result = self._measure(test_case)

        self.score = round(float(result.score), 4)
        self.success = self.score >= self.threshold
        self.reason = result.reason if self.include_reason else ""

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    def _measure(self, test_case: LLMTestCase) -> SafeMetricResult:
        raise NotImplementedError


class SafeAnswerRelevancyMetric(SafeBaseMetric):
    """Checks whether the answer addresses the question."""

    def _measure(self, test_case: LLMTestCase) -> SafeMetricResult:
        question_keywords = TextScorer.keywords(test_case.input)
        answer_keywords = TextScorer.keywords(test_case.actual_output)

        score = TextScorer.coverage(
            required_keywords=question_keywords,
            available_keywords=answer_keywords,
        )

        if test_case.expected_output:
            expected_keywords = TextScorer.keywords(test_case.expected_output)
            expected_score = TextScorer.coverage(
                required_keywords=expected_keywords,
                available_keywords=answer_keywords,
            )
            score = max(score, expected_score)

        return SafeMetricResult(
            score=score,
            reason=(
                "The score measures how many important question or expected-answer "
                "keywords are present in the generated answer."
            ),
        )


class SafeFaithfulnessMetric(SafeBaseMetric):
    """Checks whether the answer is supported by retrieved context."""

    def _measure(self, test_case: LLMTestCase) -> SafeMetricResult:
        answer_keywords = TextScorer.keywords(test_case.actual_output)
        context_keywords = TextScorer.keywords(
            " ".join(test_case.retrieval_context or [])
        )

        score = TextScorer.coverage(
            required_keywords=answer_keywords,
            available_keywords=context_keywords,
        )

        return SafeMetricResult(
            score=score,
            reason=(
                "The score measures how many important answer keywords are present "
                "in the retrieved context."
            ),
        )


class SafeContextualRelevancyMetric(SafeBaseMetric):
    """Checks whether retrieved context is relevant to the question."""

    def _measure(self, test_case: LLMTestCase) -> SafeMetricResult:
        question_keywords = TextScorer.keywords(test_case.input)
        context_keywords = TextScorer.keywords(
            " ".join(test_case.retrieval_context or [])
        )

        score = TextScorer.coverage(
            required_keywords=question_keywords,
            available_keywords=context_keywords,
        )

        return SafeMetricResult(
            score=score,
            reason=(
                "The score measures how many important question keywords are present "
                "in the retrieved context."
            ),
        )


class SafeCorrectnessMetric(SafeBaseMetric):
    """Checks whether the actual answer matches the expected answer."""

    def _measure(self, test_case: LLMTestCase) -> SafeMetricResult:
        expected_keywords = TextScorer.keywords(test_case.expected_output)
        answer_keywords = TextScorer.keywords(test_case.actual_output)

        score = TextScorer.coverage(
            required_keywords=expected_keywords,
            available_keywords=answer_keywords,
        )

        return SafeMetricResult(
            score=score,
            reason=(
                "The score measures how many important expected-answer keywords "
                "are present in the generated answer."
            ),
        )


class TextScorer:
    """Utility scoring methods for deterministic RAG evaluation."""

    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

    SYNONYMS = {
        "origin": {"origin", "laborigin", "laborigin"},
        "lab": {"lab", "laboratory"},
        "genetically": {"genetically", "genetic", "engineered"},
        "engineered": {"engineered", "genetically", "genetic"},
        "highest": {"highest", "best", "maximum"},
        "accuracy": {"accuracy", "accurate"},
        "identify": {"identify", "identifies", "detect", "classify"},
        "dataset": {"dataset", "data"},
        "model": {"model", "network", "algorithm"},
    }

    @classmethod
    def keywords(cls, text: str) -> set[str]:
        tokens = cls.tokens(text)

        keywords = {
            token
            for token in tokens
            if len(token) >= 3 and token not in STOPWORDS
        }

        return cls._normalize_keywords(keywords)

    @classmethod
    def tokens(cls, text: str) -> list[str]:
        text = cls._normalize_text(text)
        return cls.TOKEN_PATTERN.findall(text)

    @classmethod
    def coverage(
        cls,
        required_keywords: Iterable[str],
        available_keywords: Iterable[str],
    ) -> float:
        required = set(required_keywords)
        available = set(available_keywords)

        if not required:
            return 1.0

        hits = 0

        for keyword in required:
            if cls._keyword_matches(keyword, available):
                hits += 1

        return hits / len(required)

    @classmethod
    def _keyword_matches(
        cls,
        keyword: str,
        available_keywords: set[str],
    ) -> bool:
        if keyword in available_keywords:
            return True

        synonyms = cls.SYNONYMS.get(keyword, set())

        if synonyms and synonyms.intersection(available_keywords):
            return True

        for available in available_keywords:
            if keyword in available or available in keyword:
                return True

        return False

    @classmethod
    def _normalize_keywords(
        cls,
        keywords: set[str],
    ) -> set[str]:
        normalized: set[str] = set()

        for keyword in keywords:
            normalized.add(keyword)

            if keyword == "laboforigin":
                normalized.add("lab")
                normalized.add("origin")

        return normalized

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        text = text.lower()
        text = text.replace("lab-of-origin", "lab origin")
        text = text.replace("n-grams", "ngrams")
        text = text.replace("six-grams", "sixgrams")
        return text
