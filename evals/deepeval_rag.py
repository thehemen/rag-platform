from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
import json

from deepeval.test_case import LLMTestCase

from evals.datasets import EvaluationDataset, EvaluationExample
from evals.rag_api_client import RAGAPIClient, RAGAPIClientError, RAGAPIResponse
from evals.safe_deepeval_metrics import (
    SafeAnswerRelevancyMetric,
    SafeContextualRelevancyMetric,
    SafeCorrectnessMetric,
    SafeFaithfulnessMetric,
)
from src.llm import GroqClientError, GroqLLMClient
from src.utils import LLMTextUtils


@dataclass(frozen=True)
class MetricResult:
    """Result for a single evaluation metric."""

    name: str
    score: float | None
    success: bool | None
    reason: str = ""
    error: str = ""


@dataclass(frozen=True)
class RAGEvaluationResult:
    """Evaluation result for one RAG question."""

    id: str
    question: str
    expected_answer: str
    expected_sources: list[str]
    actual_answer: str
    retrieved_sources: list[str]
    retrieved_context: list[str]
    metrics: list[MetricResult] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class DeepEvalRAGRunner:
    """Runs stable RAG evaluation with DeepEval test cases and safe metrics."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        api_client: RAGAPIClient,
        generator_client: GroqLLMClient,
        generator_model_name: str = "",
        report_path: str | Path = "data/eval/reports/deepeval_results.jsonl",
        top_k: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 512,
        threshold: float = 0.7,
    ) -> None:
        self.dataset = dataset
        self.api_client = api_client
        self.generator_client = generator_client
        self.generator_model_name = generator_model_name
        self.report_path = Path(report_path)
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.threshold = threshold

    def run(self) -> list[RAGEvaluationResult]:
        print("[deepeval] Loading evaluation dataset...", flush=True)

        examples = self.dataset.load()
        results: list[RAGEvaluationResult] = []

        print(f"[deepeval] Loaded examples: {len(examples)}", flush=True)
        print(f"[deepeval] top_k={self.top_k}", flush=True)
        print(f"[deepeval] generator={self.generator_model_name}", flush=True)
        print("[deepeval] metrics=safe deterministic metrics", flush=True)

        for index, example in enumerate(examples, start=1):
            print(
                f"[deepeval] {index}/{len(examples)} {example.id}: {example.question}",
                flush=True,
            )

            result = self._run_single_example(example)
            results.append(result)
            self._append_report(result)

        self._print_summary(results)

        return results

    def _run_single_example(
        self,
        example: EvaluationExample,
    ) -> RAGEvaluationResult:
        try:
            print(f"[deepeval] {example.id} retrieving context...", flush=True)

            retrieve_response = self.api_client.retrieve(
                question=example.question,
                top_k=self.top_k,
            )

            retrieved_context = self._extract_retrieved_context(retrieve_response)
            retrieved_sources = self._extract_retrieved_sources(retrieve_response)

            print(
                f"[deepeval] {example.id} retrieved_sources={retrieved_sources}",
                flush=True,
            )

            if not retrieved_context:
                return self._error_result(
                    example=example,
                    error="No retrieved context.",
                    retrieved_sources=retrieved_sources,
                    retrieved_context=[],
                )

            print(f"[deepeval] {example.id} generating answer...", flush=True)

            generated = self._generate_answer(
                question=example.question,
                retrieved_context=retrieved_context,
            )

            print(f"[deepeval] {example.id} answer generated.", flush=True)
            print(f"[deepeval] {example.id} running safe metrics...", flush=True)

            metric_results = self._evaluate_metrics(
                example=example,
                actual_answer=generated.text,
                retrieved_context=retrieved_context,
            )

            print(f"[deepeval] {example.id} metrics finished.", flush=True)

            return RAGEvaluationResult(
                id=example.id,
                question=example.question,
                expected_answer=example.expected_answer,
                expected_sources=example.expected_sources,
                actual_answer=generated.text,
                retrieved_sources=retrieved_sources,
                retrieved_context=retrieved_context,
                metrics=metric_results,
                model=generated.model,
                usage=generated.usage,
            )

        except RAGAPIClientError as error:
            return self._error_result(
                example=example,
                error=f"RAG API error: {error}",
            )

        except GroqClientError as error:
            return self._error_result(
                example=example,
                error=f"Groq error: {error}",
            )

        except Exception as error:
            return self._error_result(
                example=example,
                error=f"Unexpected evaluation error: {type(error).__name__}: {error}",
            )

    def _generate_answer(
        self,
        question: str,
        retrieved_context: list[str],
    ):
        context = self._format_context(retrieved_context)

        prompt = (
            "Answer the question using only the provided context.\n"
            "If the answer is not in the context, say that the information was not found.\n"
            "Keep the answer short and factual.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        response = self.generator_client.generate_text(
            prompt=prompt,
            system_prompt=(
                "You are a concise RAG assistant. "
                "Do not include thinking, reasoning traces, <think> blocks, or hidden analysis. "
                "Return only the final answer."
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return type(response)(
            text=LLMTextUtils.clean_text(response.text),
            model=response.model,
            usage=response.usage,
        )

    def _evaluate_metrics(
        self,
        example: EvaluationExample,
        actual_answer: str,
        retrieved_context: list[str],
    ) -> list[MetricResult]:
        test_case = LLMTestCase(
            input=example.question,
            actual_output=actual_answer,
            expected_output=example.expected_answer,
            retrieval_context=retrieved_context,
        )

        metrics = self._build_metrics()
        results: list[MetricResult] = []

        for metric in metrics:
            metric_name = metric.__class__.__name__

            try:
                metric.measure(test_case)

                results.append(
                    MetricResult(
                        name=metric_name,
                        score=getattr(metric, "score", None),
                        success=getattr(metric, "success", None),
                        reason=str(getattr(metric, "reason", "")),
                    )
                )

                print(
                    f"[deepeval] metric={metric_name} "
                    f"score={getattr(metric, 'score', None)} "
                    f"success={getattr(metric, 'success', None)}",
                    flush=True,
                )

            except Exception as error:
                results.append(
                    MetricResult(
                        name=metric_name,
                        score=None,
                        success=False,
                        reason="",
                        error=f"{type(error).__name__}: {error}",
                    )
                )

                print(
                    f"[deepeval] metric={metric_name} failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

        return results

    def _build_metrics(self):
        return [
            SafeAnswerRelevancyMetric(
                threshold=self.threshold,
                include_reason=True,
            ),
            SafeFaithfulnessMetric(
                threshold=self.threshold,
                include_reason=True,
            ),
            SafeContextualRelevancyMetric(
                threshold=self.threshold,
                include_reason=True,
            ),
            SafeCorrectnessMetric(
                threshold=self.threshold,
                include_reason=True,
            ),
        ]

    def _extract_retrieved_context(
        self,
        api_response: RAGAPIResponse,
    ) -> list[str]:
        contexts: list[str] = []

        for source in api_response.sources:
            text = str(source.get("text", "")).strip()

            if text:
                contexts.append(text)

        return contexts

    def _extract_retrieved_sources(
        self,
        api_response: RAGAPIResponse,
    ) -> list[str]:
        source_names: list[str] = []

        for source in api_response.sources:
            metadata = source.get("metadata", {})
            file_name = str(metadata.get("file_name", "")).strip()

            if file_name:
                source_names.append(file_name)

        return source_names

    def _format_context(
        self,
        retrieved_context: list[str],
    ) -> str:
        chunks: list[str] = []

        for index, context in enumerate(retrieved_context, start=1):
            chunks.append(f"[Context {index}]\n{context}")

        return "\n\n".join(chunks)

    def _error_result(
        self,
        example: EvaluationExample,
        error: str,
        retrieved_sources: list[str] | None = None,
        retrieved_context: list[str] | None = None,
    ) -> RAGEvaluationResult:
        print(f"[deepeval] {example.id} failed: {error}", flush=True)

        return RAGEvaluationResult(
            id=example.id,
            question=example.question,
            expected_answer=example.expected_answer,
            expected_sources=example.expected_sources,
            actual_answer="",
            retrieved_sources=retrieved_sources or [],
            retrieved_context=retrieved_context or [],
            metrics=[],
            error=error,
        )

    def _append_report(
        self,
        result: RAGEvaluationResult,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        with self.report_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    asdict(result),
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(f"[deepeval] saved result for {result.id}", flush=True)

    def _print_summary(
        self,
        results: list[RAGEvaluationResult],
    ) -> None:
        if not results:
            print("[deepeval] No evaluation results.", flush=True)
            return

        metric_scores: dict[str, list[float]] = {}
        error_count = sum(1 for result in results if result.error)

        for result in results:
            for metric in result.metrics:
                if metric.score is None:
                    continue

                metric_scores.setdefault(metric.name, []).append(metric.score)

        print("", flush=True)
        print("DeepEval summary", flush=True)
        print("-" * 80, flush=True)
        print(f"Examples: {len(results)}", flush=True)
        print(f"Errors: {error_count}", flush=True)

        for metric_name, scores in metric_scores.items():
            average_score = sum(scores) / len(scores)
            print(f"{metric_name}: {average_score:.4f}", flush=True)
