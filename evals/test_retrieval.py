from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from evals.datasets import EvaluationDataset
from evals.rag_api_client import RAGAPIClient, RAGAPIClientError


@dataclass(frozen=True)
class RetrievalCheckResult:
    """Deterministic retrieval check result."""

    id: str
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    hit: bool
    error: str = ""


class RetrievalEvaluation:
    """Checks whether expected files appear in retrieved sources."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        api_client: RAGAPIClient,
        report_path: str | Path = "data/eval/reports/retrieval_results.jsonl",
        top_k: int = 3,
    ) -> None:
        self.dataset = dataset
        self.api_client = api_client
        self.report_path = Path(report_path)
        self.top_k = top_k

    def run(self) -> list[RetrievalCheckResult]:
        print("[retrieval] Loading evaluation dataset...", flush=True)

        examples = self.dataset.load()

        print(f"[retrieval] Loaded examples: {len(examples)}", flush=True)
        print(f"[retrieval] top_k={self.top_k}", flush=True)

        results: list[RetrievalCheckResult] = []

        for index, example in enumerate(examples, start=1):
            print(
                f"[retrieval] {index}/{len(examples)} "
                f"{example.id}: {example.question}",
                flush=True,
            )

            try:
                response = self.api_client.retrieve(
                    question=example.question,
                    top_k=self.top_k,
                )

                retrieved_sources = self._extract_sources(response.sources)

                hit = self._has_hit(
                    expected_sources=example.expected_sources,
                    retrieved_sources=retrieved_sources,
                )

                print(
                    f"[retrieval] {example.id} "
                    f"retrieved={retrieved_sources} "
                    f"expected={example.expected_sources} "
                    f"hit={hit}",
                    flush=True,
                )

                result = RetrievalCheckResult(
                    id=example.id,
                    question=example.question,
                    expected_sources=example.expected_sources,
                    retrieved_sources=retrieved_sources,
                    hit=hit,
                )

            except RAGAPIClientError as error:
                print(
                    f"[retrieval] {example.id} failed: {error}",
                    flush=True,
                )

                result = RetrievalCheckResult(
                    id=example.id,
                    question=example.question,
                    expected_sources=example.expected_sources,
                    retrieved_sources=[],
                    hit=False,
                    error=str(error),
                )

            results.append(result)

        self._write_report(results)
        self._print_summary(results)

        return results

    def _extract_sources(
        self,
        sources: list[dict],
    ) -> list[str]:
        names: list[str] = []

        for source in sources:
            metadata = source.get("metadata", {})
            file_name = str(metadata.get("file_name", "")).strip()

            if file_name:
                names.append(file_name)

        return names

    def _has_hit(
        self,
        expected_sources: list[str],
        retrieved_sources: list[str],
    ) -> bool:
        expected = set(expected_sources)
        retrieved = set(retrieved_sources)

        return bool(expected.intersection(retrieved))

    def _write_report(
        self,
        results: list[RetrievalCheckResult],
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        with self.report_path.open("w", encoding="utf-8") as file:
            for result in results:
                file.write(
                    json.dumps(
                        result.__dict__,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"[retrieval] Report written: {self.report_path}",
            flush=True,
        )

    def _print_summary(
        self,
        results: list[RetrievalCheckResult],
    ) -> None:
        if not results:
            print("[retrieval] No retrieval results.", flush=True)
            return

        hits = sum(1 for result in results if result.hit)
        errors = sum(1 for result in results if result.error)
        hit_rate = hits / len(results)

        print("", flush=True)
        print("Retrieval summary", flush=True)
        print("-" * 80, flush=True)
        print(f"Examples: {len(results)}", flush=True)
        print(f"Hits: {hits}", flush=True)
        print(f"Errors: {errors}", flush=True)
        print(f"Hit rate: {hit_rate:.4f}", flush=True)
