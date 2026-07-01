from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def disable_telemetry() -> None:
    """Disables optional telemetry used by evaluation packages."""

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("POSTHOG_DISABLED", "true")
    os.environ.setdefault("DO_NOT_TRACK", "1")


disable_telemetry()


from evals import DeepEvalRAGRunner, EvaluationDataset, RAGAPIClient
from evals import RetrievalEvaluation
from src.llm import GroqLLMClient


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()

    print("[evals] Starting evaluation", flush=True)
    print(f"[evals] mode={args.mode}", flush=True)
    print(f"[evals] api_url={args.api_url}", flush=True)
    print(f"[evals] questions_path={args.questions_path}", flush=True)
    print(f"[evals] api_timeout={args.timeout}", flush=True)
    print(f"[evals] generation_timeout={args.generation_timeout}", flush=True)
    print(f"[evals] top_k={args.top_k}", flush=True)

    if args.mode in {"deepeval", "all"}:
        _reset_report(args.deepeval_report_path)

    if args.mode in {"retrieval", "all"}:
        _reset_report(args.retrieval_report_path)

    dataset = EvaluationDataset(
        questions_path=args.questions_path,
    )

    api_client = RAGAPIClient(
        api_url=args.api_url,
        timeout=args.timeout,
        verbose=True,
    )

    print("[evals] Checking API health...", flush=True)
    api_client.health()

    if args.mode in {"retrieval", "all"}:
        print("[evals] Running retrieval evaluation...", flush=True)

        retrieval_eval = RetrievalEvaluation(
            dataset=dataset,
            api_client=api_client,
            report_path=args.retrieval_report_path,
            top_k=args.top_k,
        )
        retrieval_eval.run()

    if args.mode in {"deepeval", "all"}:
        print("[evals] Preparing Groq generator...", flush=True)

        generator_model = args.generation_model

        generator_client = GroqLLMClient(
            api_key=args.groq_api_key,
            model=generator_model,
            fallback_models=_parse_models(args.generation_fallback_models),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            reasoning_effort=args.reasoning_effort,
            stream=False,
            strip_reasoning=True,
            max_retries=args.llm_max_retries,
            retry_sleep_seconds=args.llm_retry_sleep_seconds,
            timeout=args.generation_timeout,
        )

        print("[evals] Running DeepEval evaluation...", flush=True)

        deepeval_runner = DeepEvalRAGRunner(
            dataset=dataset,
            api_client=api_client,
            generator_client=generator_client,
            generator_model_name=generator_model,
            report_path=args.deepeval_report_path,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            threshold=args.threshold,
        )
        deepeval_runner.run()

    print("[evals] Evaluation finished.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluations.",
    )

    parser.add_argument(
        "--mode",
        choices=["retrieval", "deepeval", "all"],
        default="retrieval",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--questions-path",
        default="data/eval/test_questions.jsonl",
        help="Path to evaluation questions JSONL.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("EVAL_API_URL", "http://localhost:8000"),
        help="FastAPI RAG service URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("EVAL_REQUEST_TIMEOUT", "30")),
        help="Timeout for /health and /retrieve requests.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("EVAL_TOP_K", "3")),
        help="Number of retrieved chunks.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("EVAL_TEMPERATURE", "0.6")),
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("EVAL_MAX_TOKENS", "4096")),
        help="Max completion tokens.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("EVAL_TOP_P", "0.95")),
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("EVAL_REASONING_EFFORT", "default"),
        help="Groq reasoning effort.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("EVAL_THRESHOLD", "0.7")),
        help="DeepEval metric threshold.",
    )
    parser.add_argument(
        "--deepeval-report-path",
        default="data/eval/reports/deepeval_results.jsonl",
        help="DeepEval report JSONL path.",
    )
    parser.add_argument(
        "--retrieval-report-path",
        default="data/eval/reports/retrieval_results.jsonl",
        help="Retrieval report JSONL path.",
    )

    parser.add_argument(
        "--groq-api-key",
        default=os.getenv("GROQ_API_KEY", ""),
        help="Groq API key.",
    )
    parser.add_argument(
        "--generation-model",
        default=os.getenv(
            "EVAL_GENERATION_MODEL",
            os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
        ),
        help="Groq model for answer generation.",
    )
    parser.add_argument(
        "--generation-fallback-models",
        default=os.getenv(
            "EVAL_GENERATION_FALLBACK_MODELS",
            os.getenv("GROQ_FALLBACK_MODELS", ""),
        ),
        help="Comma-separated fallback generation models.",
    )
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=float(os.getenv("EVAL_GENERATION_TIMEOUT", "180")),
        help="Groq timeout for answer generation.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=int(os.getenv("EVAL_LLM_MAX_RETRIES", "1")),
        help="Retries for Groq eval calls.",
    )
    parser.add_argument(
        "--llm-retry-sleep-seconds",
        type=float,
        default=float(os.getenv("EVAL_LLM_RETRY_SLEEP_SECONDS", "2.0")),
        help="Sleep between Groq eval retries.",
    )

    return parser.parse_args()


def _parse_models(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _reset_report(path: str) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.exists():
        report_path.unlink()

    print(f"[evals] Reset report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
