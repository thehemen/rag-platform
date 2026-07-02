from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.llm import GroqLLMClient


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    client = GroqLLMClient(
        api_key=os.getenv("GROQ_API_KEY") or None,
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
        fallback_models=_parse_models(os.getenv("GROQ_FALLBACK_MODELS", "")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.6")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        top_p=float(os.getenv("LLM_TOP_P", "0.95")),
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "default"),
        stream=os.getenv("LLM_STREAM", "false").lower()
        in {"1", "true", "yes", "y"},
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        retry_sleep_seconds=float(os.getenv("LLM_RETRY_SLEEP_SECONDS", "2.0")),
        timeout=float(os.getenv("LLM_TIMEOUT", "180")),
    )

    prompt = _read_prompt()

    response = client.generate_text(
        prompt=prompt,
        system_prompt="You are a concise technical assistant.",
    )

    print()
    print("Groq response")
    print("-" * 80)
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")
    print("-" * 80)
    print(response.text)


def _read_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    return input("Prompt: ").strip()


def _parse_models(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


if __name__ == "__main__":
    main()
