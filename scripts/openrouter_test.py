from pathlib import Path
import os
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.llm import OpenRouterClient


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    model = os.getenv("OPENROUTER_MODEL", "")

    client = OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
    )

    prompt = _read_prompt()

    response = client.generate_text(
        prompt=prompt,
        system_prompt="You are a concise technical assistant.",
    )

    print()
    print("OpenRouter response")
    print("-" * 80)
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")
    print("-" * 80)
    print(response.text)


def _read_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    return input("Prompt: ").strip()


if __name__ == "__main__":
    main()
