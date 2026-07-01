from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class EvaluationExample:
    """Single evaluation example."""

    id: str
    question: str
    expected_answer: str
    expected_sources: list[str]


class EvaluationDataset:
    """Loads evaluation examples from JSONL."""

    def __init__(
        self,
        questions_path: str | Path = "data/eval/test_questions.jsonl",
    ) -> None:
        self.questions_path = Path(questions_path)

    def load(self) -> list[EvaluationExample]:
        if not self.questions_path.exists():
            raise FileNotFoundError(
                f"Evaluation file not found: {self.questions_path}"
            )

        examples: list[EvaluationExample] = []

        with self.questions_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                data = json.loads(line)

                examples.append(
                    EvaluationExample(
                        id=str(data.get("id", f"q{line_number:03d}")),
                        question=str(data.get("question", "")),
                        expected_answer=str(data.get("expected_answer", "")),
                        expected_sources=list(data.get("expected_sources", [])),
                    )
                )

        return examples
