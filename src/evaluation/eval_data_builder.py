from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import shutil


@dataclass(frozen=True)
class EvalFile:
    """File selected for evaluation."""

    source_type: str
    file_name: str


@dataclass(frozen=True)
class EvalQuestion:
    """Single evaluation question."""

    id: str
    question: str
    expected_answer: str
    expected_sources: list[str]


class EvalDataBuilder:
    """Builds a small evaluation dataset from raw source files."""

    DEFAULT_EVAL_FILES = [
        EvalFile(
            source_type="markdown",
            file_name="generic-engineering-attribution-challenge.md",
        ),
        EvalFile(
            source_type="markdown",
            file_name="kitti-yolo-detector.md",
        ),
        EvalFile(
            source_type="markdown",
            file_name="minetest-assistant.md",
        ),
        EvalFile(
            source_type="markdown",
            file_name="tg-ml-contest.md",
        ),
        EvalFile(
            source_type="plaintext",
            file_name="nfl_impact_detection.txt",
        ),
    ]

    DEFAULT_QUESTIONS = [
        EvalQuestion(
            id="q001",
            question="What was the goal of the Genetic Engineering Attribution Challenge project?",
            expected_answer=(
                "The goal was to identify the lab of origin for genetically engineered DNA "
                "with the highest possible accuracy."
            ),
            expected_sources=["generic-engineering-attribution-challenge.md"],
        ),
        EvalQuestion(
            id="q002",
            question="What main feature was added to the Genetic Engineering Attribution Challenge benchmark?",
            expected_answer=(
                "The benchmark was extended with more N-Grams of the ACGT sequence, "
                "including six-grams, as input features."
            ),
            expected_sources=["generic-engineering-attribution-challenge.md"],
        ),
        EvalQuestion(
            id="q003",
            question="What dataset is used by the YOLO KITTI Detector project?",
            expected_answer=(
                "The project uses the KITTI dataset, including left images, labels, "
                "and a devkit script for evaluation."
            ),
            expected_sources=["kitti-yolo-detector.md"],
        ),
        EvalQuestion(
            id="q004",
            question="What model family is used in the YOLO KITTI Detector project?",
            expected_answer="The project uses YOLO detections with a simple tracker.",
            expected_sources=["kitti-yolo-detector.md"],
        ),
        EvalQuestion(
            id="q005",
            question="What is the purpose of the Minetest Assistant project?",
            expected_answer=(
                "The project detects Minetest MineClone2 mobs to make the game simpler."
            ),
            expected_sources=["minetest-assistant.md"],
        ),
        EvalQuestion(
            id="q006",
            question="Which Python packages are mentioned for installing Minetest Assistant?",
            expected_answer=(
                "The installation command mentions PyYaml, mss, pynput, and ultralytics."
            ),
            expected_sources=["minetest-assistant.md"],
        ),
        EvalQuestion(
            id="q007",
            question="What is the task of the Telegram ML Competition 2023 project?",
            expected_answer=(
                "The task is to create a library that detects the programming or markup "
                "language of a code snippet."
            ),
            expected_sources=["tg-ml-contest.md"],
        ),
        EvalQuestion(
            id="q008",
            question="How was the dataset created for the Telegram ML Competition 2023 project?",
            expected_answer=(
                "The dataset contains code samples and user questions, with code samples "
                "downloaded from GitHub using the GitHub REST API."
            ),
            expected_sources=["tg-ml-contest.md"],
        ),
        EvalQuestion(
            id="q009",
            question="What Kaggle competition is described in the NFL impact detection solution?",
            expected_answer=(
                "It describes the NFL 1st and Future - Impact Detection competition."
            ),
            expected_sources=["nfl_impact_detection.txt"],
        ),
        EvalQuestion(
            id="q010",
            question="What model was used in the NFL impact detection solution?",
            expected_answer=(
                "The solution used Tiny-YOLOv4 from AlexeyAB's darknet repository."
            ),
            expected_sources=["nfl_impact_detection.txt"],
        ),
    ]

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        eval_dir: str | Path = "data/eval",
        copy_files: bool = True,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.eval_dir = Path(eval_dir)
        self.copy_files = copy_files

    def build(self) -> None:
        self._prepare_eval_dirs()
        self._place_eval_files()
        self._write_questions()

    def _prepare_eval_dirs(self) -> None:
        (self.eval_dir / "markdown").mkdir(parents=True, exist_ok=True)
        (self.eval_dir / "plaintext").mkdir(parents=True, exist_ok=True)

    def _place_eval_files(self) -> None:
        for eval_file in self.DEFAULT_EVAL_FILES:
            source_path = self._find_raw_file(eval_file)
            target_path = (
                self.eval_dir
                / eval_file.source_type
                / eval_file.file_name
            )

            if source_path is None:
                print(f"Missing eval file: {eval_file.file_name}")
                continue

            if self.copy_files:
                shutil.copy2(source_path, target_path)
            else:
                shutil.move(source_path, target_path)

            print(f"Placed eval file: {target_path}")

    def _find_raw_file(self, eval_file: EvalFile) -> Path | None:
        candidates = [
            self.raw_dir / eval_file.source_type / eval_file.file_name,
            self.raw_dir / eval_file.file_name,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = list(self.raw_dir.rglob(eval_file.file_name))

        if matches:
            return matches[0]

        return None

    def _write_questions(self) -> None:
        questions_path = self.eval_dir / "test_questions.jsonl"

        with questions_path.open("w", encoding="utf-8") as file:
            for question in self.DEFAULT_QUESTIONS:
                file.write(
                    json.dumps(
                        asdict(question),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(f"Written eval questions: {questions_path}")
