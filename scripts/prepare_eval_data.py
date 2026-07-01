from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation import EvalDataBuilder


def main() -> None:
    args = parse_args()

    builder = EvalDataBuilder(
        raw_dir=args.raw_dir,
        eval_dir=args.eval_dir,
        copy_files=not args.move,
    )

    builder.build()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare small RAG evaluation dataset.",
    )

    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Raw dataset directory.",
    )
    parser.add_argument(
        "--eval-dir",
        default="data/eval",
        help="Evaluation dataset directory.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
