from __future__ import annotations

from evals.deepeval_rag import DeepEvalRAGRunner


class AnswerQualityEvaluation:
    """Runs answer quality evaluation through DeepEval."""

    def __init__(
        self,
        runner: DeepEvalRAGRunner,
    ) -> None:
        self.runner = runner

    def run(self):
        return self.runner.run()
