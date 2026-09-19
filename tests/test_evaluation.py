"""Evaluation runner control-flow test (DeepEval metrics replaced by a dummy; real runs use the Ollama judge)."""
from __future__ import annotations

import json
from pathlib import Path


class DummyMetric:
    def __init__(self, score: float) -> None:
        self.score, self.reason = score, "dummy"

    def measure(self, tc) -> float:
        assert tc.input and tc.actual_output is not None and tc.retrieval_context
        return self.score


def test_runner(client, monkeypatch):
    from app.evaluation import runner

    # requires the policies ingested by test_api_e2e (same session DB)
    monkeypatch.setattr(runner, "build_metrics", lambda judge, answerable: {
        "faithfulness": DummyMetric(0.9), "correctness": DummyMetric(0.8), "hallucination": DummyMetric(0.1),
        "prompt_alignment": DummyMetric(1.0)})
    summary = runner.run_evaluation(None, None)
    assert summary["metrics"]["faithfulness"]["mean"] == 0.9
    assert summary["metrics"]["hallucination"]["direction"] == "lower_is_better"
    assert summary["guardrail_accuracy"] == 1.0
    assert summary["retrieval"]["hit_rate"] is not None
    assert summary["refusal_accuracy"] is not None
    reports = sorted(Path(runner.settings.eval_reports_dir).glob("eval_*.json"))
    assert reports and json.loads(reports[-1].read_text())["summary"]["cases"] == summary["cases"]
