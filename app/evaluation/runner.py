"""Evaluation suite (DeepEval + deterministic metrics).

For every golden case the FULL retrieval pipeline is executed and scored on:

  Retrieval (DeepEval)   : contextual_precision, contextual_recall, contextual_relevancy
  Retrieval (no LLM)     : hit_rate@k, MRR, answer keyword recall
  Generation (DeepEval)  : answer_relevancy, faithfulness, hallucination
  LLM quality (DeepEval) : correctness (G-Eval vs expected answer), citation_quality (G-Eval),
                           professional_tone (G-Eval), bias, toxicity
  Prompt evaluation      : prompt_alignment (DeepEval PromptAlignmentMetric against the instructions in
                           prompts.yaml) + the runtime LLM self-evaluation score (prompt-based self-grading)
  Safety                 : refusal accuracy on unanswerable questions, guardrail block accuracy

Results -> eval/reports/eval_<timestamp>.json/.csv and the `evaluation_runs` table.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import yaml
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.exceptions import GuardrailViolation
from app.core.logging import get_logger
from app.db.models import EvaluationRun, Policy
from app.db.session import db_scope
from app.guardrails.input_guard import InputGuard
from app.llm.prompts import get_prompts
from app.retrieval.pipeline import RAGPipeline
from app.services.chat_service import compute_self_eval

log = get_logger("evaluation")
settings = get_settings()
LOWER_IS_BETTER = {"hallucination", "bias", "toxicity"}


def _params():
    try:
        from deepeval.test_case import SingleTurnParams as P
    except ImportError:  # older deepeval
        from deepeval.test_case import LLMTestCaseParams as P
    return P


def build_metrics(judge, answerable: bool) -> dict:
    from deepeval.metrics import (AnswerRelevancyMetric, BiasMetric, ContextualPrecisionMetric,
                                  ContextualRecallMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval,
                                  HallucinationMetric, PromptAlignmentMetric, ToxicityMetric)
    P = _params()
    th = settings.eval_threshold
    common = {"model": judge, "async_mode": False}
    metrics = {
        "prompt_alignment": PromptAlignmentMetric(prompt_instructions=get_prompts().instructions, threshold=th, **common),
        "toxicity": ToxicityMetric(threshold=0.5, **common),
    }
    if answerable:
        metrics |= {
            "contextual_precision": ContextualPrecisionMetric(threshold=th, **common),
            "contextual_recall": ContextualRecallMetric(threshold=th, **common),
            "contextual_relevancy": ContextualRelevancyMetric(threshold=th, **common),
            "answer_relevancy": AnswerRelevancyMetric(threshold=th, **common),
            "faithfulness": FaithfulnessMetric(threshold=th, **common),
            "hallucination": HallucinationMetric(threshold=0.5, **common),
            "bias": BiasMetric(threshold=0.5, **common),
            "correctness": GEval(
                name="Correctness", threshold=th, **common,
                evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
                evaluation_steps=[
                    "Check whether the facts in the actual output agree with the expected output.",
                    "Heavily penalise contradicted or missing numbers, limits, dates or eligibility rules.",
                    "Extra correct detail is acceptable; vague language is penalised lightly."]),
            "citation_quality": GEval(
                name="Citation quality", threshold=th, **common,
                evaluation_params=[P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT],
                evaluation_steps=[
                    "Check that factual statements in the actual output carry [S#] citation tags.",
                    "Check that cited statements are supported by the retrieval context.",
                    "Penalise uncited factual claims and citations that do not support the claim."]),
            "professional_tone": GEval(
                name="Professional tone", threshold=th, **common, evaluation_params=[P.ACTUAL_OUTPUT],
                criteria="The response is professional, neutral, concise and clearly structured for an employee."),
        }
    return metrics


def _retrieval_scores(contexts: list[str], answer: str, keywords: list[str]) -> dict:
    kws = [k.lower() for k in keywords if k.strip()]
    if not kws:
        return {}
    rank = next((i for i, c in enumerate(contexts, 1) if any(k in c.lower() for k in kws)), None)
    return {"hit@k": 1.0 if rank else 0.0, "mrr": round(1 / rank, 4) if rank else 0.0,
            "keyword_recall": round(sum(k in answer.lower() for k in kws) / len(kws), 4)}


def run_evaluation(run_id: str | None = None, max_cases: int | None = None) -> dict:
    from deepeval.test_case import LLMTestCase

    from app.evaluation.judge import OllamaJudge

    started = time.perf_counter()
    try:
        data = yaml.safe_load(Path(settings.eval_dataset).read_text(encoding="utf-8")) or {}
        cases = (data.get("cases") or [])[: max_cases or None]
        with db_scope() as db:
            policies = {p.name.lower(): p.id for p in db.scalars(select(Policy).where(Policy.status == "active"))}
        judge = OllamaJudge()
        pipeline = RAGPipeline()
        rows: list[dict] = []

        for i, case in enumerate(cases, 1):
            answerable = case.get("type", "answerable") == "answerable"
            pname = (case.get("policy") or "all").lower()
            policy_id = None if pname == "all" else policies.get(pname)
            row = {"id": case.get("id", f"case-{i}"), "question": case["question"], "policy": case.get("policy", "all"),
                   "type": "answerable" if answerable else "unanswerable"}
            if pname != "all" and policy_id is None:
                row["error"] = f"policy '{case.get('policy')}' not found / not active"
                rows.append(row)
                continue
            log.info("evaluating case %s/%s", i, len(cases), extra={"case": row["id"]})
            t0 = time.perf_counter()
            try:
                res = pipeline.run(case["question"], policy_id, history="")
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"pipeline: {exc}"
                rows.append(row)
                continue
            row.update(answer=res.answer, answer_type=res.answer_type, confidence=res.confidence,
                       latency_ms=round((time.perf_counter() - t0) * 1000), top_rerank=res.top_score,
                       groundedness=res.groundedness)

            if not answerable:
                row["refusal_correct"] = 1.0 if res.answer_type in ("no_context", "withheld") else 0.0
            else:
                row.update(_retrieval_scores(res.retrieval_context, res.answer, case.get("expected_keywords", [])))
                if res.context and res.answer_type == "answered":
                    try:
                        row["self_eval"] = compute_self_eval(case["question"], res.context, res.answer)["overall"]
                    except Exception as exc:  # noqa: BLE001
                        log.warning("self-eval failed in evaluation", extra={"error": str(exc)})

            tc = LLMTestCase(input=case["question"], actual_output=res.answer,
                             expected_output=case.get("expected_answer") or None,
                             retrieval_context=res.retrieval_context or ["(no context retrieved)"],
                             context=[case.get("expected_answer") or "N/A"])
            for name, metric in build_metrics(judge, answerable).items():
                try:
                    metric.measure(tc)
                    row[name] = round(float(metric.score), 4)
                    row[f"{name}_reason"] = (metric.reason or "")[:300]
                except Exception as exc:  # noqa: BLE001
                    row[f"{name}_error"] = str(exc)[:200]
                    log.warning("metric failed", extra={"metric": name, "case": row["id"], "error": str(exc)})
            rows.append(row)

        # ---- guardrail cases (no LLM) ----
        guard = InputGuard()
        g_rows = []
        for j, g in enumerate(data.get("guardrail_cases") or []):
            try:
                guard.check(g["input"], user_key=f"eval-{j}")
                blocked = False
            except GuardrailViolation:
                blocked = True
            g_rows.append({"input": g["input"], "expected": g["expect"],
                           "correct": blocked == (g["expect"] == "block")})

        summary = summarize(rows, g_rows)
        summary["duration_s"] = round(time.perf_counter() - started, 1)
        report = write_report(rows, g_rows, summary)
        _finish(run_id, "succeeded", summary, str(report))
        log.info("evaluation finished", extra={"report": str(report), "cases": len(rows)})
        return summary
    except Exception as exc:  # noqa: BLE001
        log.exception("evaluation failed")
        _finish(run_id, "failed", {}, "", error=f"{type(exc).__name__}: {exc}")
        raise


def summarize(rows: list[dict], g_rows: list[dict]) -> dict:
    metric_names = ["contextual_precision", "contextual_recall", "contextual_relevancy", "answer_relevancy",
                    "faithfulness", "hallucination", "correctness", "citation_quality", "professional_tone", "bias",
                    "toxicity", "prompt_alignment"]
    metrics = {}
    for m in metric_names:
        vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
        if vals:
            passed = [v <= 0.5 if m in LOWER_IS_BETTER else v >= settings.eval_threshold for v in vals]
            metrics[m] = {"mean": round(mean(vals), 4), "pass_rate": round(sum(passed) / len(passed), 4),
                          "n": len(vals), "direction": "lower_is_better" if m in LOWER_IS_BETTER else "higher_is_better"}

    def avg(key: str):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(mean(vals), 4) if vals else None

    with db_scope() as db:
        policy_versions = dict(db.execute(select(Policy.name, Policy.version).where(Policy.status == "active")).all())
        n_active = db.scalar(select(func.count()).select_from(Policy).where(Policy.status == "active"))
    return {
        "cases": len(rows), "errors": sum(1 for r in rows if r.get("error")),
        "metrics": metrics,
        "retrieval": {"hit_rate": avg("hit@k"), "mrr": avg("mrr"), "answer_keyword_recall": avg("keyword_recall")},
        "prompt_self_eval_mean": avg("self_eval"),
        "refusal_accuracy": avg("refusal_correct"),
        "guardrail_accuracy": round(sum(g["correct"] for g in g_rows) / len(g_rows), 4) if g_rows else None,
        "avg_latency_ms": avg("latency_ms"),
        "config": {"llm": settings.llm_model, "judge": settings.eval_judge_model, "embedding": settings.embedding_model,
                   "reranker": settings.reranker_model, "prompt_version": get_prompts().version,
                   "child_chunk_tokens": settings.child_chunk_tokens, "parent_chunk_tokens": settings.parent_chunk_tokens,
                   "rerank_top_n": settings.rerank_top_n, "multi_query": settings.multi_query_enabled,
                   "active_policies": n_active, "policy_versions": policy_versions},
    }


def write_report(rows: list[dict], g_rows: list[dict], summary: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(settings.eval_reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{ts}.json"
    path.write_text(json.dumps({"summary": summary, "cases": rows, "guardrail_cases": g_rows}, indent=2, default=str),
                    encoding="utf-8")
    keys = sorted({k for r in rows for k in r if not k.endswith("_reason")})
    with open(out / f"eval_{ts}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _finish(run_id: str | None, status: str, summary: dict, report: str, error: str = "") -> None:
    if not run_id:
        return
    with db_scope() as db:
        run = db.get(EvaluationRun, run_id)
        if run:
            run.status, run.summary, run.report_path, run.error = status, summary, report, error[:2000]
            run.finished_at = datetime.now(timezone.utc)
