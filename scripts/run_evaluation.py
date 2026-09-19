"""Run the evaluation suite from the command line (also usable as a CI quality gate).

    python scripts/run_evaluation.py                 # all golden cases
    python scripts/run_evaluation.py --max-cases 3   # quick smoke run
    python scripts/run_evaluation.py --fail-under 0.7  # exit 1 if faithfulness/correctness mean is lower
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

setup_logging(get_settings().log_dir, "INFO")

from app.db.models import EvaluationRun  # noqa: E402
from app.db.session import db_scope, init_db  # noqa: E402
from app.evaluation.runner import run_evaluation  # noqa: E402


def _check_vector_store() -> bool:
    """Open the vector store up front so a locked local database gives a clear message, not a traceback."""
    from app.core.exceptions import AppError
    from app.db.vector_store import get_vector_store
    try:
        get_vector_store()
        return True
    except AppError as exc:
        print(f"ERROR: {exc.user_message}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--fail-under", type=float, default=None)
    args = ap.parse_args()
    init_db()
    if not _check_vector_store():
        return 1
    with db_scope() as db:
        run = EvaluationRun(triggered_by="cli")
        db.add(run)
        db.flush()
        run_id = run.id
    summary = run_evaluation(run_id, args.max_cases)
    print(json.dumps(summary, indent=2))
    if args.fail_under is not None:
        gate = [summary["metrics"].get(m, {}).get("mean", 0) for m in ("faithfulness", "correctness")]
        if min(gate) < args.fail_under:
            print(f"QUALITY GATE FAILED: {gate} < {args.fail_under}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
