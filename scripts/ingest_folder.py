"""Bulk-ingest every PDF in a local folder (synchronously) - e.g. the policies already on your machine.

    python scripts/ingest_folder.py --folder "C:/Users/me/Documents/Policies" --department HR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

setup_logging(get_settings().log_dir, "INFO")

from app.core.exceptions import AppError  # noqa: E402
from app.db.models import Policy  # noqa: E402
from app.db.session import SessionLocal, init_db  # noqa: E402
from app.ingestion.pipeline import register_policy, run_ingestion  # noqa: E402
from app.services import audit  # noqa: E402
from app.services.audit import Actions  # noqa: E402


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
    ap.add_argument("--folder", required=True)
    ap.add_argument("--department", default="General")
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--by", default="cli-admin")
    args = ap.parse_args()
    init_db()
    if not _check_vector_store():
        return 1
    pdfs = sorted(Path(args.folder).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.folder}")
        return 1
    failures = 0
    for pdf in pdfs:
        with SessionLocal() as db:
            try:
                policy, job = register_policy(db, filename=pdf.name, content=pdf.read_bytes(),
                                              name=pdf.stem.replace("_", " ").replace("-", " "),
                                              department=args.department, version=args.version, effective_date="",
                                              description="", tags=[], uploaded_by=args.by)
                audit.record(Actions.POLICY_UPLOAD, db=db, username=args.by, role="admin",
                             resource=f"policy:{policy.id}", details={"source": "cli", "file": pdf.name})
            except AppError as exc:
                print(f"SKIP  {pdf.name}: {exc.user_message}")
                continue
        run_ingestion(policy.id, job.id, args.by)
        with SessionLocal() as db:
            p = db.get(Policy, policy.id)
            ok = p.status == "active"
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'}  {pdf.name}: {p.chunk_count} chunks, {p.page_count} pages {p.error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
