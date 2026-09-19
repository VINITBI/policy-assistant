from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, client_ip, get_current, require_admin
from app.api.schemas import JobOut, PolicyBrief, PolicyOut, UploadResponse
from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.db.models import IngestionJob, Policy
from app.db.session import get_db
from app.ingestion import pipeline as ingestion
from app.services import audit
from app.services.audit import Actions

settings = get_settings()
router = APIRouter(tags=["Policies"])


@router.get("/policies", response_model=list[PolicyBrief], summary="Active policies (for the user's policy selector)")
def list_active(_: CurrentUser = Depends(get_current), db: Session = Depends(get_db)):
    return db.scalars(select(Policy).where(Policy.status == "active").order_by(Policy.name)).all()


# ------------------------- admin only -------------------------
@router.post("/admin/policies", response_model=UploadResponse, status_code=202,
             summary="Upload a policy PDF (admin) - ingestion runs in the background")
async def upload_policy(request: Request, background: BackgroundTasks, file: UploadFile = File(...),
                        policy_name: str = Form(""), department: str = Form("General"), version: str = Form("1.0"),
                        effective_date: str = Form(""), description: str = Form(""), tags: str = Form(""),
                        admin: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    policy, job = ingestion.register_policy(
        db, filename=file.filename or "", content=content, name=policy_name, department=department, version=version,
        effective_date=effective_date, description=description, tags=tags.split(","), uploaded_by=admin.user.username)
    audit.record(Actions.POLICY_UPLOAD, db=db, user_id=admin.user.id, username=admin.user.username, role="admin",
                 ip=client_ip(request), resource=f"policy:{policy.id}",
                 details={"name": policy.name, "file": policy.file_name, "bytes": policy.file_size,
                          "version": policy.version})
    background.add_task(ingestion.run_ingestion, policy.id, job.id, admin.user.username)
    return UploadResponse(policy_id=policy.id, job_id=job.id, status="processing",
                          message=f"'{policy.name}' uploaded. Indexing has started - it will be searchable shortly.")


@router.post("/admin/policies/ingest-inbox", summary="Bulk-ingest every PDF placed in data/inbox (admin)")
def ingest_inbox(request: Request, background: BackgroundTasks, admin: CurrentUser = Depends(require_admin),
                 db: Session = Depends(get_db)):
    queued, skipped = ingestion.register_inbox(db, admin.user.username)
    for pid, jid in queued:
        audit.record(Actions.POLICY_UPLOAD, db=db, user_id=admin.user.id, username=admin.user.username, role="admin",
                     ip=client_ip(request), resource=f"policy:{pid}", details={"source": "inbox"})
        background.add_task(ingestion.run_ingestion, pid, jid, admin.user.username)
    if not queued and not skipped:
        raise ValidationError(f"No PDF files found in {settings.inbox_dir}.")
    return {"queued": len(queued), "skipped": skipped}


@router.get("/admin/policies", response_model=list[PolicyOut], summary="All policies incl. status (admin)")
def list_all(_: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    return db.scalars(select(Policy).where(Policy.status != "deleted").order_by(Policy.uploaded_at.desc())).all()


@router.delete("/admin/policies/{policy_id}", summary="Delete a policy and its vectors (admin)")
def delete(policy_id: str, request: Request, admin: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    policy = ingestion.delete_policy(db, policy_id)
    audit.record(Actions.POLICY_DELETE, db=db, user_id=admin.user.id, username=admin.user.username, role="admin",
                 ip=client_ip(request), resource=f"policy:{policy_id}", details={"name": policy.name})
    return {"message": f"Policy '{policy.name}' deleted."}


@router.post("/admin/policies/{policy_id}/reindex", response_model=UploadResponse, status_code=202,
             summary="Re-run ingestion for a policy (admin)")
def reindex(policy_id: str, request: Request, background: BackgroundTasks, admin: CurrentUser = Depends(require_admin),
            db: Session = Depends(get_db)):
    job = ingestion.prepare_reindex(db, policy_id, admin.user.username)
    audit.record(Actions.POLICY_REINDEX, db=db, user_id=admin.user.id, username=admin.user.username, role="admin",
                 ip=client_ip(request), resource=f"policy:{policy_id}")
    background.add_task(ingestion.run_ingestion, policy_id, job.id, admin.user.username)
    return UploadResponse(policy_id=policy_id, job_id=job.id, status="processing", message="Re-indexing started.")


@router.get("/admin/ingestion-jobs", response_model=list[JobOut], summary="Ingestion job history (admin)")
def jobs(limit: int = 50, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    return db.scalars(select(IngestionJob).order_by(IngestionJob.started_at.desc().nulls_first()).limit(limit)).all()
