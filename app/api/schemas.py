from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    role: str
    username: str
    full_name: str
    login_session_id: str


class UserOut(ORM):
    id: str
    username: str
    full_name: str
    role: str
    is_active: bool
    last_login_at: datetime | None = None


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(default="user", pattern="^(admin|user)$")
    full_name: str = ""


class PolicyOut(ORM):
    id: str
    name: str
    department: str
    version: str
    effective_date: str
    description: str
    tags: list[str]
    status: str
    page_count: int
    chunk_count: int
    uploaded_by: str
    uploaded_at: datetime
    indexed_at: datetime | None = None
    error: str = ""


class PolicyBrief(ORM):
    id: str
    name: str
    department: str
    version: str
    effective_date: str


class UploadResponse(BaseModel):
    policy_id: str
    job_id: str
    status: str
    message: str


class JobOut(ORM):
    id: str
    policy_id: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    stats: dict
    error: str
    triggered_by: str
    trace_id: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    policy_id: str | None = Field(default=None, description="Policy id, or null/'all' to search every policy")
    session_id: str | None = Field(default=None, description="Omit to start a new conversation")


class Source(BaseModel):
    ref: str
    policy_id: str | None = None
    policy_name: str | None = None
    version: str | None = None
    section: str = ""
    pages: str = ""
    score: float | None = None
    snippet: str = ""
    chunk_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    answer_type: str
    confidence: str
    sources: list[Source]
    warnings: list[str]
    standalone_query: str
    trace_id: str
    latency_ms: float


class SessionOut(ORM):
    id: str
    title: str
    policy_id: str | None
    created_at: datetime
    updated_at: datetime


class MessageOut(ORM):
    id: str
    role: str
    content: str
    policy_id: str | None
    sources: list[Any]
    answer_type: str
    confidence: str
    feedback: int | None
    self_eval: dict
    trace_id: str
    created_at: datetime


class FeedbackRequest(BaseModel):
    rating: int = Field(description="1 = helpful, -1 = not helpful")
    comment: str = ""


class AuditOut(ORM):
    id: int
    ts: datetime
    username: str
    role: str
    action: str
    resource: str
    outcome: str
    details: dict
    ip_address: str
    trace_id: str
    login_session_id: str


class LoginSessionOut(ORM):
    id: str
    username: str
    role: str
    login_at: datetime
    last_seen_at: datetime
    expires_at: datetime
    logout_at: datetime | None
    status: str
    ip_address: str
    user_agent: str
    request_count: int


class TraceOut(ORM):
    trace_id: str
    name: str
    username: str
    total_ms: float
    status: str
    spans: list[Any]
    created_at: datetime


class EvalRunOut(ORM):
    id: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    triggered_by: str
    summary: dict
    report_path: str
    error: str
