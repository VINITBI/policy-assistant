"""Thin HTTP client used by the Streamlit UI. Converts API errors into friendly messages."""
from __future__ import annotations

import os
from typing import Any

from pathlib import Path

import httpx

try:  # read API_BASE_URL from the project's .env (python-dotenv ships with pydantic-settings)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover
    pass

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/") + "/api/v1"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 0, code: str = "", trace_id: str = "") -> None:
        super().__init__(message)
        self.message, self.status, self.code, self.trace_id = message, status, code, trace_id


class ApiClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def _request(self, method: str, path: str, timeout: float = 30, **kwargs) -> Any:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            r = httpx.request(method, API_BASE + path, headers=headers, timeout=timeout, **kwargs)
        except httpx.ConnectError:
            raise ApiError("Cannot reach the Policy Assistant service. Please make sure the API is running.")
        except httpx.TimeoutException:
            raise ApiError("The request took too long. The assistant may be busy - please try again.")
        if r.status_code >= 400:
            try:
                err = r.json().get("error", {})
            except ValueError:
                err = {}
            raise ApiError(err.get("message") or f"Request failed ({r.status_code}).", r.status_code,
                           err.get("code", ""), err.get("trace_id", ""))
        return r.json() if r.content else None

    # auth
    def login(self, username: str, password: str) -> dict:
        return self._request("POST", "/auth/login", json={"username": username, "password": password})

    def logout(self) -> None:
        self._request("POST", "/auth/logout")

    # policies
    def policies(self) -> list[dict]:
        return self._request("GET", "/policies")

    def admin_policies(self) -> list[dict]:
        return self._request("GET", "/admin/policies")

    def upload_policy(self, file_name: str, data: bytes, meta: dict) -> dict:
        return self._request("POST", "/admin/policies", timeout=120,
                             files={"file": (file_name, data, "application/pdf")}, data=meta)

    def ingest_inbox(self) -> dict:
        return self._request("POST", "/admin/policies/ingest-inbox")

    def delete_policy(self, policy_id: str) -> dict:
        return self._request("DELETE", f"/admin/policies/{policy_id}")

    def reindex_policy(self, policy_id: str) -> dict:
        return self._request("POST", f"/admin/policies/{policy_id}/reindex")

    def jobs(self) -> list[dict]:
        return self._request("GET", "/admin/ingestion-jobs")

    # chat
    def ask(self, question: str, policy_id: str | None, session_id: str | None) -> dict:
        return self._request("POST", "/chat/query", timeout=300,
                             json={"question": question, "policy_id": policy_id, "session_id": session_id})

    def sessions(self) -> list[dict]:
        return self._request("GET", "/chat/sessions")

    def messages(self, session_id: str) -> list[dict]:
        return self._request("GET", f"/chat/sessions/{session_id}/messages")

    def delete_session(self, session_id: str) -> dict:
        return self._request("DELETE", f"/chat/sessions/{session_id}")

    def feedback(self, message_id: str, rating: int) -> dict:
        return self._request("POST", f"/chat/messages/{message_id}/feedback", json={"rating": rating})

    # admin
    def audit_logs(self, **params) -> list[dict]:
        return self._request("GET", "/admin/audit-logs", params={k: v for k, v in params.items() if v})

    def login_sessions(self, **params) -> list[dict]:
        return self._request("GET", "/admin/login-sessions", params={k: v for k, v in params.items() if v})

    def stats(self) -> dict:
        return self._request("GET", "/admin/stats")

    def traces(self, limit: int = 50) -> list[dict]:
        return self._request("GET", "/admin/traces", params={"limit": limit})

    def trace(self, trace_id: str) -> dict:
        return self._request("GET", f"/admin/traces/{trace_id}")

    def users(self) -> list[dict]:
        return self._request("GET", "/admin/users")

    def create_user(self, payload: dict) -> dict:
        return self._request("POST", "/admin/users", json=payload)

    def run_evaluation(self, max_cases: int | None) -> dict:
        return self._request("POST", "/admin/evaluations", params={"max_cases": max_cases} if max_cases else None)

    def evaluations(self) -> list[dict]:
        return self._request("GET", "/admin/evaluations")
