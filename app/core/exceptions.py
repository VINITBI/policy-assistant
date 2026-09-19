"""Application error hierarchy. Every error carries a stable code and a user-friendly message.

Internal details are logged, never returned to the client.
"""
from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    user_message: str = "Something went wrong on our side. Please try again in a moment."

    def __init__(self, user_message: str | None = None, *, detail: str | None = None, code: str | None = None,
                 status_code: int | None = None) -> None:
        self.user_message = user_message or self.user_message
        self.detail = detail  # internal only (logged)
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        super().__init__(detail or self.user_message)


class AuthenticationError(AppError):
    status_code = 401
    code = "AUTH_FAILED"
    user_message = "Invalid username or password."


class SessionExpiredError(AppError):
    status_code = 401
    code = "SESSION_EXPIRED"
    user_message = "Your session has expired. Please log in again."


class AccountLockedError(AppError):
    status_code = 423
    code = "ACCOUNT_LOCKED"
    user_message = "Your account is temporarily locked after several failed attempts. Please try again later."


class PermissionDeniedError(AppError):
    status_code = 403
    code = "FORBIDDEN"
    user_message = "You do not have permission to perform this action."


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"
    user_message = "The requested item was not found."


class ValidationError(AppError):
    status_code = 422
    code = "INVALID_INPUT"
    user_message = "The request is not valid."


class DuplicateError(AppError):
    status_code = 409
    code = "DUPLICATE"
    user_message = "This item already exists."


class GuardrailViolation(AppError):
    status_code = 400
    code = "GUARDRAIL_BLOCKED"
    user_message = "Your message could not be processed."

    def __init__(self, user_message: str, *, rule: str, detail: str | None = None) -> None:
        self.rule = rule
        super().__init__(user_message, detail=detail, code=f"GUARDRAIL_{rule.upper()}")


class RateLimitError(AppError):
    status_code = 429
    code = "RATE_LIMITED"
    user_message = "You are sending messages too quickly. Please wait a few seconds and try again."


class IngestionError(AppError):
    status_code = 422
    code = "INGESTION_FAILED"
    user_message = "The policy document could not be processed."


class VectorStoreUnavailable(AppError):
    status_code = 503
    code = "VECTOR_DB_UNAVAILABLE"
    user_message = "The policy search service is temporarily unavailable. Please try again shortly."


class LLMUnavailable(AppError):
    status_code = 503
    code = "LLM_UNAVAILABLE"
    user_message = "The answer service is temporarily unavailable. Please try again shortly."
