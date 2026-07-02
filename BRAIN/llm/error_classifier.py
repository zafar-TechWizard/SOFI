"""
BRAIN/llm/error_classifier.py — Structured error classification

Centralizes API error handling into an enum + dataclass so retry/recovery
decisions are consistent across both LLM clients and the agentic loop.

Pattern from Hermes Agent (agent/error_classifier.py).
"""

import enum
from dataclasses import dataclass


class ErrorReason(enum.Enum):
    AUTH = "auth"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    CONTENT_FILTER = "content_filter"
    SERVER_ERROR = "server_error"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    MODEL_NOT_FOUND = "model_not_found"
    FORMAT_ERROR = "format_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedError:
    reason: ErrorReason
    retryable: bool
    should_trim_context: bool = False
    original: str = ""

    @property
    def user_message(self) -> str:
        """SOFi-voice error message for the user."""
        _MESSAGES = {
            ErrorReason.AUTH: "API key issue — check the key in .env.",
            ErrorReason.BILLING: "Account billing issue with the API provider.",
            ErrorReason.RATE_LIMIT: "I've hit a rate limit. Give it a moment and try again, sir.",
            ErrorReason.CONTEXT_OVERFLOW: "The conversation has grown too long — I'll trim it down.",
            ErrorReason.PAYLOAD_TOO_LARGE: "That message was too large for the API to handle.",
            ErrorReason.CONTENT_FILTER: "My response was blocked by a content filter. Try rephrasing.",
            ErrorReason.SERVER_ERROR: "The API is having issues. I'll retry in a moment.",
            ErrorReason.OVERLOADED: "The API is overloaded right now. I'll try again shortly.",
            ErrorReason.TIMEOUT: "The request timed out. I'll retry.",
            ErrorReason.MODEL_NOT_FOUND: "The configured model wasn't found. Check the model name.",
            ErrorReason.FORMAT_ERROR: "The request format was rejected by the API.",
        }
        return _MESSAGES.get(self.reason, f"Something went wrong: {self.original}")


# Reason → retry/trim defaults
_RETRY_MAP = {
    ErrorReason.AUTH: False,
    ErrorReason.BILLING: False,
    ErrorReason.RATE_LIMIT: True,
    ErrorReason.CONTEXT_OVERFLOW: True,
    ErrorReason.PAYLOAD_TOO_LARGE: False,
    ErrorReason.CONTENT_FILTER: False,
    ErrorReason.SERVER_ERROR: True,
    ErrorReason.OVERLOADED: True,
    ErrorReason.TIMEOUT: True,
    ErrorReason.MODEL_NOT_FOUND: False,
    ErrorReason.FORMAT_ERROR: False,
    ErrorReason.UNKNOWN: False,
}

_TRIM_MAP = {
    ErrorReason.CONTEXT_OVERFLOW: True,
    ErrorReason.PAYLOAD_TOO_LARGE: True,
}


def classify_error(exc: Exception) -> ClassifiedError:
    """
    Classify an LLM API exception into a structured decision object.

    Tries SDK-level type checks first (faster, more reliable), then
    falls back to string matching for generic exceptions.
    """
    exc_type = type(exc).__name__
    msg = str(exc).lower()

    # SDK-level type checks (Groq and Google SDKs expose typed exceptions)
    if "AuthenticationError" in exc_type or "Unauthenticated" in exc_type:
        reason = ErrorReason.AUTH
    elif "RateLimitError" in exc_type or "ResourceExhausted" in exc_type:
        reason = ErrorReason.RATE_LIMIT
    elif "ContentFilter" in exc_type:
        reason = ErrorReason.CONTENT_FILTER
    elif "NotFound" in exc_type and "model" in msg:
        reason = ErrorReason.MODEL_NOT_FOUND
    elif "Timeout" in exc_type or "DeadlineExceeded" in exc_type:
        reason = ErrorReason.TIMEOUT
    # String-based fallback
    elif "401" in msg or "unauthorized" in msg:
        reason = ErrorReason.AUTH
    elif "402" in msg or "billing" in msg or "credit" in msg:
        reason = ErrorReason.BILLING
    elif "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg:
        reason = ErrorReason.RATE_LIMIT
    elif "context" in msg and ("long" in msg or "length" in msg or "overflow" in msg):
        reason = ErrorReason.CONTEXT_OVERFLOW
    elif "413" in msg or "payload" in msg or "too large" in msg:
        reason = ErrorReason.PAYLOAD_TOO_LARGE
    elif any(w in msg for w in ("safety", "blocked", "harm", "content_filter", "content" )):
        if any(w in msg for w in ("filter", "policy", "safety", "blocked", "harm")):
            reason = ErrorReason.CONTENT_FILTER
        else:
            reason = ErrorReason.UNKNOWN
    elif "503" in msg or "unavailable" in msg or "overloaded" in msg:
        reason = ErrorReason.OVERLOADED
    elif "500" in msg or "502" in msg or "internal" in msg or "server" in msg:
        reason = ErrorReason.SERVER_ERROR
    elif "timeout" in msg or "deadline" in msg or "timed out" in msg:
        reason = ErrorReason.TIMEOUT
    elif "404" in msg or "not found" in msg:
        reason = ErrorReason.MODEL_NOT_FOUND
    elif "400" in msg or "invalid" in msg or "bad request" in msg:
        reason = ErrorReason.FORMAT_ERROR
    else:
        reason = ErrorReason.UNKNOWN

    return ClassifiedError(
        reason=reason,
        retryable=_RETRY_MAP.get(reason, False),
        should_trim_context=_TRIM_MAP.get(reason, False),
        original=str(exc)[:200],
    )
