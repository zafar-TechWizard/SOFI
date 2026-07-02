"""
BRAIN/llm/sanitizer.py — Message sanitization + response validation

Runs BEFORE each LLM call (sanitize) and AFTER response parse (validate).
Both are O(n) scans with zero allocations on clean data — only copy/fix
when something is actually wrong.

Zero latency on the happy path: clean messages pass through untouched,
valid responses return unchanged.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

_log = logging.getLogger("sofi.brain.sanitizer")


def sanitize_messages(messages: List[Dict]) -> List[Dict]:
    """
    Fix common message-list issues that cause provider API rejections.

    Fixes applied (only when needed — clean data passes through uncopied):
      1. content: None → "" for user/system messages (Gemini rejects None)
      2. tool_calls arguments as raw string → try json.loads → fallback {}
      3. tool messages missing tool_call_id → generate a placeholder UUID
      4. Empty tool_call_id on assistant tool_calls → generate UUID
      5. Drop messages with no role
    """
    dirty = False
    out = messages

    for idx, msg in enumerate(messages):
        role = msg.get("role")

        if not role:
            if not dirty:
                out = list(messages)
                dirty = True
            out[idx] = None
            _log.warning("sanitize | dropping message at idx=%d — no role", idx)
            continue

        # Fix 1: None content
        if role in ("user", "system") and msg.get("content") is None:
            if not dirty:
                out = list(messages)
                dirty = True
            out[idx] = {**msg, "content": ""}
            _log.debug("sanitize | idx=%d role=%s — content None → ''", idx, role)

        # Fix 2: tool_calls with string arguments
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            fixed_tcs = None
            for tc_idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                args = fn.get("arguments")

                needs_fix = False

                # Arguments is a raw non-JSON-string object (dict passed directly)
                if isinstance(args, dict):
                    args = json.dumps(args)
                    needs_fix = True

                # Arguments is a string but not valid JSON
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        _log.warning(
                            "sanitize | idx=%d tc=%d — malformed arguments, resetting to {}",
                            idx, tc_idx,
                        )
                        args = "{}"
                        needs_fix = True

                # Missing or empty tool call ID
                tc_id = tc.get("id")
                if not tc_id:
                    tc_id = str(uuid.uuid4())
                    needs_fix = True
                    _log.debug("sanitize | idx=%d tc=%d — missing id, generated %s", idx, tc_idx, tc_id[:8])

                if needs_fix:
                    if fixed_tcs is None:
                        fixed_tcs = list(tool_calls)
                    fixed_tcs[tc_idx] = {
                        **tc,
                        "id": tc_id,
                        "function": {**fn, "arguments": args},
                    }

            if fixed_tcs is not None:
                if not dirty:
                    out = list(messages)
                    dirty = True
                out[idx] = {**msg, "tool_calls": fixed_tcs}

        # Fix 3: tool messages missing tool_call_id
        if role == "tool" and not msg.get("tool_call_id"):
            if not dirty:
                out = list(messages)
                dirty = True
            placeholder = str(uuid.uuid4())
            out[idx] = {**msg, "tool_call_id": placeholder}
            _log.warning("sanitize | idx=%d — tool message missing tool_call_id, generated %s", idx, placeholder[:8])

    if dirty:
        out = [m for m in out if m is not None]

    return out


def validate_response(response) -> None:
    """
    Post-parse validation of LLMResponse. Mutates in-place to fix issues.

    Fixes applied:
      1. Tool calls with empty/None name → dropped with warning
      2. Tool calls with non-dict arguments → attempt parse, fallback {}
      3. text that's not a string → str()
      4. Duplicate tool call IDs → regenerate
    """
    if not hasattr(response, "tool_calls"):
        return

    # Fix text type
    if response.text and not isinstance(response.text, str):
        _log.warning("validate | text is %s, converting to str", type(response.text).__name__)
        response.text = str(response.text)

    if not response.tool_calls:
        return

    valid = []
    seen_ids = set()

    for tc in response.tool_calls:
        # Drop empty names
        if not tc.name or not tc.name.strip():
            _log.warning("validate | dropping tool call with empty name | id=%s", tc.id)
            continue

        # Fix non-dict arguments
        if not isinstance(tc.arguments, dict):
            if isinstance(tc.arguments, str):
                try:
                    tc.arguments = json.loads(tc.arguments)
                except (json.JSONDecodeError, TypeError):
                    _log.warning("validate | tool=%s — string arguments not parseable, using {}", tc.name)
                    tc.arguments = {}
            else:
                _log.warning("validate | tool=%s — arguments type %s, using {}", tc.name, type(tc.arguments).__name__)
                tc.arguments = {}

        # Fix duplicate IDs
        if tc.id in seen_ids:
            old_id = tc.id
            tc.id = str(uuid.uuid4())
            _log.debug("validate | tool=%s — duplicate id %s → %s", tc.name, old_id[:8], tc.id[:8])
        seen_ids.add(tc.id)

        valid.append(tc)

    dropped = len(response.tool_calls) - len(valid)
    if dropped > 0:
        _log.warning("validate | dropped %d invalid tool call(s)", dropped)
        response.tool_calls = valid


def extract_retry_after(exc: Exception) -> Optional[float]:
    """
    Extract Retry-After delay from a rate-limit exception.

    Providers embed this differently:
      - Gemini: "Retry after X seconds" in error message
      - Groq: HTTP Retry-After header (accessible via exc.response.headers)
      - Generic: look for numeric seconds in the error string

    Returns seconds to wait, or None if not found.
    """
    # Try Groq-style: exception has a .response with headers
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers:
            retry_val = headers.get("retry-after") or headers.get("Retry-After")
            if retry_val:
                try:
                    return float(retry_val)
                except (ValueError, TypeError):
                    pass

    # Try Gemini-style: "Retry after X seconds" in the message
    msg = str(exc)
    import re
    match = re.search(r"[Rr]etry\s+after\s+(\d+(?:\.\d+)?)\s*s", msg)
    if match:
        return float(match.group(1))

    # Try generic: "retry in X seconds" or "wait X seconds"
    match = re.search(r"(?:retry|wait)\s+(?:in\s+)?(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None
