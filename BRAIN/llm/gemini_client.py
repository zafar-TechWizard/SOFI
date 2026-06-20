"""
BRAIN/llm/gemini_client.py — Async Google Gemini/Gemma Client

Drop-in replacement for GroqClient. Identical public interface:
    stream()          — pure text streaming
    call_with_tools() — non-streaming with tool support
    stream_final()    — streaming for the final turn (delegates to stream)
    _classify_error() — error classification for retry logic in brain.py

Available models (all on your account):
    gemini-2.0-flash         — default: fastest, generous free tier
    gemini-2.0-flash-lite    — even lighter (mind the quota)
    gemini-2.5-flash         — more capable, still fast
    gemma-4-31b-it           — Gemma 4 31B instruction-tuned
    gemini-3.1-flash-lite    — Gemini 3.1 Flash Lite
    gemini-3.1-flash-image   — if you need vision later

Usage (in brain.py or sofi.py):
    from BRAIN.llm.gemini_client import GeminiClient
    brain = Brain()
    brain._llm = GeminiClient()                        # default: gemini-2.0-flash
    brain._llm = GeminiClient(model="gemma-4-31b-it")  # Gemma 4
    brain._llm = GeminiClient(model="gemini-3.1-flash-lite")

API key: reads `gemini_api_key` from .env (lowercase, as you set it).
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from google import genai
from google.genai import types

from BRAIN.llm.groq_client import LLMResponse, LLMToolCall


_log = logging.getLogger("sofi.brain.gemini")

# ── Available model constants (for easy switching) ──────────────────────────
GEMINI_FLASH          = "gemini-2.0-flash"          # default — fast, free tier generous
GEMINI_FLASH_LITE     = "gemini-2.0-flash-lite"
GEMINI_25_FLASH       = "gemini-2.5-flash"
GEMINI_31_FLASH_LITE  = "gemini-3.1-flash-lite"
GEMMA_4_31B           = "gemma-4-31b-it"            # 1.5k/day on free tier

# Models that accept thinking_budget=0 to fully disable thinking.
_THINKING_BUDGET_MODELS: frozenset = frozenset({
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-preview-04-17",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash",
})

# Models that support thinking but reject thinking_budget — use include_thoughts=False
# to suppress thought_signature parts from the response without touching the budget.
_THINKING_HIDE_MODELS: frozenset = frozenset({
    "gemma-4-31b-it",
    "gemma-4-26b-it",
})


def _thinking_config_for(model: str) -> Optional[types.ThinkingConfig]:
    """
    Return the right ThinkingConfig to prevent thought_signature parts leaking
    into responses (which causes 400 on subsequent tool-call messages).

    - Gemini 2.5/3.1 thinking models: thinking_budget=0 disables thinking entirely.
    - Gemma 4 models: support thinking but reject thinking_budget; use
      include_thoughts=False to keep thinking internal but off the wire.
    - Gemini 2.0 / other models: no thinking config needed.
    """
    if model in _THINKING_BUDGET_MODELS:
        return types.ThinkingConfig(thinking_budget=0)
    if model in _THINKING_HIDE_MODELS:
        return types.ThinkingConfig(include_thoughts=False)
    return None


class GeminiClient:
    """
    Async Google Gemini/Gemma client with the same interface as GroqClient.

    brain.py calls exactly three methods: stream(), call_with_tools(),
    stream_final() — and _classify_error() for retry classification.
    All three are implemented here with identical signatures and return types.
    """

    DEFAULT_MODEL = GEMINI_31_FLASH_LITE
    MAX_RETRIES = 90

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = (
            api_key
            or os.environ.get("gemini_api_key")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise RuntimeError(
                "gemini_api_key is not set. Add it to .env at the workspace root."
            )

        self._client = genai.Client(api_key=resolved_key)
        _log.info("GeminiClient ready | model=%s", self.model)

    # ─── Pure text streaming ─────────────────────────────────────────────────

    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict],
    ) -> AsyncIterator[str]:
        """
        Stream tokens for a pure conversation turn (no tools).
        Mirrors GroqClient.stream() exactly.
        """
        contents = _messages_to_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=_thinking_config_for(self.model),
        )

        _log.debug("stream | model=%s contents=%d", self.model, len(contents))

        import asyncio
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                async for chunk in self._client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as exc:
                last_exc = exc
                classified = self._classify_error(exc)
                _log.warning("stream | attempt=%d error=%s cls=%s", attempt + 1, exc, classified)
                if classified in ("auth", "format_error", "content_filter"):
                    raise
                if classified == "rate_limit" and attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if classified in ("server_error", "timeout") and attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(1 + attempt)
                    continue
                raise
        if last_exc:
            raise last_exc

    # ─── Non-streaming with tool support ─────────────────────────────────────

    async def call_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """
        Call Gemini with tool definitions. Returns LLMResponse that may contain
        text, tool_calls, or both. Mirrors GroqClient.call_with_tools() exactly.
        """
        import asyncio
        contents = _messages_to_contents(messages)
        google_tools = _openai_tools_to_google(tools)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            tools=google_tools if google_tools else None,
            thinking_config=_thinking_config_for(self.model),
        )

        _log.debug(
            "call_with_tools | model=%s contents=%d tools=%d",
            self.model, len(contents), len(tools),
        )

        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:
                last_exc = exc
                classified = self._classify_error(exc)
                _log.warning(
                    "call_with_tools | attempt=%d error=%s cls=%s",
                    attempt + 1, exc, classified,
                )
                if classified in ("auth", "format_error", "content_filter"):
                    raise
                if classified == "rate_limit" and attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if classified in ("server_error", "timeout") and attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(1 + attempt)
                    continue
                raise
        else:
            if last_exc:
                raise last_exc

        return _parse_response(response)

    # ─── Stream final response (after tool loop) ──────────────────────────────

    async def stream_final(
        self,
        system_prompt: str,
        messages: List[Dict],
    ) -> AsyncIterator[str]:
        """
        Stream final response with no tools. Delegates to stream().
        Mirrors GroqClient.stream_final() exactly.
        """
        async for token in self.stream(system_prompt, messages):
            yield token

    # ─── Error classification (mirrors GroqClient._classify_error) ───────────

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        msg = str(exc).lower()

        if "401" in msg or "unauthenticated" in msg or ("invalid" in msg and "key" in msg):
            return "auth"
        if "429" in msg or "resource_exhausted" in msg or "rate" in msg or "quota" in msg:
            return "rate_limit"
        if "413" in msg or "payload" in msg or "too large" in msg:
            return "payload_too_large"
        if "context" in msg and ("long" in msg or "length" in msg or "overflow" in msg):
            return "context_overflow"
        if "safety" in msg or "blocked" in msg or "harm" in msg:
            return "content_filter"
        if "503" in msg or "unavailable" in msg or "overloaded" in msg:
            return "overloaded"
        if "500" in msg or "502" in msg or "internal" in msg:
            return "server_error"
        if "timeout" in msg or "deadline" in msg or "timed out" in msg:
            return "timeout"
        if "404" in msg or "not found" in msg:
            return "model_not_found"
        if "400" in msg or "invalid" in msg:
            return "format_error"
        return "unknown"


# =============================================================================
# Message format conversion: OpenAI → Google Content objects
# =============================================================================

def _messages_to_contents(messages: List[Dict]) -> List[types.Content]:
    """
    Convert brain.py's OpenAI-style message list to Google Content objects.

    OpenAI roles → Google roles:
        user       → user
        assistant  → model
        tool       → user (function_response parts, merged with consecutive tool messages)
        system     → user (wrapped in [System: ...])

    Tool call messages from the assistant become function_call parts.
    Tool result messages become function_response parts in a user Content.
    """
    contents: List[types.Content] = []
    i = 0

    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "user")

        if role == "user":
            text = msg.get("content") or ""
            if text:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=text)],
                ))
            i += 1

        elif role == "assistant":
            # If a raw Gemini Content object was preserved (set by _parse_response
            # and carried via brain.py's "_raw_content" key), use it directly.
            # This is critical for thinking models: the native Content keeps the
            # thought_signature bytes inside FunctionCall parts intact. Reconstructing
            # a new types.FunctionCall from our parsed data drops those bytes, causing
            # a 400 "missing thought_signature" error on the next API call.
            raw = msg.get("_raw_content")
            if raw is not None:
                contents.append(raw)
                i += 1
                continue

            parts = []

            # Text part (may be None when tool_calls accompanies it)
            text = msg.get("content")
            if text:
                parts.append(types.Part(text=text))

            # Function call parts — only reached for Groq-sourced messages
            # (Gemini messages always have raw_content and take the branch above)
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parts.append(types.Part(
                    function_call=types.FunctionCall(name=name, args=args)
                ))

            if parts:
                contents.append(types.Content(role="model", parts=parts))
            i += 1

        elif role == "tool":
            # Merge consecutive "tool" messages into a single user Content
            # with multiple function_response parts (Google's requirement).
            fn_response_parts = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tmsg = messages[i]
                call_id = tmsg.get("tool_call_id", "")
                tool_name = _find_tool_name(messages, i, call_id)
                result_content = tmsg.get("content", "")
                fn_response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response={"result": result_content},
                    )
                ))
                i += 1
            if fn_response_parts:
                contents.append(types.Content(role="user", parts=fn_response_parts))

        elif role == "system":
            # Mid-conversation system injection — treat as user note
            text = msg.get("content", "")
            if text:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"[Context update: {text}]")],
                ))
            i += 1

        else:
            i += 1

    # Google requires at least one content item
    if not contents:
        contents.append(types.Content(role="user", parts=[types.Part(text="Hello.")]))

    return contents


def _find_tool_name(messages: List[Dict], tool_idx: int, call_id: str) -> str:
    """
    Look backwards from tool_idx to find the function name for a given
    tool_call_id in the preceding assistant message.
    """
    for j in range(tool_idx - 1, -1, -1):
        msg = messages[j]
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                if tc.get("id") == call_id:
                    return tc.get("function", {}).get("name", "unknown_function")
    return "unknown_function"


# =============================================================================
# Tool schema conversion: OpenAI JSON Schema → Google FunctionDeclarations
# =============================================================================

def _openai_tools_to_google(openai_tools: List[Dict]) -> List[types.Tool]:
    """
    Convert brain.py's OpenAI-format tool list to a single Google Tool object.
    Returns empty list if no tools.
    """
    if not openai_tools:
        return []

    declarations = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        params_schema = fn.get("parameters", {})
        google_schema = _convert_schema(params_schema)

        declarations.append(types.FunctionDeclaration(
            name=fn.get("name", ""),
            description=fn.get("description", ""),
            parameters=google_schema,
        ))

    return [types.Tool(function_declarations=declarations)]


def _convert_schema(schema: Dict) -> types.Schema:
    """
    Recursively convert an OpenAI JSON Schema dict to a Google Schema object.

    Type mapping (OpenAI → Google):
        string  → STRING
        integer → INTEGER
        number  → NUMBER
        boolean → BOOLEAN
        array   → ARRAY
        object  → OBJECT  (default)
    """
    if not schema:
        return types.Schema(type="OBJECT")

    TYPE_MAP = {
        "string":  "STRING",
        "integer": "INTEGER",
        "number":  "NUMBER",
        "boolean": "BOOLEAN",
        "array":   "ARRAY",
        "object":  "OBJECT",
    }
    raw_type = schema.get("type", "object")
    google_type = TYPE_MAP.get(str(raw_type).lower(), "STRING")

    kwargs: Dict[str, Any] = {"type": google_type}

    desc = schema.get("description", "")
    if desc:
        kwargs["description"] = desc

    # Properties (for OBJECT types)
    raw_props = schema.get("properties") or {}
    if raw_props:
        kwargs["properties"] = {
            name: _convert_schema(prop)
            for name, prop in raw_props.items()
        }

    # Required fields
    required = schema.get("required")
    if required:
        kwargs["required"] = list(required)

    # Enum values
    enum_vals = schema.get("enum")
    if enum_vals:
        kwargs["enum"] = [str(v) for v in enum_vals]

    # Array items schema
    if google_type == "ARRAY" and schema.get("items"):
        kwargs["items"] = _convert_schema(schema["items"])

    return types.Schema(**kwargs)


# =============================================================================
# Response parsing: Google GenerateContentResponse → LLMResponse
# =============================================================================

def _parse_response(response) -> LLMResponse:
    """
    Parse a Google GenerateContentResponse into an LLMResponse.

    Maps Google finish reasons to the strings brain.py checks:
        STOP / FUNCTION_CALL → "stop" (or detected from tool_calls)
        MAX_TOKENS           → "length"
        SAFETY               → "content_filter"
        others               → "stop"

    Tool calls get a generated UUID for id (Google doesn't provide one).
    """
    result = LLMResponse()

    # response.text is a SDK-level shortcut (concatenated text of all text parts).
    # We try it first so we have a fallback even if candidate.content is None.
    # Suppress the SDK RuntimeWarning about non-text parts (thought_signature,
    # function_call) — we handle those manually from content.parts below.
    shortcut_text: Optional[str] = None
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            shortcut_text = response.text  # None when no text; raises on multiple candidates
    except Exception:
        pass

    if not response.candidates:
        _log.warning("_parse_response | no candidates | shortcut_text=%s", bool(shortcut_text))
        if shortcut_text:
            result.text = shortcut_text
            result.finish_reason = "stop"
        return result

    candidate = response.candidates[0]

    # ── finish_reason ──
    fr = candidate.finish_reason
    if hasattr(fr, "name"):
        fr_name = fr.name.upper()
    else:
        fr_name = str(fr).upper().split(".")[-1]

    if fr_name in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS"):
        result.finish_reason = "length"
    elif fr_name in ("SAFETY", "PROHIBITED_CONTENT", "SPII", "RECITATION"):
        result.finish_reason = "content_filter"
    else:
        result.finish_reason = "stop"

    _log.debug("_parse_response | raw_finish=%s mapped=%s", fr_name, result.finish_reason)

    # ── Content can be None for blocked/safety-filtered responses ──
    content = candidate.content
    # Preserve the raw Content object so callers can pass it back to the API
    # unchanged — Gemini thinking models embed a thought_signature bytes field
    # inside FunctionCall parts that gets silently dropped when we reconstruct
    # the message from our parsed LLMToolCall objects. Passing raw_content back
    # directly is the only reliable way to keep thought_signatures intact.
    result.raw_content = content
    if content is None:
        _log.warning(
            "_parse_response | candidate.content is None | finish=%s | fallback_text=%s",
            fr_name, bool(shortcut_text),
        )
        if shortcut_text:
            result.text = shortcut_text
        return result

    parts = content.parts or []
    _log.debug("_parse_response | parts_count=%d", len(parts))

    for part in parts:
        # text — unset text fields return "" in proto3; guard anyway
        try:
            text = part.text
        except Exception:
            text = None
        if text:
            result.text += text

        # function_call — present only when model chose to call a tool
        try:
            fc = part.function_call
        except Exception:
            fc = None
        if fc is not None and getattr(fc, "name", None):
            args = dict(fc.args) if fc.args else {}
            result.tool_calls.append(LLMToolCall(
                id=str(uuid.uuid4()),   # Google doesn't provide tool-call IDs
                name=fc.name,
                arguments=args,
            ))
            _log.debug(
                "_parse_response | tool_call name=%s args_keys=%s",
                fc.name, list(args.keys()),
            )

    # Last-resort fallback: if parts gave us nothing, use the shortcut
    if not result.text and not result.tool_calls and shortcut_text:
        _log.debug("_parse_response | parts empty — using response.text shortcut")
        result.text = shortcut_text

    _log.debug(
        "_parse_response | done | text_len=%d tool_calls=%d finish=%s",
        len(result.text), len(result.tool_calls), result.finish_reason,
    )
    return result
