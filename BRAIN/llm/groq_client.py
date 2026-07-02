"""
BRAIN/llm/groq_client.py — Async Streaming Groq Client

Wraps the official `groq` SDK with the surface BRAIN needs:
  - stream()          — pure text streaming (conversation-only turns)
  - call_with_tools() — non-streaming call that supports tool calling
  - stream_final()    — stream the final response after tool loop completes

Usage:
    from BRAIN.llm import GroqClient

    client = GroqClient()
    async for token in client.stream(system_prompt, messages):
        print(token, end="", flush=True)
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import logging

from groq import AsyncGroq

from BRAIN.llm.retry_utils import jittered_backoff
from BRAIN.llm.sanitizer import extract_retry_after

_log = logging.getLogger("sofi.brain.groq")


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    finish_reason: str = ""
    # Preserved native content object from Gemini responses.
    # Required for thought_signature round-trip when using Gemini thinking models with tools.
    # Groq responses leave this None.
    raw_content: Any = None


class GroqClient:
    """Async wrapper around the Groq chat completions API with tool support."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env at the workspace "
                "root or export it before running."
            )
        self._client = AsyncGroq(api_key=resolved_key)

    # ─── Pure text streaming (unchanged — used for conversation-only turns) ───

    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> AsyncIterator[str]:
        """
        Stream tokens for a pure conversation turn (no tools).
        """
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        import time as _time
        response_stream = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        _t0 = _time.perf_counter()
        _chunks = 0
        _bytes = 0
        async for chunk in response_stream:
            _chunks += 1
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                _bytes += len(delta)
                yield delta
        _elapsed = (_time.perf_counter() - _t0) * 1000
        _log.debug(
            "stream | done | chunks=%d bytes=%d elapsed_ms=%.0f",
            _chunks, _bytes, _elapsed,
        )

    # ─── Non-streaming call with tool support (for the agentic loop) ─────────

    async def call_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        max_tokens_override: Optional[int] = None,
    ) -> LLMResponse:
        """
        Call Groq with tool definitions. Returns structured response
        that may contain text, tool calls, or both.

        Non-streaming — the agentic loop needs the complete response
        to decide what to do next.

        Handles Hermes-style error classification:
          - rate_limit (429) → raise with retry hint
          - context_overflow (400 + "context") → raise with compress hint
          - content_filter → return failed response
          - transient (500/502/503) → retry up to 3 times
        """
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens_override or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                last_error = exc
                classified = self._classify_error(exc)
                if classified in ("content_policy", "auth", "billing", "format_error"):
                    raise
                if classified == "rate_limit" and attempt < self.MAX_RETRIES - 1:
                    retry_after = extract_retry_after(exc)
                    delay = retry_after if retry_after else jittered_backoff(attempt, base_delay=3.0)
                    _log.debug("call_with_tools | rate_limit — waiting %.1fs (provider=%s)", delay, retry_after is not None)
                    await asyncio.sleep(delay)
                    continue
                if classified in ("server_error", "overloaded", "timeout") and attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(jittered_backoff(attempt, base_delay=1.0))
                    continue
                raise
        else:
            raise last_error  # type: ignore[misc]

        if not response.choices:
            return LLMResponse(finish_reason="error")

        choice = response.choices[0]
        result = LLMResponse(finish_reason=choice.finish_reason or "")

        if choice.message.content:
            result.text = choice.message.content

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result.tool_calls.append(LLMToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        if not result.text and not result.tool_calls:
            result.finish_reason = "error"

        return result

    # ─── Stream final response (after tool loop, no tools) ───────────────────

    async def stream_final(
        self,
        system_prompt: str,
        messages: List[Dict],
    ) -> AsyncIterator[str]:
        """
        Stream the final text response to the user after the agentic
        loop finishes. No tools passed — just text generation.
        """
        import time as _time
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        response_stream = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        _t0 = _time.perf_counter()
        _chunks = 0
        _bytes = 0
        async for chunk in response_stream:
            _chunks += 1
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                _bytes += len(delta)
                yield delta
        _elapsed = (_time.perf_counter() - _t0) * 1000
        _log.debug(
            "stream_final | done | chunks=%d bytes=%d elapsed_ms=%.0f",
            _chunks, _bytes, _elapsed,
        )

    # ─── Error classification (Hermes pattern) ───────────────────────────────

    MAX_RETRIES = 3

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        msg = str(exc).lower()
        exc_type = type(exc).__name__.lower()

        if "401" in msg or "unauthorized" in msg or "auth" in exc_type:
            return "auth"
        if "402" in msg or "billing" in msg or "credit" in msg:
            return "billing"
        if "429" in msg or "rate" in msg:
            return "rate_limit"
        if "413" in msg or "payload" in msg:
            return "payload_too_large"
        if "context" in msg and ("long" in msg or "length" in msg or "overflow" in msg):
            return "context_overflow"
        if "content" in msg and ("filter" in msg or "policy" in msg or "safety" in msg):
            return "content_policy"
        if "503" in msg or "529" in msg or "overloaded" in msg:
            return "overloaded"
        if "500" in msg or "502" in msg or "server" in msg:
            return "server_error"
        if "timeout" in msg or "timed out" in msg:
            return "timeout"
        if "404" in msg or "not found" in msg:
            return "model_not_found"
        if "400" in msg or "bad request" in msg:
            return "format_error"
        return "unknown"
