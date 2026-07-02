"""
BRAIN/llm/provider_pool.py — Multi-provider LLM pool with automatic failover.

Wraps a prioritised list of LLMProvider clients, each guarded by its own
CircuitBreaker. On a non-recoverable error from the primary provider, the
pool tries the next healthy provider transparently.

Usage (configured via SOFI_LLM_PROVIDERS env var):
    pool = ProviderPool.from_env()          # e.g. "gemini,groq"
    async for token in pool.stream(...):    # same interface as a single client
        ...

If SOFI_LLM_PROVIDERS has only one entry, the pool is a zero-overhead pass-through.

The pool exposes the same interface as LLMProvider so Brain needs no
knowledge of whether it's talking to one client or many.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from BRAIN.llm.circuit_breaker import CircuitBreaker
from BRAIN.llm.error_classifier import classify_error, ErrorReason
from BRAIN.llm.groq_client import LLMResponse

_log = logging.getLogger("sofi.brain.provider_pool")

# Errors that mean "try the next provider" vs. "retry the same one"
_FAILOVER_REASONS = frozenset({
    ErrorReason.AUTH,
    ErrorReason.BILLING,
    ErrorReason.MODEL_NOT_FOUND,
    ErrorReason.SERVER_ERROR,
    ErrorReason.OVERLOADED,
    ErrorReason.TIMEOUT,
})


@dataclass
class _ProviderSlot:
    client: Any
    breaker: CircuitBreaker
    name: str


class ProviderPool:
    """
    Ordered pool of LLM providers with per-provider circuit breakers.

    Tries providers in order. If the primary is OPEN (tripped) or returns a
    failover-class error, the next provider is tried. Rate-limit errors are
    retried on the same provider (they're transient, not provider-down).
    """

    def __init__(self, slots: List[_ProviderSlot]) -> None:
        if not slots:
            raise ValueError("ProviderPool requires at least one provider")
        self._slots = slots
        # Expose primary provider's attributes at pool level for type checks
        self.model = slots[0].client.model
        self.temperature = slots[0].client.temperature
        self.max_tokens = slots[0].client.max_tokens

    @classmethod
    def from_clients(cls, clients: List[Any]) -> "ProviderPool":
        """Build a pool from a list of already-constructed LLMProvider clients."""
        slots = []
        for client in clients:
            name = type(client).__name__
            slots.append(_ProviderSlot(
                client=client,
                breaker=CircuitBreaker(name=name, threshold=5, cooldown_ms=5000),
                name=name,
            ))
        return cls(slots)

    @classmethod
    def from_env(cls, model_override: Optional[str] = None) -> "ProviderPool":
        """
        Build a pool from SOFI_LLM_PROVIDERS env var (comma-separated).
        E.g. SOFI_LLM_PROVIDERS=gemini,groq

        Falls back to SOFI_LLM_BACKEND (single-provider legacy env var)
        if SOFI_LLM_PROVIDERS is not set.
        """
        providers_str = os.environ.get("SOFI_LLM_PROVIDERS", "").strip()
        if not providers_str:
            # Fallback: single provider from legacy env var
            backend = os.environ.get("SOFI_LLM_BACKEND", "gemini").lower().strip()
            providers_str = backend

        names = [p.strip().lower() for p in providers_str.split(",") if p.strip()]
        clients = []
        for name in names:
            try:
                if name == "gemini":
                    from BRAIN.llm.gemini_client import GeminiClient
                    clients.append(GeminiClient(model=model_override))
                elif name == "groq":
                    from BRAIN.llm.groq_client import GroqClient
                    clients.append(GroqClient(model=model_override))
                else:
                    _log.warning("provider_pool | unknown provider name: %s — skipping", name)
            except Exception as exc:
                _log.warning("provider_pool | failed to init %s: %s", name, exc)

        if not clients:
            raise RuntimeError(
                f"ProviderPool.from_env: no providers could be initialized "
                f"(SOFI_LLM_PROVIDERS={providers_str!r})"
            )
        _log.info(
            "provider_pool | initialized | providers=%s",
            [type(c).__name__ for c in clients],
        )
        return cls.from_clients(clients)

    # ── LLMProvider interface ─────────────────────────────────────────────────

    async def stream(
        self,
        system_prompt: str,
        messages: List[Dict],
    ) -> AsyncIterator[str]:
        """Stream with automatic failover to next healthy provider.

        Failover only happens if an error occurs BEFORE any tokens are yielded.
        Mid-stream errors re-raise immediately (can't undo already-sent tokens).
        """
        last_exc: Optional[Exception] = None
        for slot in self._healthy_slots():
            tokens_yielded = 0
            try:
                async for token in slot.client.stream(system_prompt, messages):
                    tokens_yielded += 1
                    yield token
                slot.breaker.record_success()
                return
            except Exception as exc:
                classified = classify_error(exc)
                slot.breaker.record_failure()
                last_exc = exc
                if tokens_yielded > 0:
                    # Already sent tokens — failover would corrupt the response.
                    raise
                _log.warning(
                    "provider_pool | stream error (pre-token) | provider=%s reason=%s",
                    slot.name, classified.reason.value,
                )
                if classified.reason not in _FAILOVER_REASONS:
                    raise
                _log.info("provider_pool | failing over from %s", slot.name)
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All providers exhausted — no healthy provider available")

    async def call_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        max_tokens_override: Optional[int] = None,
    ) -> LLMResponse:
        """Call with tools, failing over to next healthy provider on error."""
        last_exc: Optional[Exception] = None

        for slot in self._healthy_slots():
            try:
                result = await slot.client.call_with_tools(
                    system_prompt, messages, tools, tool_choice, max_tokens_override,
                )
                slot.breaker.record_success()
                return result
            except Exception as exc:
                classified = classify_error(exc)
                last_exc = exc
                _log.warning(
                    "provider_pool | call_with_tools error | provider=%s reason=%s",
                    slot.name, classified.reason.value,
                )
                slot.breaker.record_failure()
                if classified.reason not in _FAILOVER_REASONS:
                    raise
                _log.info("provider_pool | failing over from %s", slot.name)
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All providers exhausted — no healthy provider available")

    async def stream_final(
        self,
        system_prompt: str,
        messages: List[Dict],
    ) -> AsyncIterator[str]:
        """stream_final with failover (same pattern as stream)."""
        last_exc: Optional[Exception] = None
        for slot in self._healthy_slots():
            tokens_yielded = 0
            try:
                async for token in slot.client.stream_final(system_prompt, messages):
                    tokens_yielded += 1
                    yield token
                slot.breaker.record_success()
                return
            except Exception as exc:
                classified = classify_error(exc)
                slot.breaker.record_failure()
                last_exc = exc
                if tokens_yielded > 0:
                    raise
                if classified.reason not in _FAILOVER_REASONS:
                    raise
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All providers exhausted — no healthy provider available")

    def _classify_error(self, exc: Exception) -> str:
        """Backward-compat shim used by SubAgentRunner."""
        return classify_error(exc).reason.value

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def health_report(self) -> List[Dict]:
        return [slot.breaker.status() for slot in self._slots]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _healthy_slots(self) -> List[_ProviderSlot]:
        """Return slots whose circuit breakers allow requests, primary first."""
        healthy = [s for s in self._slots if s.breaker.allow_request()]
        if not healthy:
            # All tripped — force a probe on the primary (better than silent failure)
            _log.warning(
                "provider_pool | all providers open — forcing probe on %s",
                self._slots[0].name,
            )
            return [self._slots[0]]
        return healthy
