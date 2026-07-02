"""
BRAIN/llm/circuit_breaker.py — Per-provider circuit breaker

States:
  CLOSED     → normal; requests flow through
  OPEN       → tripped; requests blocked for cooldown period, then probe
  HALF_OPEN  → one probe allowed; success → CLOSED, failure → OPEN

Usage:
    cb = CircuitBreaker(threshold=5, cooldown_ms=5000)
    if cb.allow_request():
        try:
            result = await provider.call(...)
            cb.record_success()
        except Exception as exc:
            cb.record_failure()
            raise
"""

import logging
import time
import threading
from enum import Enum

_log = logging.getLogger("sofi.brain.circuit_breaker")


class CBState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Thread-safe circuit breaker for one LLM provider.

    threshold:   consecutive failures before tripping OPEN
    cooldown_ms: how long to stay OPEN before probing (milliseconds)
    """

    def __init__(
        self,
        name: str = "",
        threshold: int = 5,
        cooldown_ms: float = 5000.0,
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.cooldown_ms = cooldown_ms

        self._lock = threading.Lock()
        self._state = CBState.CLOSED
        self._failures = 0
        self._opened_at: float = 0.0
        self._total_trips = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def allow_request(self) -> bool:
        """True if a request should be attempted through this provider."""
        with self._lock:
            if self._state == CBState.CLOSED:
                return True

            if self._state == CBState.OPEN:
                elapsed_ms = (time.monotonic() - self._opened_at) * 1000
                if elapsed_ms >= self.cooldown_ms:
                    _log.info(
                        "circuit_breaker | %s | cooldown elapsed (%.0fms) → HALF_OPEN",
                        self.name, elapsed_ms,
                    )
                    self._state = CBState.HALF_OPEN
                    return True
                return False

            # HALF_OPEN — allow exactly one probe
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state == CBState.HALF_OPEN:
                _log.info(
                    "circuit_breaker | %s | probe succeeded → CLOSED", self.name
                )
            self._state = CBState.CLOSED
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CBState.HALF_OPEN:
                _log.warning(
                    "circuit_breaker | %s | probe failed → OPEN again", self.name
                )
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                return

            if self._failures >= self.threshold:
                _log.warning(
                    "circuit_breaker | %s | %d consecutive failures → OPEN "
                    "(cooldown=%.0fms)",
                    self.name, self._failures, self.cooldown_ms,
                )
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                self._total_trips += 1

    def reset(self) -> None:
        with self._lock:
            self._state = CBState.CLOSED
            self._failures = 0

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def state(self) -> CBState:
        with self._lock:
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CBState.OPEN

    def status(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "threshold": self.threshold,
                "total_trips": self._total_trips,
                "cooldown_ms": self.cooldown_ms,
            }
