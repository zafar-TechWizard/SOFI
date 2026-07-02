"""
BRAIN/agents/budget.py — Per-agent iteration and token budgets.

Each sub-agent gets its own budget, independent of the parent.
Thread-safe — multiple sub-agents can run concurrently.

IterationBudget: hard cap on LLM call count.
TokenBudget: approximate tracking of input/output tokens.
"""

import threading
from dataclasses import dataclass, field


class IterationBudget:
    """
    Thread-safe iteration counter for one sub-agent.

    Each sub-agent gets its own budget. Parent budget is NOT affected.
    Budget is consumed on each LLM call. Refundable for special cases
    (e.g., pure-progress-update turns that don't cost real reasoning).
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration. Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_iterations:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_iterations - self._used)

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._used >= self.max_iterations

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "max": self.max_iterations,
                "used": self._used,
                "remaining": max(0, self.max_iterations - self._used),
            }


class TokenBudget:
    """
    Approximate token usage tracker for one sub-agent.

    Uses character-based estimation (chars / 4) since Groq/Gemini
    don't always return exact token counts in non-streaming mode.
    Thread-safe for concurrent tool execution within an agent.
    """

    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        max_input_tokens: int = 100_000,
        max_output_tokens: int = 16_000,
    ) -> None:
        self.max_input = max_input_tokens
        self.max_output = max_output_tokens
        self._input_used = 0
        self._output_used = 0
        self._lock = threading.Lock()

    def record(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            self._input_used += input_tokens
            self._output_used += output_tokens

    def record_chars(self, input_chars: int = 0, output_chars: int = 0) -> None:
        self.record(
            input_tokens=input_chars // self.CHARS_PER_TOKEN,
            output_tokens=output_chars // self.CHARS_PER_TOKEN,
        )

    @property
    def input_remaining(self) -> int:
        with self._lock:
            return max(0, self.max_input - self._input_used)

    @property
    def output_remaining(self) -> int:
        with self._lock:
            return max(0, self.max_output - self._output_used)

    @property
    def input_exhausted(self) -> bool:
        with self._lock:
            return self._input_used >= self.max_input

    @property
    def output_exhausted(self) -> bool:
        with self._lock:
            return self._output_used >= self.max_output

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "input": {"used": self._input_used, "max": self.max_input},
                "output": {"used": self._output_used, "max": self.max_output},
            }
