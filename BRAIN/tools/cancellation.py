"""
BRAIN/tools/cancellation.py — Opt-in cancellation for long-running tools

Tools that accept a `cancellation_token` kwarg get one injected by the
registry. They can check `token.is_cancelled` periodically to abort early.

Short tools don't need this — asyncio.Task.cancel() handles them.
This is for tools that do chunked work (pagination, multi-step processing)
where checking a flag between chunks is cleaner than catching CancelledError.

Usage in a tool handler:
    async def my_long_tool(query: str, cancellation_token=None):
        for page in pages:
            if cancellation_token and cancellation_token.is_cancelled:
                return "Cancelled — partial results: ..."
            await fetch(page)
"""

import threading


class CancellationToken:
    """Lightweight, thread-safe cancellation flag."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def __repr__(self) -> str:
        return f"CancellationToken(cancelled={self._cancelled})"
