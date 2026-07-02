"""
BRAIN/prompt/token_estimator.py — Lightweight token estimation

Uses a fast character-based heuristic (~4 chars per token for English).
No dependency on tiktoken or transformers — those add 100MB+ and cold-start
latency for marginal accuracy gain on this use case.

The estimate is conservative (rounds up) so we truncate rather than overflow.
"""

import logging

_log = logging.getLogger("sofi.brain.prompt")

CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length. Rounds up."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def check_budget(
    system_prompt: str,
    messages: list,
    max_tokens: int = 120_000,
    reserved_output: int = 8192,
) -> tuple:
    """
    Check whether the prompt + messages fit within the context window.

    Returns:
        (fits: bool, estimated_input_tokens: int, budget_remaining: int)
    """
    prompt_tokens = estimate_tokens(system_prompt)
    message_tokens = sum(
        estimate_tokens(m.get("content") or "")
        for m in messages
    )
    total_input = prompt_tokens + message_tokens
    budget = max_tokens - reserved_output
    fits = total_input <= budget

    if not fits:
        _log.warning(
            "token_budget | OVER | input=%d budget=%d (max=%d - reserved=%d) | "
            "prompt=%d messages=%d",
            total_input, budget, max_tokens, reserved_output,
            prompt_tokens, message_tokens,
        )

    return fits, total_input, max(0, budget - total_input)
