"""
BRAIN/tools/turn_budget.py — Per-turn aggregate tool output budget.

After all tool results are collected in one iteration, this module
checks whether their combined size exceeds the turn budget. If so,
the largest results are spilled to disk with a preview + path,
letting the model read_file if it needs the full content.

Layer 2 of the 3-layer tool result defense:
  Layer 1: per-tool truncation (output_limits.py)
  Layer 2: per-turn aggregate budget (this file)
  Layer 3: result persistence via spill-to-disk
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

_log = logging.getLogger("sofi.brain.tools.budget")

MAX_TURN_TOOL_CHARS = 50_000
SPILL_PREVIEW_CHARS = 1_500


def enforce_turn_budget(
    messages: List[Dict],
    iteration_start_idx: int,
    spill_dir: str | Path,
    budget: int = MAX_TURN_TOOL_CHARS,
) -> List[Dict]:
    """
    Enforce aggregate char budget on tool results from the current iteration.

    Args:
        messages: full messages list (mutated in place for spilled results)
        iteration_start_idx: index of the first message added this iteration
        spill_dir: directory for spilled results (e.g. .temp/)
        budget: max total chars across all tool results this iteration
    """
    tool_indices = []
    total_chars = 0

    for i in range(iteration_start_idx, len(messages)):
        msg = messages[i]
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            tool_indices.append((i, len(content)))
            total_chars += len(content)

    if total_chars <= budget:
        return messages

    _log.info(
        "turn_budget | over budget | total=%d budget=%d tools=%d",
        total_chars, budget, len(tool_indices),
    )

    spill_path = Path(spill_dir)
    spill_path.mkdir(parents=True, exist_ok=True)

    candidates = sorted(tool_indices, key=lambda x: x[1], reverse=True)

    for idx, content_len in candidates:
        if total_chars <= budget:
            break

        msg = messages[idx]
        content = msg.get("content", "")
        if content_len <= SPILL_PREVIEW_CHARS:
            continue

        tool_call_id = msg.get("tool_call_id", "unknown")
        filename = f"tool_{tool_call_id}_{int(time.time())}.txt"
        filepath = spill_path / filename

        try:
            filepath.write_text(content, encoding="utf-8")
        except OSError as exc:
            _log.warning("turn_budget | spill write failed | err=%s", exc)
            continue

        preview = content[:SPILL_PREVIEW_CHARS]
        replacement = (
            f"{preview}\n\n"
            f"[... {content_len:,} chars total — full output saved to {filepath}. "
            f"Use read_file to access if needed.]"
        )

        messages[idx] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": replacement,
        }
        saved = content_len - len(replacement)
        total_chars -= saved

        _log.info(
            "turn_budget | spilled | tool_call_id=%s saved=%d path=%s",
            tool_call_id, saved, filepath,
        )

    return messages
