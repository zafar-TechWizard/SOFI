"""
BRAIN/context_compression.py — Deterministic message compression.

Hermes-style pruning: old tool results are never dropped entirely —
they're replaced with structured 1-line summaries that preserve what
was done and what was found. No context is lost, only verbosity.

Used by:
  - brain.py Phase D: per-iteration budget re-check
  - brain.py error recovery: context_overflow → compress and retry
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger("sofi.brain.compression")

MAX_TOTAL_CHARS = 280_000
RESERVED_OUTPUT_CHARS = 28_672
PROTECT_RECENT_EXCHANGES = 2
SUMMARY_MAX_CHARS = 120


def _estimate_messages_chars(messages: List[Dict]) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(fn.get("arguments", ""))
    return total


def compress_loop_messages(
    messages: List[Dict],
    system_prompt_chars: int = 0,
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> List[Dict]:
    """
    Compress messages to fit within the context window budget.

    Uses Hermes-style deterministic pruning — old tool results become
    structured 1-line summaries. No information is dropped silently.

    Args:
        messages: the conversation messages list
        system_prompt_chars: length of the system prompt (includes WorkingContext)
        max_total_chars: total context window budget in chars
    """
    messages_budget = max_total_chars - system_prompt_chars - RESERVED_OUTPUT_CHARS
    if messages_budget < 10_000:
        messages_budget = 10_000

    current = _estimate_messages_chars(messages)
    if current <= messages_budget:
        return messages

    _log.info(
        "compress | start | current=%d budget=%d sys_prompt=%d",
        current, messages_budget, system_prompt_chars,
    )

    # Pass 1: Prune old tool results to 1-line summaries
    messages = _prune_old_tool_results(messages, PROTECT_RECENT_EXCHANGES)
    current = _estimate_messages_chars(messages)
    if current <= messages_budget:
        _log.info("compress | pass1 (prune tools) sufficient | now=%d", current)
        return messages

    # Pass 2: Deduplicate identical tool calls
    messages = _deduplicate_tool_calls(messages)
    current = _estimate_messages_chars(messages)
    if current <= messages_budget:
        _log.info("compress | pass2 (dedup) sufficient | now=%d", current)
        return messages

    # Pass 3: Truncate old assistant reasoning text
    messages = _truncate_old_assistant_text(messages, PROTECT_RECENT_EXCHANGES)
    current = _estimate_messages_chars(messages)
    if current <= messages_budget:
        _log.info("compress | pass3 (truncate reasoning) sufficient | now=%d", current)
        return messages

    # Pass 4: Drop oldest exchanges (never drop first user msg or last 3)
    messages = _drop_oldest_exchanges(messages, messages_budget)
    current = _estimate_messages_chars(messages)
    _log.info("compress | pass4 (drop oldest) | now=%d budget=%d", current, messages_budget)

    return messages


def _prune_old_tool_results(
    messages: List[Dict],
    protect_recent: int,
) -> List[Dict]:
    """
    Replace old tool-role messages with structured 1-line summaries.

    Protects the N most recent assistant+tool exchanges from pruning.
    """
    tool_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "tool"
    ]
    if len(tool_indices) <= protect_recent:
        return messages

    indices_to_prune = tool_indices[:-protect_recent] if protect_recent > 0 else tool_indices

    for idx in indices_to_prune:
        msg = messages[idx]
        content = msg.get("content", "")
        if len(content) <= SUMMARY_MAX_CHARS:
            continue

        tool_call_id = msg.get("tool_call_id", "")
        tool_name, tool_args = _find_tool_info(messages, idx, tool_call_id)
        summary = _build_tool_summary(tool_name, tool_args, content)
        messages[idx] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": summary,
        }

    return messages


def _deduplicate_tool_calls(messages: List[Dict]) -> List[Dict]:
    """
    If the same tool was called with identical args multiple times,
    keep only the most recent result.
    """
    seen: Dict[str, int] = {}
    tool_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "tool"
    ]

    for idx in tool_indices:
        msg = messages[idx]
        tool_call_id = msg.get("tool_call_id", "")
        tool_name, tool_args = _find_tool_info(messages, idx, tool_call_id)
        key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

        if key in seen:
            old_idx = seen[key]
            old_content = messages[old_idx].get("content", "")
            if len(old_content) > SUMMARY_MAX_CHARS:
                messages[old_idx] = {
                    "role": "tool",
                    "tool_call_id": messages[old_idx].get("tool_call_id", ""),
                    "content": f"[{tool_name}] (duplicate of later call, removed)",
                }
        seen[key] = idx

    return messages


def _truncate_old_assistant_text(
    messages: List[Dict],
    protect_recent: int,
) -> List[Dict]:
    """
    Truncate text in old assistant messages that also have tool_calls.
    The tool_calls structure is preserved (maps to tool results).
    """
    assistant_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]

    if len(assistant_indices) <= protect_recent:
        return messages

    indices_to_truncate = (
        assistant_indices[:-protect_recent] if protect_recent > 0
        else assistant_indices
    )

    for idx in indices_to_truncate:
        msg = messages[idx]
        text = msg.get("content") or ""
        if text and len(text) > 100:
            messages[idx] = {**msg, "content": text[:100] + "..."}

    return messages


def _drop_oldest_exchanges(
    messages: List[Dict],
    budget: int,
) -> List[Dict]:
    """
    Drop oldest assistant+tool pairs until under budget.
    Never drops: first user message, system messages, last 3 messages.
    """
    if len(messages) <= 4:
        return messages

    first_msg = messages[0]
    protected_tail = 3
    middle = messages[1:-protected_tail] if len(messages) > protected_tail + 1 else []
    tail = messages[-protected_tail:]

    while _estimate_messages_chars([first_msg] + middle + tail) > budget and len(middle) > 0:
        dropped = middle.pop(0)
        _log.debug(
            "compress | dropping msg role=%s len=%d",
            dropped.get("role", "?"),
            len(dropped.get("content", "") or ""),
        )

    return [first_msg] + middle + tail


def _find_tool_info(
    messages: List[Dict],
    tool_msg_idx: int,
    tool_call_id: str,
) -> Tuple[str, dict]:
    """
    Look backward from a tool-role message to find the tool name and
    args from the preceding assistant message's tool_calls.
    """
    for j in range(tool_msg_idx - 1, -1, -1):
        msg = messages[j]
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                if tc.get("id") == tool_call_id:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown_tool")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    return name, args
            break
    return "unknown_tool", {}


def _build_tool_summary(tool_name: str, args: dict, result_content: str) -> str:
    """
    Build a structured 1-line summary of a tool result.

    Format: [{tool_name}] {args_summary} → {outcome}
    """
    args_summary = _summarize_args(tool_name, args)
    outcome = _extract_outcome(result_content)
    summary = f"[{tool_name}] {args_summary} → {outcome}"
    return summary[:SUMMARY_MAX_CHARS]


def _summarize_args(tool_name: str, args: dict) -> str:
    """Extract the most informative argument for the summary."""
    if not args:
        return ""

    priority_keys = {
        "read_file": "path",
        "write_file": "path",
        "patch_file": "path",
        "search_files": "pattern",
        "list_directory": "path",
        "web_search": "query",
        "web_fetch": "url",
        "run_command": "command",
        "run_python": "code",
        "spawn_agent": "agent_type",
    }

    key = priority_keys.get(tool_name)
    if key and key in args:
        val = str(args[key])
        if len(val) > 60:
            val = val[:57] + "..."
        return f"{key}={val}"

    first_key = next(iter(args), None)
    if first_key:
        val = str(args[first_key])
        if len(val) > 60:
            val = val[:57] + "..."
        return f"{first_key}={val}"

    return ""


def _extract_outcome(content: str) -> str:
    """
    Extract a brief outcome from the tool result content.
    First meaningful line, capped at 60 chars.
    """
    if not content:
        return "(empty)"

    if content.startswith("Error:"):
        return content[:60]

    lines = content.strip().splitlines()
    for line in lines:
        clean = line.strip()
        if clean and len(clean) > 5:
            if len(clean) > 60:
                return clean[:57] + "..."
            return clean

    return f"({len(content)} chars)"
