"""
BRAIN/agents/safety.py — Sub-agent tool access control and file safety.

Enforces:
  - Tool intersection: sub-agent can never gain tools the parent lacks
  - Blocked tools: some tools are never available to sub-agents
  - Dangerous tools: auto-denied in sub-agents (no interactive approval)
  - File path safety: certain paths are never writable/readable
  - File conflict detection: warns parent when child writes files parent read
"""

import logging
import os
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

_log = logging.getLogger("sofi.brain.agents.safety")

# ── Tools sub-agents can NEVER call ─────────────────────────────────────
BLOCKED_TOOLS: FrozenSet[str] = frozenset({
    "spawn_agent",
    "spawn_agent_background",
})

# ── Tools that require approval (auto-denied in sub-agents) ────────────
# Sub-agents run without user interaction. Dangerous tools that normally
# prompt for confirmation are auto-denied to prevent silent damage.
DANGEROUS_TOOLS: FrozenSet[str] = frozenset({
    # Currently empty — run_command and run_python have their own
    # safety checks via check_command_safety/check_python_safety.
    # Add tools here if they should be blanket-denied in sub-agents.
})

# ── File paths sub-agents can NEVER write to ───────────────────────────
BLOCKED_WRITE_PATHS: List[str] = [
    ".env",
    ".git/",
    ".gitignore",
    "CLAUDE.md",
    "__pycache__/",
    "node_modules/",
    ".venv/",
]

# ── File paths sub-agents can NEVER read ───────────────────────────────
BLOCKED_READ_PATHS: List[str] = [
    ".env",
    ".git/config",
    ".git/credentials",
]


def filter_tools(
    agent_tools: List[str],
    parent_available: Optional[List[str]] = None,
) -> List[str]:
    """
    Filter tool list for a sub-agent.

    1. Remove BLOCKED_TOOLS
    2. If parent_available is provided, intersect (child can't gain tools parent lacks)
    3. Return filtered list

    Args:
        agent_tools: tools declared in agent definition
        parent_available: tools available in the parent's registry (None = no filtering)
    """
    filtered = [t for t in agent_tools if t not in BLOCKED_TOOLS]

    if parent_available is not None:
        parent_set = set(parent_available)
        filtered = [t for t in filtered if t in parent_set]

    return filtered


def check_tool_safety(tool_name: str, args: dict) -> Tuple[str, str]:
    """
    Pre-execution safety check for a sub-agent tool call.

    Returns (tier, reason) where tier is:
      - "safe": proceed
      - "blocked": never allow
      - "dangerous": auto-deny in sub-agents

    File safety checks are applied for file-operating tools.
    """
    if tool_name in BLOCKED_TOOLS:
        return "blocked", f"Tool '{tool_name}' is not available to sub-agents"

    if tool_name in DANGEROUS_TOOLS:
        return "dangerous", f"Tool '{tool_name}' requires approval (auto-denied in sub-agents)"

    if tool_name == "write_file":
        path = args.get("path", "") or args.get("file_path", "")
        ok, reason = check_file_write_safety(path)
        if not ok:
            return "blocked", reason

    if tool_name == "patch_file":
        path = args.get("path", "") or args.get("file_path", "")
        ok, reason = check_file_write_safety(path)
        if not ok:
            return "blocked", reason

    if tool_name == "read_file":
        path = args.get("path", "") or args.get("file_path", "")
        ok, reason = check_file_read_safety(path)
        if not ok:
            return "blocked", reason

    return "safe", ""


def check_file_write_safety(path: str) -> Tuple[bool, str]:
    """Check if a file path is safe for a sub-agent to write."""
    if not path:
        return True, ""

    normalized = path.replace("\\", "/")

    for blocked in BLOCKED_WRITE_PATHS:
        if blocked.endswith("/"):
            if f"/{blocked}" in f"/{normalized}" or normalized.startswith(blocked):
                return False, f"Sub-agents cannot write to '{blocked}' paths"
        else:
            basename = os.path.basename(normalized)
            if basename == blocked or normalized.endswith(f"/{blocked}"):
                return False, f"Sub-agents cannot write to '{blocked}'"

    return True, ""


def check_file_read_safety(path: str) -> Tuple[bool, str]:
    """Check if a file path is safe for a sub-agent to read."""
    if not path:
        return True, ""

    normalized = path.replace("\\", "/")

    for blocked in BLOCKED_READ_PATHS:
        basename = os.path.basename(normalized)
        if basename == blocked or normalized.endswith(f"/{blocked}"):
            return False, f"Sub-agents cannot read '{blocked}'"

    return True, ""


def detect_file_conflicts(
    parent_reads: Set[str],
    child_writes: Set[str],
) -> List[str]:
    """
    After a child finishes, check if it wrote files the parent had read.

    This matters because the parent's cached view of those files is now
    stale. Returns list of conflicting paths the parent should re-read.
    """
    if not parent_reads or not child_writes:
        return []

    def _normalize(p: str) -> str:
        return os.path.normpath(p).replace("\\", "/").lower()

    parent_set = {_normalize(p) for p in parent_reads}
    conflicts = []

    for w in child_writes:
        if _normalize(w) in parent_set:
            conflicts.append(w)

    if conflicts:
        _log.info(
            "file_conflict | child wrote %d file(s) parent had read: %s",
            len(conflicts), conflicts,
        )

    return conflicts
