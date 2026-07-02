"""
BRAIN/tools/output_limits.py — Per-tool output size caps.

Every tool result flows through ToolRegistry.execute(). This module
provides truncate_result() which is called there as a universal
backstop — even if a tool self-limits, this enforces a hard cap.
"""

from typing import Dict

DEFAULT_MAX_RESULT_CHARS = 15_000

TOOL_OUTPUT_LIMITS: Dict[str, int] = {
    "read_file":        6_000,
    "list_directory":   8_000,
    "search_files":     8_000,
    "web_search":       6_000,
    "web_fetch":        4_000,
    "run_command":      5_000,
    "run_python":       5_000,
    "spawn_agent":     30_000,
    "write_file":       1_000,
    "patch_file":       1_000,
    "get_current_time":   500,
    "get_weather":      1_000,
    "skills_list":      2_000,
    "skills_load":     10_000,
    "search_in_files": 10_000,
    "file_info":        2_000,
}


def truncate_result(tool_name: str, text: str) -> str:
    """
    Truncate a tool result to its per-tool limit.

    Returns the text unchanged if under limit, or truncated with
    a marker showing original vs truncated size.
    """
    limit = TOOL_OUTPUT_LIMITS.get(tool_name, DEFAULT_MAX_RESULT_CHARS)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated from {len(text):,} to {limit:,} chars]"
