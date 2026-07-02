"""
BRAIN/agents/result.py — Structured sub-agent output.

Every sub-agent run produces a SubAgentResult — not raw text.
Contains everything needed for delivery, debugging, cost tracking,
and file conflict detection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolTraceEntry:
    """One tool invocation recorded during a sub-agent run."""

    tool: str
    args_preview: str
    result_bytes: int
    status: str  # "ok" | "error" | "blocked" | "timeout"
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args_preview": self.args_preview,
            "result_bytes": self.result_bytes,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class SubAgentResult:
    """
    Structured output from a sub-agent run.

    This is what flows back to the parent — never raw text.
    Contains everything needed for delivery, debugging, and cost tracking.
    """

    subagent_id: str
    task_id: str
    agent_type: str

    # Outcome
    status: str  # "completed" | "failed" | "timeout" | "interrupted" | "budget_exhausted"
    exit_reason: str  # "final_response" | "max_iterations" | "timeout" | "interrupted" | "error" | "budget_exhausted"

    # Content
    summary: str
    content: str

    # Metrics
    iterations: int
    duration_seconds: float
    tokens: Dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    tool_trace: List[ToolTraceEntry] = field(default_factory=list)

    # File tracking
    files_written: List[str] = field(default_factory=list)
    file_conflicts: List[str] = field(default_factory=list)

    # Error info
    error: Optional[str] = None

    def to_delivery(self) -> dict:
        """Convert to TaskManager delivery format."""
        if self.status == "completed":
            delivery_status = "fulfilled"
        elif self.status in ("timeout", "budget_exhausted", "interrupted"):
            delivery_status = "partial"
        else:
            delivery_status = "failed"

        return {
            "status": delivery_status,
            "summary": self.summary,
            "content": self.content,
            "gaps": self.error,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "exit_reason": self.exit_reason,
            "summary": self.summary,
            "content_length": len(self.content),
            "iterations": self.iterations,
            "duration_seconds": round(self.duration_seconds, 2),
            "tokens": self.tokens,
            "tool_trace": [t.to_dict() for t in self.tool_trace],
            "files_written": self.files_written,
            "file_conflicts": self.file_conflicts,
            "error": self.error,
        }

    def metrics_dict(self) -> dict:
        """Compact metrics for TaskManager storage."""
        return {
            "iterations": self.iterations,
            "tokens": self.tokens,
            "duration_seconds": round(self.duration_seconds, 2),
            "tool_count": len(self.tool_trace),
            "tools_used": list({t.tool for t in self.tool_trace}),
            "exit_reason": self.exit_reason,
        }


def extract_summary(content: str, max_len: int = 200) -> str:
    """Extract a one-line summary from sub-agent output."""
    lines = content.strip().split("\n")
    for line in lines:
        clean = line.strip().lstrip("#").strip()
        if clean and len(clean) > 10:
            return clean[:max_len]
    return content[:max_len] if content else ""
