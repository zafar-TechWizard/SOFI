"""
BRAIN/tools/background_tools.py — Background (fire-and-forget) tools

These are background=True tools. SOFi dispatches them immediately after
acknowledging, without waiting for the result. The result flows back via
AgenticWorkspace and surfaces in WHAT I'VE BEEN DOING on the next turn.

All tools write to assistant/outputs/ so you can verify they actually ran.

Testing the orchestration pattern:
  1. Tell SOFi to write/save something → she acknowledges immediately
  2. The tool runs in the background
  3. Next message: SOFi naturally mentions it's done (from WHAT I'VE BEEN DOING)
  4. Check assistant/outputs/ to verify the file was actually written
"""

import asyncio
import datetime
import json
import logging
from pathlib import Path
from typing import Optional

from BRAIN.tools.registry import ToolEntry, ToolRegistry

_log = logging.getLogger("sofi.brain.bg_tools")

# Output directory — assistant/outputs/ (created if absent)
_OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


def _ensure_outputs_dir() -> Path:
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUTS_DIR


# =============================================================================
# TOOL HANDLERS
# =============================================================================

async def write_report(
    path: str,
    content: str,
    operation: str = "append",
    section_header: Optional[str] = None,
) -> str:
    """
    Write or append to a markdown report file.

    Args:
        path:           Filename (e.g. 'report.md', 'feature_spec.md').
                        Saved under assistant/outputs/.
        content:        Text to write. Plain text or markdown.
        operation:      'create' (overwrite), 'append' (add to end), or
                        'section' (append under a new ## header).
        section_header: Required when operation='section'. The ## heading text.
    """
    _log.info("write_report start | path=%s op=%s", path, operation)

    # Simulate brief work (in production this would be real I/O latency)
    await asyncio.sleep(0.1)

    out_dir = _ensure_outputs_dir()
    file_path = out_dir / Path(path).name  # strip any directory traversal
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        if operation == "create":
            file_path.write_text(
                f"# {Path(path).stem.replace('_', ' ').title()}\n"
                f"*Created by SOFi — {now}*\n\n"
                f"{content}\n",
                encoding="utf-8",
            )
            msg = f"Created {file_path.name} ({len(content)} chars)"

        elif operation == "section" and section_header:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(f"\n\n## {section_header}\n*{now}*\n\n{content}\n")
            msg = f"Added section '{section_header}' to {file_path.name}"

        else:  # append
            with file_path.open("a", encoding="utf-8") as f:
                f.write(f"\n{content}\n")
            msg = f"Appended to {file_path.name} ({len(content)} chars)"

        _log.info("write_report done | %s | file=%s", msg, file_path)
        return json.dumps({"status": "ok", "message": msg, "file": str(file_path)})

    except Exception as exc:
        _log.error("write_report failed | path=%s error=%s", path, exc)
        return json.dumps({"status": "error", "message": str(exc)})


async def save_note(title: str, content: str, tags: Optional[str] = None) -> str:
    """
    Save a quick note to notes.md in assistant/outputs/.
    Appends with timestamp so notes accumulate over time.
    """
    _log.info("save_note start | title=%s", title)
    await asyncio.sleep(0.05)

    out_dir = _ensure_outputs_dir()
    notes_file = out_dir / "notes.md"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    tag_line = f"*Tags: {tags}*  " if tags else ""

    entry = f"\n---\n### {title}\n*{now}*  {tag_line}\n\n{content}\n"

    try:
        with notes_file.open("a", encoding="utf-8") as f:
            f.write(entry)
        msg = f"Note '{title}' saved to notes.md"
        _log.info("save_note done | %s", msg)
        return json.dumps({"status": "ok", "message": msg, "file": str(notes_file)})
    except Exception as exc:
        _log.error("save_note failed | title=%s error=%s", title, exc)
        return json.dumps({"status": "error", "message": str(exc)})


async def log_decision(
    decision: str,
    context: str,
    category: Optional[str] = None,
) -> str:
    """
    Log a finalized decision to decisions.md in assistant/outputs/.
    Use this proactively when Zafar reaches a conclusion or finalizes something
    during conversation — no need to ask again once Zafar has set this up.
    """
    _log.info("log_decision start | decision=%.60s...", decision)
    await asyncio.sleep(0.05)

    out_dir = _ensure_outputs_dir()
    decisions_file = out_dir / "decisions.md"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cat_line = f"**Category:** {category}  " if category else ""

    entry = (
        f"\n---\n"
        f"### {now}\n"
        f"{cat_line}\n\n"
        f"**Decision:** {decision}\n\n"
        f"**Context:** {context}\n"
    )

    try:
        with decisions_file.open("a", encoding="utf-8") as f:
            f.write(entry)
        msg = f"Decision logged to decisions.md"
        _log.info("log_decision done | %s", msg)
        return json.dumps({"status": "ok", "message": msg, "file": str(decisions_file)})
    except Exception as exc:
        _log.error("log_decision failed | error=%s", exc)
        return json.dumps({"status": "error", "message": str(exc)})


async def create_task_item(
    title: str,
    description: str = "",
    priority: str = "normal",
) -> str:
    """
    Add a task to the todo list (tasks.md in assistant/outputs/).
    Background — SOFi dispatches and moves on immediately.
    """
    _log.info("create_task_item start | title=%s priority=%s", title, priority)
    await asyncio.sleep(0.05)

    out_dir = _ensure_outputs_dir()
    tasks_file = out_dir / "tasks.md"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    desc_line = f"   {description}" if description else ""
    priority_badge = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(priority, "🟡")

    entry = f"- [ ] {priority_badge} **{title}** *(added {now})*{chr(10) + desc_line if desc_line else ''}\n"

    try:
        with tasks_file.open("a", encoding="utf-8") as f:
            f.write(entry)
        msg = f"Task '{title}' added to tasks.md"
        _log.info("create_task_item done | %s", msg)
        return json.dumps({"status": "ok", "message": msg, "file": str(tasks_file)})
    except Exception as exc:
        _log.error("create_task_item failed | title=%s error=%s", title, exc)
        return json.dumps({"status": "error", "message": str(exc)})


async def simulate_slow_search(query: str, depth: str = "normal") -> str:
    """
    Simulate a slow background research operation (web search + summarize).
    Use this to test that SOFi stays non-blocked during long operations.
    Depth='deep' takes ~3s; depth='normal' takes ~1.5s.
    """
    _log.info("simulate_slow_search start | query=%.60s depth=%s", query, depth)
    delay = 3.0 if depth == "deep" else 1.5
    await asyncio.sleep(delay)

    summary = (
        f"Research on '{query}' complete. "
        f"Key findings: (1) Overview point about {query}. "
        f"(2) Current status and context. "
        f"(3) Relevant considerations for Zafar's work."
    )
    _log.info("simulate_slow_search done | query=%.60s delay=%.1fs", query, delay)
    return json.dumps({"status": "ok", "query": query, "summary": summary, "depth": depth})


# =============================================================================
# REGISTRATION
# =============================================================================

def register_background_tools(registry: ToolRegistry) -> None:
    """Register all background tools into the given registry."""

    registry.register(ToolEntry(
        name="write_report",
        description=(
            "Write or append content to a markdown report file. "
            "Background operation — runs immediately after SOFi responds, without blocking conversation. "
            "Use 'create' to start a new file, 'append' to add to an existing one, "
            "'section' to add a new ## section with a header. "
            "Use this proactively when Zafar finalizes points during a conversation "
            "that he has asked to be tracked — no need to ask each time."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filename for the report (e.g. 'report.md', 'feature_spec.md').",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write. Markdown is fine.",
                },
                "operation": {
                    "type": "string",
                    "enum": ["create", "append", "section"],
                    "description": "'create' overwrites, 'append' adds to end, 'section' adds a ## heading.",
                    "default": "append",
                },
                "section_header": {
                    "type": "string",
                    "description": "Heading text — required when operation='section'.",
                },
            },
            "required": ["path", "content"],
        },
        handler=write_report,
        background=True,
        category="workspace",
        capability_name="write_report",
        capability_description="Write and maintain report and specification files.",
    ))

    registry.register(ToolEntry(
        name="save_note",
        description=(
            "Save a quick note to notes.md. Background — non-blocking. "
            "Use when Zafar wants to capture something that doesn't need a full report."
        ),
        schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Note title.",
                },
                "content": {
                    "type": "string",
                    "description": "Note body text.",
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated tags (e.g. 'idea, project-x, follow-up').",
                },
            },
            "required": ["title", "content"],
        },
        handler=save_note,
        background=True,
        category="workspace",
        capability_name="save_note",
        capability_description="Save notes for later reference.",
    ))

    registry.register(ToolEntry(
        name="log_decision",
        description=(
            "Log a finalized decision or conclusion to decisions.md. Background — non-blocking. "
            "Use proactively when Zafar reaches a clear conclusion or makes a final call "
            "during conversation. Category examples: architecture, product, personal, technical."
        ),
        schema={
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "description": "The finalized decision or conclusion, stated clearly.",
                },
                "context": {
                    "type": "string",
                    "description": "Brief context — why this decision was made, what it resolves.",
                },
                "category": {
                    "type": "string",
                    "description": "Category label (e.g. 'architecture', 'product', 'technical', 'personal').",
                },
            },
            "required": ["decision", "context"],
        },
        handler=log_decision,
        background=True,
        category="workspace",
        capability_name="log_decision",
        capability_description="Log finalized decisions and conclusions for later reference.",
    ))

    registry.register(ToolEntry(
        name="create_task_item",
        description=(
            "Add a task to tasks.md. Background — non-blocking. "
            "Use when Zafar mentions something that needs to be done or tracked."
        ),
        schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Task title.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional details.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "normal", "low"],
                    "description": "Task priority.",
                    "default": "normal",
                },
            },
            "required": ["title"],
        },
        handler=create_task_item,
        background=True,
        category="workspace",
        capability_name="create_task",
        capability_description="Add tasks to the task list.",
    ))

    registry.register(ToolEntry(
        name="simulate_slow_search",
        description=(
            "Simulate a slow background research operation (for testing the fire-and-forget pattern). "
            "depth='normal' takes ~1.5s, depth='deep' takes ~3s. "
            "Use this to test that SOFi continues conversing while research runs in background."
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research query.",
                },
                "depth": {
                    "type": "string",
                    "enum": ["normal", "deep"],
                    "description": "How thorough: 'normal' (~1.5s) or 'deep' (~3s).",
                    "default": "normal",
                },
            },
            "required": ["query"],
        },
        handler=simulate_slow_search,
        background=True,
        category="information",
        capability_name="background_research",
        capability_description="Run a research query in the background while conversing.",
    ))


# Auto-discovery alias — brain.py looks for register(registry)
register = register_background_tools
