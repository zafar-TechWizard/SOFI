"""
BRAIN/tools/exec_tools.py — Code and command execution tools for SOFi

- run_command : Execute a shell command and return output
- run_python  : Execute a Python code snippet and return output

Both require confirmation before execution (needs_confirmation=True).
Both are inline (not background) — result feeds back into the LLM's next response.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

_log = logging.getLogger("sofi.brain.tools.exec")

_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.resolve()

# Hard-blocked patterns — substring match on the lowercased command
_BLOCKED = [
    "rm -rf /",
    "del /f /s /q c:",
    "format c:",
    ":(){ :|: & };:",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "mkfs",
    "> /dev/sda",
]


def _safe(command: str) -> bool:
    c = command.lower().strip()
    return not any(b in c for b in _BLOCKED)


# ── Run Shell Command ─────────────────────────────────────────────────────────

async def run_command(
    command: str,
    working_dir: str = None,
    timeout: int = 30,
) -> str:
    if not _safe(command):
        return f"Command blocked for safety. Not running: {command!r}"

    timeout = max(1, min(timeout, 120))  # clamp 1–120s
    cwd = Path(working_dir).expanduser().resolve() if working_dir else _WORKSPACE_ROOT

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return f"Command timed out after {timeout}s: {command!r}"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        rc = proc.returncode

        parts = [f"$ {command}  [exit {rc}]"]
        if out:
            # Cap stdout at 3000 chars
            parts.append(out[:3000] + (" [... truncated]" if len(out) > 3000 else ""))
        if err:
            parts.append(f"stderr:\n{err[:800]}" + (" [... truncated]" if len(err) > 800 else ""))
        if not out and not err:
            parts.append("(no output)")

        return "\n".join(parts)

    except Exception as exc:
        _log.error("run_command error cmd=%r: %s", command, exc)
        return f"Error running command: {exc}"


# ── Run Python ────────────────────────────────────────────────────────────────

async def run_python(code: str, timeout: int = 15) -> str:
    timeout = max(1, min(timeout, 60))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_WORKSPACE_ROOT),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return f"Python execution timed out after {timeout}s."

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        rc = proc.returncode

        parts = [f"Python [exit {rc}]:"]
        if out:
            parts.append(out[:3000] + (" [... truncated]" if len(out) > 3000 else ""))
        if err:
            parts.append(f"stderr:\n{err[:800]}")
        if not out and not err:
            parts.append("(no output)")

        return "\n".join(parts)

    except Exception as exc:
        _log.error("run_python error: %s", exc)
        return f"Error running Python: {exc}"

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ── Registration ──────────────────────────────────────────────────────────────

def register_exec_tools(registry) -> None:
    from BRAIN.tools.registry import ToolEntry

    registry.register(ToolEntry(
        name="run_command",
        description=(
            "Execute a shell command on Zafar's machine and return stdout/stderr. "
            "On Windows this runs via the system shell (cmd/PowerShell). "
            "Good for: git operations, file management, pip installs, checking system info, "
            "running scripts, checking processes, directory operations. "
            "Requires Zafar's confirmation before running. Result feeds back immediately."
        ),
        schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory path (default: assistant workspace root)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (1–120, default 30)",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
        handler=run_command,
        needs_confirmation=True,
        category="execution",
        capability_name="run_command",
        capability_description="Execute shell commands on Zafar's local machine.",
        capability_refusal="I won't run that command.",
    ))

    registry.register(ToolEntry(
        name="run_python",
        description=(
            "Execute a Python code snippet on Zafar's machine and return the output. "
            "Runs in a subprocess with the assistant workspace root as cwd. "
            "Can import any installed package. Good for: data processing, calculations, "
            "file manipulation, quick scripts, testing an idea. "
            "Requires Zafar's confirmation before running."
        ),
        schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute (full script, use print() for output)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (1–60, default 15)",
                    "default": 15,
                },
            },
            "required": ["code"],
        },
        handler=run_python,
        needs_confirmation=True,
        category="execution",
        capability_name="run_python",
        capability_description="Execute Python code snippets on Zafar's machine.",
        capability_refusal="I won't run that code.",
    ))
