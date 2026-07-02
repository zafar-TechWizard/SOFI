"""
BRAIN/tools/exec_tools.py — Shell command and Python snippet execution.

Two tools:
  run_command  Execute a shell command and capture stdout + stderr.
  run_python   Execute a Python script (inline code or file path) in a subprocess.

Safety model (two tiers):
  Tier 1 — Hardline blocks: catastrophic patterns, always refused, no bypass,
            no confirmation offered. Checked inside the handler as a last-resort
            guard even when the pre-flight in brain.py already confirmed.
  Tier 2 — Dangerous patterns: destructive-but-recoverable operations. Flagged
            as needs_confirmation=True so brain.py asks Zafar before running.
            check_command_safety() / check_python_safety() expose this logic
            for any caller that wants to show a reason string.

Process lifecycle:
  - Subprocess runs with a hard timeout.
  - On timeout: SIGTERM sent first, then SIGKILL after 3s if still alive,
    then wait() to reap — never leaves zombies.
  - On completion: stdout + stderr are returned with timing and exit code.

Output limits (configurable, hard-capped):
  - stdout: up to 10 000 chars; large outputs truncated with head + tail shown.
  - stderr: up to 3 000 chars; truncated head-only (errors are usually short).
  - Combined hard cap: 15 000 chars to stay inside the LLM context budget.

Platform notes:
  - run_command uses the system shell (cmd.exe on Windows, /bin/sh on POSIX).
    PowerShell commands must be prefixed: powershell -Command "..."
  - run_python always uses sys.executable (the current venv / interpreter).
"""

import asyncio
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger("sofi.brain.tools.exec")

# Workspace root — default cwd for both tools.
_WORKSPACE_ROOT: Path = Path(__file__).parent.parent.parent.resolve()

# ── Output limits ──────────────────────────────────────────────────────────────

_STDOUT_LIMIT: int = 10_000   # chars shown from stdout
_STDERR_LIMIT: int  = 3_000   # chars shown from stderr
_HEAD_FRACTION: float = 0.25  # when truncating, show this fraction from the top


# ── Tier 1: Hardline blocks ────────────────────────────────────────────────────
# Substring match on lowercased, whitespace-normalised command.
# No confirmation offered. Hard refusal only.

_HARDLINE_BLOCKS: tuple[str, ...] = (
    # Filesystem nukes
    "rm -rf /",
    "rm -fr /",
    "rm --no-preserve-root",
    # Windows filesystem nukes
    "del /f /s /q c:\\",
    "del /f /s /q c:/",
    "rd /s /q c:\\",
    "rd /s /q c:/",
    "rmdir /s /q c:\\",
    "format c:",
    "remove-item -recurse -force c:\\",
    "remove-item -recurse -force c:/",
    # Disk destruction
    "dd if=/dev/zero of=/dev/sd",
    "dd if=/dev/random of=/dev/sd",
    "> /dev/sda",
    "mkfs",
    # Fork bomb
    ":(){ :|: & };:",
    # Kill init / all processes
    "kill -1 1",
    "taskkill /f /im *",
)


# ── Tier 2: Dangerous patterns (require confirmation) ─────────────────────────
# Regex patterns with a human-readable reason string.
# Checked after hardline passes. First match wins.

_DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    # File / directory deletion
    (r"\brm\s",                              "file/directory remove (rm)"),
    (r"\bdel\s",                             "file delete (del)"),
    (r"\berase\s",                           "file delete (erase)"),
    (r"\brmdir\b",                           "directory remove (rmdir)"),
    (r"\brd\s+/",                            "directory remove (rd /s)"),
    (r"\bremove-item\b",                     "file/directory delete (PowerShell Remove-Item)"),
    (r"\bunlink\b",                          "file unlink"),
    # Git destructive
    (r"\bgit\s+reset\s+--hard\b",           "hard git reset (local changes lost)"),
    (r"\bgit\s+clean\s+-[a-z]*[fdx]",       "destructive git clean"),
    (r"\bgit\s+push\s+[^\n]*--force",       "force push"),
    (r"\bgit\s+branch\s+-[dD]\b",           "branch delete"),
    # Database drops
    (r"\bdrop\s+table\b",                   "SQL DROP TABLE"),
    (r"\bdrop\s+database\b",               "SQL DROP DATABASE"),
    (r"\btruncate\s+table\b",              "SQL TRUNCATE TABLE"),
    # Download and execute (arbitrary remote code)
    (r"(curl|wget)\s[^\n]+\|\s*(ba)?sh",   "download and pipe to shell"),
    (r"(curl|wget)\s[^\n]+\|\s*python",    "download and pipe to python"),
    # Recursive permission / ownership blasts
    (r"\bchmod\s+-[rR]\b",                 "recursive chmod"),
    (r"\bchmod\s+777\b",                   "open permissions (chmod 777)"),
    (r"\bchown\s+-[rR]\b",                 "recursive chown"),
    # System shutdown / reboot
    (r"\b(shutdown|reboot|halt|poweroff)\b", "system shutdown/reboot"),
    (r"\bstop-computer\b",                 "PowerShell Stop-Computer"),
    (r"\brestart-computer\b",              "PowerShell Restart-Computer"),
)

_DANGEROUS_RE: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), reason)
    for pat, reason in _DANGEROUS_PATTERNS
)


# ── Python-specific dangerous patterns ────────────────────────────────────────

_PYTHON_DANGEROUS: tuple[tuple[str, str], ...] = (
    ("shutil.rmtree",  "shutil.rmtree (recursive delete)"),
    ("os.remove(",     "os.remove (file delete)"),
    ("os.unlink(",     "os.unlink (file delete)"),
    ("os.rmdir(",      "os.rmdir (directory remove)"),
    (".unlink(",       "pathlib .unlink() (file delete)"),
    (".rmdir(",        "pathlib .rmdir() (directory remove)"),
    ("shutil.move(",   "shutil.move (move / rename)"),
    ("os.system(",     "os.system — use run_command instead"),
)


# ── Safety API (exported for callers that need the reason string) ─────────────

def _normalise(command: str) -> str:
    """Collapse whitespace and strip control chars for reliable pattern matching."""
    c = re.sub(r"[\x00-\x1f\x7f]", " ", command.strip())
    return re.sub(r"\s+", " ", c)


def check_command_safety(command: str) -> tuple[str, str]:
    """
    Classify a shell command against the two-tier safety model.

    Returns:
        ("blocked",   reason) — hardline, never run, no bypass
        ("dangerous", reason) — requires confirmation before running
        ("safe",      "")     — run without asking
    """
    norm = _normalise(command)
    lower = norm.lower()

    for pattern in _HARDLINE_BLOCKS:
        if pattern in lower:
            return ("blocked", f"hardline block matched: {pattern!r}")

    for regex, reason in _DANGEROUS_RE:
        if regex.search(norm):
            return ("dangerous", reason)

    return ("safe", "")


def check_python_safety(code: str) -> tuple[str, str]:
    """
    Scan Python code for dangerous operations.

    Returns same ("blocked"|"dangerous"|"safe", reason) triple.
    """
    code_lower = code.lower()
    for pattern, reason in _PYTHON_DANGEROUS:
        if pattern in code_lower:
            return ("dangerous", reason)
    return ("safe", "")


# ── Output formatting helpers ─────────────────────────────────────────────────

def _truncate_output(text: str, limit: int, label: str = "") -> str:
    """
    Truncate *text* to *limit* chars, showing a head + tail split so
    both the beginning and end of long output are visible.

    The head fraction (_HEAD_FRACTION) is intentionally small — for most
    commands the important information is at the end (final state, result,
    error summary). But some output (e.g. test runner output) is structured
    top-to-bottom, so a small head preview helps orient the reader.
    """
    if len(text) <= limit:
        return text

    head_chars = int(limit * _HEAD_FRACTION)
    tail_chars = limit - head_chars
    dropped = len(text) - limit

    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()
    mid  = f"\n\n[... {dropped:,} chars omitted{' (' + label + ')' if label else ''} ...]\n\n"
    return head + mid + tail


def _format_result(
    command_display: str,
    cwd: str,
    exit_code: int,
    elapsed: float,
    stdout: str,
    stderr: str,
) -> str:
    """
    Build the human-readable result string returned by the tool.

    Format:
        $ <command>
        cwd: <path>  |  exit: <N>  |  time: <Xs>
        ──────────────────────────────────────────
        <stdout>

        stderr:
        <stderr>
    """
    exit_label = "✓" if exit_code == 0 else f"✗ {exit_code}"
    header = f"$ {command_display}"
    meta   = f"cwd: {cwd}  |  exit: {exit_label}  |  time: {elapsed:.2f}s"
    sep    = "─" * min(max(len(meta), len(header)) + 2, 80)

    parts = [header, meta, sep]

    if stdout:
        parts.append(_truncate_output(stdout, _STDOUT_LIMIT, "stdout"))
    if stderr:
        parts.append("")
        parts.append("stderr:")
        # Stderr: head-only truncation (errors read top-to-bottom)
        if len(stderr) > _STDERR_LIMIT:
            parts.append(stderr[:_STDERR_LIMIT] + f"\n[... {len(stderr) - _STDERR_LIMIT:,} chars omitted]")
        else:
            parts.append(stderr)
    if not stdout and not stderr:
        parts.append("(no output)")

    return "\n".join(parts)


# ── Process cleanup helper ─────────────────────────────────────────────────────

async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    """
    Graceful → forceful process shutdown.

    1. Send SIGTERM (or terminate() on Windows).
    2. Wait up to 3s for voluntary exit.
    3. If still alive, SIGKILL (or kill() on Windows).
    4. wait() to reap — no zombie processes.
    """
    try:
        proc.terminate()
    except (ProcessLookupError, OSError):
        return  # already gone

    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
        return
    except asyncio.TimeoutError:
        pass

    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        _log.warning("exec | process %s did not die after SIGKILL", proc.pid)


# ═══════════════════════════════════════════════════════════════════════════════
# run_command
# ═══════════════════════════════════════════════════════════════════════════════

async def run_command(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 30,
    env_vars: Optional[dict] = None,
    stdin_data: Optional[str] = None,
) -> str:
    """
    Execute a shell command and return stdout + stderr.

    Safety: hardline patterns are always blocked. Dangerous patterns are gated
    by needs_confirmation=True in the ToolEntry — brain.py asks Zafar first.
    A last-resort hardline check runs here too as defence-in-depth.
    """
    # ── Last-resort hardline guard ────────────────────────────────────────────
    tier, reason = check_command_safety(command)
    if tier == "blocked":
        return f"Command refused: {reason}\nCommand: {command!r}"

    # ── Parameter validation ──────────────────────────────────────────────────
    timeout = max(1, min(timeout, 300))   # clamp 1–300s

    if working_dir:
        cwd = Path(working_dir).expanduser().resolve()
        if not cwd.exists():
            return (
                f"Error: working_dir '{working_dir}' does not exist.\n"
                f"Run the command from an existing directory."
            )
        if not cwd.is_dir():
            return f"Error: working_dir '{working_dir}' is not a directory."
    else:
        cwd = _WORKSPACE_ROOT

    # ── Environment ───────────────────────────────────────────────────────────
    env = os.environ.copy()
    if env_vars:
        env.update({str(k): str(v) for k, v in env_vars.items()})

    # ── stdin ─────────────────────────────────────────────────────────────────
    stdin_bytes: Optional[bytes] = None
    if stdin_data is not None:
        stdin_bytes = stdin_data.encode("utf-8", errors="replace")

    # ── Subprocess ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    proc: Optional[asyncio.subprocess.Process] = None

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
            cwd=str(cwd),
            env=env,
        )

        try:
            raw_stdout, raw_stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            await _kill_process(proc)
            elapsed = time.perf_counter() - t0
            return (
                f"$ {command}\n"
                f"cwd: {cwd}  |  exit: TIMEOUT  |  time: {elapsed:.1f}s\n"
                f"─────────────────────────────────────────────────────\n"
                f"Command timed out after {timeout}s and was killed."
            )

        elapsed = time.perf_counter() - t0
        stdout  = raw_stdout.decode("utf-8", errors="replace").rstrip()
        stderr  = raw_stderr.decode("utf-8", errors="replace").rstrip()
        rc      = proc.returncode if proc.returncode is not None else -1

        return _format_result(command, str(cwd), rc, elapsed, stdout, stderr)

    except FileNotFoundError:
        # Shell not found — extremely rare, but guard it
        return f"Error: shell not found. Cannot execute: {command!r}"
    except PermissionError as exc:
        return f"Error: permission denied running command: {exc}"
    except Exception as exc:
        _log.error("run_command | unexpected error | cmd=%r err=%s", command, exc, exc_info=True)
        return f"Error running command: {type(exc).__name__}: {exc}"


def _classify_python_error(stderr: str) -> str:
    """
    Identify the exception type from Python's stderr output.
    Returns a short label like "SyntaxError", "ModuleNotFoundError", etc.
    Used to annotate the result header for quick diagnosis.
    """
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line or line.startswith(" ") or line.startswith("^"):
            continue
        m = re.match(r"^([A-Z][A-Za-z]+(?:Error|Exception|Warning|Interrupt))\b", line)
        if m:
            return m.group(1)
        m2 = re.match(r"^(KeyboardInterrupt|SystemExit|GeneratorExit)$", line)
        if m2:
            return m2.group(1)
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# run_python
# ═══════════════════════════════════════════════════════════════════════════════

async def run_python(
    code: str,
    timeout: int = 30,
    stdin_data: Optional[str] = None,
) -> str:
    """
    Execute a Python code snippet in a subprocess and return the output.

    The code runs as a standalone script using the current interpreter
    (sys.executable — respects active venv). The working directory is the
    assistant workspace root so relative imports from the project work.

    Safety: dangerous patterns (file deletion, os.system) are gated by
    needs_confirmation=True. A last-resort check runs here too.
    """
    # ── Last-resort dangerous check ───────────────────────────────────────────
    tier, reason = check_python_safety(code)
    if tier == "blocked":
        return f"Code refused: {reason}"

    # ── Parameter validation ──────────────────────────────────────────────────
    timeout = max(1, min(timeout, 120))   # clamp 1–120s

    if not code.strip():
        return "Error: no code provided."

    # ── Write code to a temp file ─────────────────────────────────────────────
    # NamedTemporaryFile with delete=False so it can be opened by the subprocess
    # on Windows (Windows can't open a file that's already open by the creating
    # process when using delete=True).
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="sofi_",
            delete=False,
            encoding="utf-8",
            dir=tempfile.gettempdir(),
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)

        # ── stdin ─────────────────────────────────────────────────────────────
        stdin_bytes: Optional[bytes] = None
        if stdin_data is not None:
            stdin_bytes = stdin_data.encode("utf-8", errors="replace")

        # ── Subprocess ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        proc: Optional[asyncio.subprocess.Process] = None

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
                cwd=str(_WORKSPACE_ROOT),
            )

            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_bytes),
                    timeout=float(timeout),
                )
            except asyncio.TimeoutError:
                await _kill_process(proc)
                elapsed = time.perf_counter() - t0
                return (
                    f"Python [TIMEOUT after {timeout}s — killed]\n"
                    f"─────────────────────────────────────────────\n"
                    f"The script was still running after {timeout}s and was killed.\n"
                    f"Consider breaking it into smaller pieces or using a longer timeout."
                )

            elapsed = time.perf_counter() - t0
            stdout  = raw_stdout.decode("utf-8", errors="replace").rstrip()
            stderr  = raw_stderr.decode("utf-8", errors="replace").rstrip()
            rc      = proc.returncode if proc.returncode is not None else -1

            # ── Classify the result ───────────────────────────────────────────
            # Detect common Python error types from stderr for a cleaner header.
            exit_label  = "✓" if rc == 0 else f"✗ {rc}"
            error_type  = _classify_python_error(stderr) if rc != 0 and stderr else ""
            header_note = f" [{error_type}]" if error_type else ""

            header = f"Python [exit: {exit_label}{header_note}  |  time: {elapsed:.2f}s]"
            sep    = "─" * min(len(header) + 2, 80)

            parts = [header, sep]

            if stdout:
                parts.append(_truncate_output(stdout, _STDOUT_LIMIT, "stdout"))

            if stderr:
                if stdout:
                    parts.append("")
                if rc != 0:
                    parts.append("error output:")
                else:
                    parts.append("stderr:")
                if len(stderr) > _STDERR_LIMIT:
                    parts.append(
                        stderr[:_STDERR_LIMIT]
                        + f"\n[... {len(stderr) - _STDERR_LIMIT:,} chars omitted]"
                    )
                else:
                    parts.append(stderr)

            if not stdout and not stderr:
                if rc == 0:
                    parts.append("(script completed with no output)")
                else:
                    parts.append(f"(script exited with code {rc}, no output captured)")

            return "\n".join(parts)

        except FileNotFoundError:
            return f"Error: Python interpreter not found: {sys.executable}"
        except Exception as exc:
            _log.error("run_python | unexpected error | err=%s", exc, exc_info=True)
            return f"Error running Python: {type(exc).__name__}: {exc}"

    finally:
        # Always clean up the temp file, even on exception / timeout
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
            "Execute a shell command on Zafar's machine and return stdout, stderr, "
            "exit code, and execution time.\n\n"
            "Platform: Windows uses cmd.exe. For PowerShell commands, prefix with "
            "'powershell -Command \"...\"'. Git, pip, npm, and most CLIs work directly.\n\n"
            "Good for: git operations (git status, git log, git commit), package installs "
            "(pip install, npm install), build scripts, checking system state, running CLIs.\n\n"
            "Optional: pass env_vars to add/override environment variables for this run. "
            "Pass stdin_data to pipe text into the command's stdin.\n\n"
            "Output: stdout and stderr are returned separately. Large outputs are "
            "truncated (head + tail shown) with a notice. Timeout kills the process "
            "gracefully (SIGTERM → SIGKILL).\n\n"
            "Safety: catastrophic patterns (rm -rf /, format c:) are always blocked. "
            "Destructive-but-recoverable patterns (rm, del, git reset --hard, force push) "
            "require confirmation."
        ),
        schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The shell command to execute. Supports pipes, redirects, "
                        "and multi-command sequences (&&, ;). "
                        "On Windows: cmd.exe syntax. For PowerShell: prefix with "
                        "'powershell -Command \"...\"'."
                    ),
                },
                "working_dir": {
                    "type": "string",
                    "description": (
                        "Working directory for the command. Absolute path. "
                        "Default: the assistant workspace root."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait before killing the process. Range: 1–300. Default: 30.",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 30,
                },
                "env_vars": {
                    "type": "object",
                    "description": (
                        "Additional environment variables to set for this command. "
                        "Merges with the current environment — existing vars are not removed. "
                        "Example: {\"DEBUG\": \"1\", \"API_KEY\": \"xxx\"}"
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "stdin_data": {
                    "type": "string",
                    "description": (
                        "Text to pipe into the command's stdin. "
                        "Useful for commands that read from stdin (e.g. python scripts, "
                        "interactive CLI tools in non-interactive mode)."
                    ),
                },
            },
            "required": ["command"],
        },
        handler=run_command,
        needs_confirmation=True,
        timeout=320.0,      # tool timeout > max command timeout so we get the result
        category="execution",
        capability_name="run_command",
        capability_description=(
            "Execute any shell command on Zafar's local machine — git, pip, npm, "
            "build scripts, CLI tools, system queries. Requires confirmation for "
            "destructive operations."
        ),
        capability_refusal="I won't run that command.",
    ))

    registry.register(ToolEntry(
        name="run_python",
        description=(
            "Execute a Python code snippet in a subprocess and return its output.\n\n"
            "The code runs using the current Python interpreter (respects the active "
            "venv). Working directory is the assistant workspace root, so project "
            "imports work.\n\n"
            "Good for: data processing, calculations, file manipulation, "
            "testing an idea, running a script that doesn't have a CLI entry point, "
            "generating data for further use.\n\n"
            "Use print() for output — the return value of the last expression is not "
            "automatically printed (this is a subprocess, not a REPL).\n\n"
            "Pass stdin_data to provide input to input() calls in the script.\n\n"
            "Output: stdout and stderr are returned with the exit code and timing. "
            "Python exception type is shown in the header on failure "
            "(e.g. [SyntaxError], [ModuleNotFoundError]).\n\n"
            "Safety: file-deletion patterns (os.remove, shutil.rmtree, .unlink()) "
            "require confirmation."
        ),
        schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source code to execute. Write a complete script — "
                        "use print() to produce output. Can import any installed package. "
                        "Runs in the assistant workspace root, so project imports work."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait before killing the process. Range: 1–120. Default: 30.",
                    "minimum": 1,
                    "maximum": 120,
                    "default": 30,
                },
                "stdin_data": {
                    "type": "string",
                    "description": (
                        "Text to send to the script's stdin. "
                        "Provide newline-separated values if the script calls input() "
                        "multiple times."
                    ),
                },
            },
            "required": ["code"],
        },
        handler=run_python,
        needs_confirmation=True,
        timeout=130.0,      # tool timeout > max script timeout so we get the result
        category="execution",
        capability_name="run_python",
        capability_description=(
            "Execute Python code snippets in a subprocess. Respects the active venv. "
            "Shows stdout, stderr, exit code, execution time, and exception type on failure."
        ),
        capability_refusal="I won't run that code.",
    ))


# Auto-discovery alias — _auto_register_tools looks for register(registry)
register = register_exec_tools
