"""
BRAIN/tools/make_directory.py — Production directory creation tool.

Creates a directory and all necessary parent directories.

Workspace policy (enforced here, not in a middleware layer):
  • Internal paths (inside sofi-workspace/)   → created directly
  • External paths (anywhere else)            → mirrored to sofi-workspace/active/
    The real path on the filesystem is NEVER touched for external paths.

Edge cases handled:
  - Empty / whitespace-only path          → clear error
  - Path resolves to an existing file     → error with kind + fix hint
  - Path resolves to an existing dir      → success, already_existed=True
  - Path exists as a special node         → descriptive error (socket, pipe, …)
  - Permission denied                     → error with what to check
  - Windows MAX_PATH (260 chars)          → error with "enable long paths" hint
  - UNC / network paths                   → rejected (unpredictable behaviour)
  - Path depth > 64 components           → rejected (almost certainly a bug)
  - Symlinks in path components          → resolved before policy check
  - Creating inside backup/ directly     → rejected (would corrupt backup index)
  - Race condition (dir created between  → harmless — mkdir exist_ok catches it
    the exists-check and the mkdir call)
  - WorkspaceManager unavailable         → fails safe: rejects write, logs warning
  - Any other OSError                    → caught, message surfaced to caller

Uses Pattern B (@tool decorator) — auto-discovered by _auto_register_tools.
All blocking I/O runs in asyncio.to_thread.
"""

import asyncio
import logging
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from BRAIN.tools import tool

_log = logging.getLogger("sofi.brain.tools.make_directory")

# ── Constants ──────────────────────────────────────────────────────────────────

# Windows legacy MAX_PATH — guard here so tools that depend on the directory
# don't silently fail later with cryptic errors.
_WIN_MAX_PATH: int = 260

# Reject paths deeper than this regardless of OS. Anything this deep is either
# a bug (infinite loop writing paths) or a deliberate attempt to create a
# difficult-to-remove tree.
_MAX_DEPTH: int = 64

# Age threshold for the "last modified" annotation on existing dirs.
_RECENT_SECS: float = 300.0   # 5 minutes


# ── Pure helpers ───────────────────────────────────────────────────────────────

def _time_ago(ts: float) -> str:
    """Human-readable elapsed time since a Unix timestamp."""
    try:
        delta = datetime.now(timezone.utc).timestamp() - ts
    except Exception:
        return "?"

    if delta < 5:
        return "just now"
    if delta < 3_600:
        m = int(delta / 60)
        return f"{m}m ago" if m > 0 else "just now"
    if delta < 86_400:
        h = int(delta / 3_600)
        return f"{h}h ago"
    d = int(delta / 86_400)
    if d == 1:
        return "yesterday"
    if d < 30:
        return f"{d}d ago"
    if d < 365:
        w = int(d / 7)
        return f"{w}w ago"
    return f"{int(d / 365)}y ago"


def _kind_of(p: Path) -> str:
    """Classify a path into a human-readable type string."""
    try:
        if p.is_symlink():
            return "symlink"
        m = p.stat().st_mode
        if stat.S_ISREG(m):  return "file"
        if stat.S_ISDIR(m):  return "directory"
        if stat.S_ISCHR(m):  return "character device"
        if stat.S_ISBLK(m):  return "block device"
        if stat.S_ISFIFO(m): return "named pipe (FIFO)"
        if stat.S_ISSOCK(m): return "socket"
        return "unknown special file"
    except OSError:
        return "unknown"


def _count_parents_created(target: Path) -> int:
    """
    Count how many components of *target* do not yet exist.

    Used to generate "N directories created (including parents)" messages.
    Walks from target upward until an existing ancestor is found.
    """
    count = 0
    p = target
    while not p.exists():
        count += 1
        parent = p.parent
        if parent == p:  # hit the filesystem root
            break
        p = parent
    return count


# ── Workspace policy helpers ───────────────────────────────────────────────────

def _get_workspace_manager():
    """
    Return the WorkspaceManager singleton, or None if unavailable.

    Failure here is not an option — we fail safe: if the workspace manager
    cannot be loaded, we refuse any write operation rather than silently
    letting it through unguarded.
    """
    try:
        from BRAIN.tools.workspace import get_manager
        return get_manager()
    except Exception as exc:
        _log.warning("make_directory | workspace manager unavailable | %s", exc)
        return None


# ── Synchronous core (runs inside asyncio.to_thread) ──────────────────────────

def _mkdir_sync(target: Path, depth_hint: int) -> dict:
    """
    Create *target* and return a result dict.

    Called via asyncio.to_thread — all I/O is synchronous here.

    Returns a dict with keys:
      ok              bool
      already_existed bool   (only when ok=True)
      parents_created int    (how many new components were created)
      error           str    (only when ok=False)
      error_kind      str    (short category, for rendering)
    """
    # ── Conflict: path exists as a non-directory ───────────────────────────────
    # Check *before* mkdir so the error message names the kind precisely.
    if target.exists() and not target.is_dir():
        kind = _kind_of(target)
        return {
            "ok": False,
            "error_kind": "exists_as_non_dir",
            "error": (
                f"A {kind} already exists at this path.\n"
                f"  Path: {target}\n"
                f"  Fix:  Remove or rename the existing {kind} first."
            ),
        }

    # ── Already a directory ────────────────────────────────────────────────────
    if target.is_dir():
        try:
            mtime = target.stat().st_mtime
            age = _time_ago(mtime)
        except OSError:
            age = "?"
        return {
            "ok": True,
            "already_existed": True,
            "parents_created": 0,
            "mtime_label": age,
        }

    # ── Count how many new directories will be created (for the success msg) ──
    parents_to_create = _count_parents_created(target)

    # ── Create ─────────────────────────────────────────────────────────────────
    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        # Find the first ancestor that exists to report where access fails
        blame = target
        while not blame.parent.exists():
            blame = blame.parent
        blame = blame.parent
        return {
            "ok": False,
            "error_kind": "permission",
            "error": (
                f"Permission denied.\n"
                f"  Blocked at: {blame}\n"
                f"  Check write permissions on that directory.\n"
                f"  (OS: {exc})"
            ),
        }
    except FileExistsError:
        # Race: another process created it between our check and mkdir.
        # The directory exists — that's exactly what we wanted.
        pass
    except OSError as exc:
        return {
            "ok": False,
            "error_kind": "oserror",
            "error": f"Could not create directory: {exc}",
        }

    # ── Verify (defense in depth) ──────────────────────────────────────────────
    # mkdir should never succeed without producing a directory, but we've
    # seen this fail on network drives and some FUSE filesystems.
    if not target.is_dir():
        return {
            "ok": False,
            "error_kind": "verify_failed",
            "error": (
                f"mkdir appeared to succeed but {target} is not a directory.\n"
                f"  Possible causes: full disk, FUSE filesystem bug, network drive issue.\n"
                f"  Check df -h and the filesystem logs."
            ),
        }

    return {
        "ok": True,
        "already_existed": False,
        "parents_created": parents_to_create,
    }


# ── Output rendering ───────────────────────────────────────────────────────────

def _render_success(
    target: Path,
    already_existed: bool,
    parents_created: int,
    workspace_copy: bool,
    original: Optional[Path],
    active_root: Optional[Path],
    mtime_label: str = "",
) -> str:
    """Render a human-readable success message."""
    lines: list[str] = []

    if already_existed:
        lines.append(f"Already exists:  {target}")
        if mtime_label:
            lines.append(f"                 (directory · last modified {mtime_label})")
        if workspace_copy and original:
            lines.append(
                f"\n  This is the workspace copy of {original}. "
                f"The real path is unchanged."
            )
    else:
        verb = "Created (workspace copy):" if workspace_copy else "Created:"
        lines.append(f"{verb}  {target}")

        if parents_created > 1:
            lines.append(
                f"                  ({parents_created} directories created, including parents)"
            )
        elif parents_created == 1 and target.parent != target:
            # The directory itself was the only new one
            pass   # no extra note needed

        if workspace_copy and original and active_root:
            try:
                rel = target.relative_to(active_root)
            except ValueError:
                rel = target
            lines.append("")
            lines.append(f"  Actual path:   {target}")
            lines.append(f"  Mirrors real:  {original}")
            lines.append("")
            lines.append(
                "  The real path was NOT modified — this is a workspace copy.\n"
                "  Work inside this copy; use confirm_workspace_changes() when ready\n"
                "  to apply it back to the real location."
            )

    return "\n".join(lines)


def _render_error(error_text: str) -> str:
    """Render an error as a clean headed message."""
    return f"Error: make_directory failed\n{error_text}"


# ── Tool entry point ───────────────────────────────────────────────────────────

@tool(
    name="make_directory",
    description=(
        "Create a directory (and all intermediate parent directories) at the "
        "given path. Safe to call when the directory already exists — returns "
        "success without modifying anything.\n\n"
        "WORKSPACE POLICY:\n"
        "• Paths inside sofi-workspace/ are created directly.\n"
        "• External paths (anywhere else on the filesystem) are created inside "
        "sofi-workspace/active/ as a mirror. The real filesystem is not touched.\n\n"
        "WHEN TO USE:\n"
        "• Before write_file() — ensure the target directory exists first.\n"
        "• Setting up a project structure before populating it with files.\n"
        "• Creating output directories for run_command() or run_python().\n\n"
        "RETURNS:\n"
        "Plain text: the created path, whether it already existed, how many "
        "parent directories were created, and — for external paths — a note "
        "that the real path is unchanged and this is a workspace copy."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute or relative path to the directory to create. "
                    "Relative paths resolve from the current working directory. "
                    "All parent directories are created automatically. "
                    "Examples:\n"
                    "  'sofi-workspace/outputs/reports'\n"
                    "  'C:\\\\Users\\\\mdzaf\\\\projects\\\\myapp\\\\logs'\n"
                    "  '/home/user/my-project/data'\n"
                    "  './results/run-001'"
                ),
            },
        },
        "required": ["path"],
    },
    category="filesystem",
    timeout=10.0,
    capability_name="make_directory",
    capability_description=(
        "Create directories (and all parent directories) on the local filesystem. "
        "External paths are mirrored to sofi-workspace/active/ for safety."
    ),
    capability_refusal="I can't create directories right now.",
)
async def make_directory(path: str) -> str:
    """
    Public handler. All blocking I/O runs in asyncio.to_thread.
    """

    # ── Input validation ───────────────────────────────────────────────────────

    if not path or not path.strip():
        return _render_error("path must not be empty or whitespace-only.")

    raw = path.strip()

    # Reject UNC / network paths — they bypass workspace policy and have
    # unpredictable locking behaviour across OS/network configurations.
    #
    # Three detection layers (defense in depth):
    #   1. String prefix check (fast, catches obvious \\server and //server).
    #   2. os.path.splitdrive() — the Windows-idiomatic UNC detector; returns
    #      a drive component starting with "\\" for all UNC mount points.
    #   3. AFTER resolve(): checked again post-normalization to catch paths that
    #      look local but resolve through a UNC symlink (checked further below).
    _is_unc = (
        raw.startswith("\\\\")
        or raw.startswith("//")
        or os.path.splitdrive(raw)[0].startswith("\\\\")
    )
    if _is_unc:
        return _render_error(
            f"  Network/UNC paths are not supported: {raw!r}\n"
            f"  Use a local path instead."
        )

    # ── Path resolution ────────────────────────────────────────────────────────

    try:
        # expanduser handles "~" — friendlier than rejecting it.
        # resolve(strict=False) makes path absolute and collapses ".." without
        # requiring the path to already exist.
        resolved: Path = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        return _render_error(
            f"  Invalid path {raw!r}: {exc}"
        )

    # ── Post-resolution UNC check ─────────────────────────────────────────────
    # Catches paths that resolve through a UNC symlink even if the raw string
    # looked local. Uses splitdrive on the resolved absolute path.

    if os.path.splitdrive(str(resolved))[0].startswith("\\\\"):
        return _render_error(
            f"  Network/UNC path detected after resolution: {resolved}\n"
            f"  UNC paths are not supported — use a local path."
        )

    # ── Windows MAX_PATH ───────────────────────────────────────────────────────

    if os.name == "nt" and len(str(resolved)) > _WIN_MAX_PATH:
        return _render_error(
            f"  Path too long: {len(str(resolved))} characters "
            f"(Windows legacy limit is {_WIN_MAX_PATH}).\n"
            f"  Path: {resolved}\n"
            f"  Fix:  Shorten the path, or enable Long Path Support:\n"
            f"        Group Policy > Computer Configuration > Administrative Templates\n"
            f"        > System > Filesystem > Enable Win32 long paths"
        )

    # ── Depth guard ────────────────────────────────────────────────────────────

    depth = len(resolved.parts)
    if depth > _MAX_DEPTH:
        return _render_error(
            f"  Path has {depth} components (limit {_MAX_DEPTH}) — "
            f"this is almost certainly a bug.\n"
            f"  Path: {resolved}"
        )

    # ── Workspace policy ───────────────────────────────────────────────────────

    wm = _get_workspace_manager()
    if wm is None:
        return _render_error(
            "  WorkspaceManager is not available — cannot enforce workspace policy.\n"
            "  External filesystem writes are blocked until it's online.\n"
            "  Check BRAIN/tools/workspace/ and the startup log."
        )

    is_internal = wm.is_internal(resolved)

    if is_internal:
        target = resolved
        workspace_copy = False
        original_path: Optional[Path] = None
    else:
        # External path → redirect to active/ mirror
        target = wm.to_active_copy(resolved)
        workspace_copy = True
        original_path = resolved
        _log.debug(
            "make_directory | external path redirect | original=%s mirror=%s",
            resolved, target,
        )

    # ── Guard: never create inside backup/ ────────────────────────────────────
    # backup/ is managed exclusively by BackupManager. External code creating
    # directories there could corrupt the backup index.
    try:
        if target == wm.backup_root or wm.backup_root in target.parents:
            return _render_error(
                f"  Cannot create directories inside sofi-workspace/backup/.\n"
                f"  backup/ is managed by BackupManager — modifying it directly\n"
                f"  would corrupt the backup index.\n"
                f"  Use delete_file() / restore_backup() to interact with backups."
            )
    except Exception:
        pass   # skip guard if comparison fails — harmless

    # ── Also guard: target exists as a file in the mirror ─────────────────────
    # This catches the case where an external FILE was previously copied to
    # active/ and the LLM now tries to create a directory at the same mirror path.
    if workspace_copy and target.exists() and not target.is_dir():
        kind = _kind_of(target)
        return _render_error(
            f"  Cannot create directory mirror: a {kind} already exists at\n"
            f"  the active/ copy path: {target}\n"
            f"  This means an earlier tool created a file there. "
            f"Check sofi-workspace/active/."
        )

    # ── Delegate blocking mkdir to a thread ───────────────────────────────────

    result = await asyncio.to_thread(_mkdir_sync, target, depth)

    # ── Render result ──────────────────────────────────────────────────────────

    if not result["ok"]:
        _log.info(
            "make_directory | failed | kind=%s path=%s",
            result.get("error_kind"), target,
        )
        return _render_error(result["error"])

    already_existed: bool = result.get("already_existed", False)
    parents_created: int  = result.get("parents_created", 0)
    mtime_label: str      = result.get("mtime_label", "")

    _log.info(
        "make_directory | %s | path=%s workspace_copy=%s parents_created=%d",
        "already_existed" if already_existed else "created",
        target,
        workspace_copy,
        parents_created,
    )

    return _render_success(
        target          = target,
        already_existed = already_existed,
        parents_created = parents_created,
        workspace_copy  = workspace_copy,
        original        = original_path,
        active_root     = getattr(wm, "active_root", None) if workspace_copy else None,
        mtime_label     = mtime_label,
    )
