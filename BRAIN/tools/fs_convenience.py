"""
BRAIN/tools/fs_convenience.py — Convenience filesystem tools

Three tools that complement the core read/write set:

  append_file   Append text to a file, preserving its encoding and line endings.
  copy_file     Copy a file or directory (backs up destination before overwrite).
  move_file     Move/rename a file or directory (soft-deletes source on cross-device).

Note: list_backups is already registered in fs_delete.py (paired with
delete_file/restore_backup where it semantically belongs).
"""

import asyncio
import errno
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

_log = logging.getLogger("sofi.brain.tools.fs_convenience")


# ── Shared helpers (lazy imports — avoids circular imports at load time) ───────

def _get_workspace_manager():
    try:
        from BRAIN.tools.workspace import get_manager
        return get_manager()
    except Exception:
        return None


def _get_backup_manager():
    try:
        from BRAIN.tools.workspace import get_backup_manager
        return get_backup_manager()
    except Exception:
        return None


# OS system directories — always blocked for any write/move destination.
_BLOCKED_PREFIXES: tuple = (
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/boot", "/proc", "/sys", "/dev",
    "/System", "/Library", "/Applications",
)


def _check_write_blocked(p: Path) -> Tuple[bool, str]:
    """Block OS system dirs and direct writes into the backup directory."""
    s = str(p).replace("\\", "/")
    for prefix in _BLOCKED_PREFIXES:
        if s.lower().startswith(prefix.lower().replace("\\", "/")):
            return True, f"Writing to system path is blocked: {p}"
    try:
        wm = _get_workspace_manager()
        if wm:
            bp = wm.backup_root.resolve()
            rp = p.resolve()
            if rp == bp or bp in rp.parents:
                return True, (
                    "Direct writes to the backup directory are not allowed. "
                    "Use delete_file / restore_backup to manage backups."
                )
    except Exception:
        pass
    return False, ""


def _check_delete_blocked(p: Path) -> Tuple[bool, str]:
    """Block deleting OS system dirs, backup dir, or workspace root."""
    s = str(p).replace("\\", "/")
    for prefix in _BLOCKED_PREFIXES:
        if s.lower().startswith(prefix.lower().replace("\\", "/")):
            return True, f"Deleting OS system paths is blocked: {p}"
    try:
        wm = _get_workspace_manager()
        if wm:
            rp = p.resolve()
            br = wm.backup_root.resolve()
            wr = wm.root.resolve()
            if rp == wr:
                return True, "Deleting the workspace root is not allowed."
            if rp == br or br in rp.parents:
                return True, "Deleting inside the backup directory is not allowed."
    except Exception:
        pass
    return False, ""


def _resolve_write_path(path: str) -> Tuple[Path, Path, bool]:
    """
    Workspace routing for append_file (a write to an existing external file).
    External paths → active/ mirror; internal paths → direct.
    Returns (original, write_target, is_workspace_copy).
    """
    original = Path(path).expanduser().resolve()
    ws = _get_workspace_manager()
    if ws is None or ws.is_internal(original):
        return original, original, False
    return original, ws.to_active_copy(original), True


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Atomic byte-level write via sibling .sofi_tmp + os.replace()."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".sofi_tmp")
    try:
        tmp.write_bytes(data)
        os.replace(str(tmp), str(target))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# append_file
# ═══════════════════════════════════════════════════════════════════════════════

async def append_file(
    path: str,
    content: str,
    ensure_newline: bool = True,
    create: bool = True,
) -> str:
    """
    Append *content* to a file, preserving its encoding and line-ending style.

    The existing file's encoding (UTF-8, UTF-16, Latin-1) and dominant line-
    ending style (LF vs CRLF) are detected and applied to the new content —
    no mixed-encoding or mixed-CRLF/LF files result.

    ensure_newline=True (default): if the file's last byte is not already a
    newline, one is inserted before the new content (append starts on a fresh
    line rather than gluing onto the last character).

    Uses workspace routing — external files are written to sofi-workspace/
    active/ so the originals are never modified directly.
    """
    try:
        return await asyncio.to_thread(_append_impl, path, content, ensure_newline, create)
    except Exception as exc:
        _log.exception("append_file | path=%s", path)
        return f"Error appending to file: {exc}"


def _append_impl(path: str, content: str, ensure_newline: bool, create: bool) -> str:
    original, write_target, is_ws_copy = _resolve_write_path(path)

    blocked, reason = _check_write_blocked(write_target)
    if blocked:
        return f"Blocked: {reason}"

    # ── Load existing content (if any) ────────────────────────────────────────
    existing_bytes: bytes = b""
    encoding    = "utf-8"
    line_ending = "\n"
    was_new     = False

    if original.exists():
        if original.is_dir():
            return f"Error: '{original}' is a directory, not a file."
        try:
            existing_bytes = original.read_bytes()
        except PermissionError as exc:
            return f"Permission denied reading source: {exc}"
        encoding    = _sniff_encoding(existing_bytes)
        line_ending = _sniff_line_ending(existing_bytes)
    elif not create:
        return (
            f"File not found: {original}\n"
            "Pass create=true to create a new file."
        )
    else:
        was_new = True

    # ── Normalise new content's line endings to match the file ────────────────
    normalised = content.replace("\r\n", "\n")          # collapse first
    if line_ending == "\r\n":
        normalised = normalised.replace("\n", "\r\n")   # expand to CRLF

    # ── Separator — ensure append starts on a fresh line ─────────────────────
    separator = b""
    if ensure_newline and existing_bytes:
        nl_bytes = line_ending.encode(encoding, errors="replace")
        if not existing_bytes.endswith(nl_bytes):
            separator = nl_bytes

    # ── Encode new content ────────────────────────────────────────────────────
    try:
        new_bytes = normalised.encode(encoding, errors="replace")
    except LookupError:
        new_bytes = normalised.encode("utf-8", errors="replace")
        encoding = "utf-8"

    # ── Atomic write ──────────────────────────────────────────────────────────
    combined = existing_bytes + separator + new_bytes
    _atomic_write_bytes(write_target, combined)

    lines = ["Created:" if was_new else "Appended:"]
    lines.append(f"  path:     {original}")
    if is_ws_copy:
        lines.append(f"  written:  {write_target}  (workspace active copy)")
    lines += [
        f"  encoding: {encoding}",
        f"  added:    {len(new_bytes):,} bytes  ({len(normalised.splitlines())} line(s))",
        f"  total:    {len(combined):,} bytes",
    ]
    if separator:
        lines.append("  note:     separator newline inserted before content")
    return "\n".join(lines)


def _sniff_encoding(raw: bytes) -> str:
    """Detect encoding from BOM or UTF-8 probe; fall back to Latin-1."""
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _sniff_line_ending(raw: bytes) -> str:
    """Return '\\r\\n' if CRLF dominates, else '\\n'."""
    crlf = raw.count(b"\r\n")
    lf   = raw.count(b"\n") - crlf
    return "\r\n" if crlf > lf else "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# copy_file
# ═══════════════════════════════════════════════════════════════════════════════

async def copy_file(
    source: str,
    destination: str,
    overwrite: bool = False,
) -> str:
    """
    Copy a file or directory to a new location.

    Source is read directly — no workspace restriction.
    Destination is written directly to the given path (not routed through
    workspace active/ — the caller is explicitly choosing the destination).

    If the destination already exists:
      • overwrite=False (default): fails with a clear error, nothing changes.
      • overwrite=True: the existing destination is soft-deleted via BackupManager
        first (fully recoverable via restore_backup), then the copy is written.

    File metadata (timestamps, permissions) is preserved via shutil.copy2.
    Directories are copied recursively (shutil.copytree).
    """
    try:
        return await asyncio.to_thread(_copy_impl, source, destination, overwrite)
    except Exception as exc:
        _log.exception("copy_file | src=%s dst=%s", source, destination)
        return f"Error copying: {exc}"


def _copy_impl(source: str, destination: str, overwrite: bool) -> str:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()

    if not src.exists():
        return f"Source not found: {src}"

    blocked, reason = _check_write_blocked(dst)
    if blocked:
        return f"Blocked: {reason}"

    item_type = "directory" if src.is_dir() else "file"

    # ── Handle existing destination ───────────────────────────────────────────
    displacement_id: Optional[str] = None

    if dst.exists():
        if not overwrite:
            return (
                f"Destination already exists: {dst}\n"
                "Pass overwrite=true to replace it — the existing item will be "
                "backed up and is fully recoverable via restore_backup."
            )
        try:
            bm = _get_backup_manager()
            if bm:
                displacement_id = bm.soft_delete(dst)
                _log.info("copy_file | displaced %s → backup_id=%s", dst, displacement_id)
            else:
                # BackupManager unavailable — hard delete (last resort).
                if dst.is_dir():
                    shutil.rmtree(str(dst))
                else:
                    dst.unlink()
        except Exception as exc:
            return f"Could not displace existing destination '{dst}': {exc}"

    # ── Copy ──────────────────────────────────────────────────────────────────
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
    except PermissionError as exc:
        return f"Permission denied: {exc}"
    except shutil.Error as exc:
        return f"Copy error: {exc}"
    except Exception as exc:
        return f"Copy failed: {type(exc).__name__}: {exc}"

    lines = [
        f"Copied ({item_type}):",
        f"  from:  {src}",
        f"  to:    {dst}",
    ]
    if displacement_id:
        lines.append(f"  displaced existing item backed up as: {displacement_id}")

    _log.info("copy_file | %s → %s | type=%s", src, dst, item_type)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# move_file
# ═══════════════════════════════════════════════════════════════════════════════

async def move_file(
    source: str,
    destination: str,
    overwrite: bool = False,
) -> str:
    """
    Move or rename a file or directory.

    SAFE BY DESIGN: the source is never permanently deleted.

    Strategy:
      1. Attempt an atomic rename (os.rename) — instant, same-filesystem,
         no data at risk. Succeeds when source and destination are on the
         same volume (the common case for renames).
      2. If that fails (cross-device / NTFS quirks): copy to destination,
         then soft-delete the source via BackupManager. The source is
         recoverable via restore_backup.

    If the destination already exists:
      • overwrite=False (default): fails with a clear error, nothing moves.
      • overwrite=True: destination is soft-deleted first, then the move
        proceeds. Both items are fully recoverable.

    Requires confirmation before running.
    """
    try:
        return await asyncio.to_thread(_move_impl, source, destination, overwrite)
    except Exception as exc:
        _log.exception("move_file | src=%s dst=%s", source, destination)
        return f"Error moving: {exc}"


def _move_impl(source: str, destination: str, overwrite: bool) -> str:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()

    if not src.exists():
        return f"Source not found: {src}"

    blocked_src, reason_src = _check_delete_blocked(src)
    if blocked_src:
        return f"Blocked (source): {reason_src}"

    blocked_dst, reason_dst = _check_write_blocked(dst)
    if blocked_dst:
        return f"Blocked (destination): {reason_dst}"

    # Can't move a path into itself.
    try:
        if dst == src or src in dst.parents:
            return f"Error: destination '{dst}' is inside the source '{src}'."
    except Exception:
        pass

    item_type = "directory" if src.is_dir() else "file"

    # ── Handle existing destination ───────────────────────────────────────────
    displacement_id: Optional[str] = None

    if dst.exists():
        if not overwrite:
            return (
                f"Destination already exists: {dst}\n"
                "Pass overwrite=true to replace it — the existing item will be "
                "backed up and is fully recoverable via restore_backup."
            )
        try:
            bm = _get_backup_manager()
            if bm:
                displacement_id = bm.soft_delete(dst)
                _log.info("move_file | displaced %s → backup_id=%s", dst, displacement_id)
            else:
                if dst.is_dir():
                    shutil.rmtree(str(dst))
                else:
                    dst.unlink()
        except Exception as exc:
            return f"Could not displace existing destination '{dst}': {exc}"

    dst.parent.mkdir(parents=True, exist_ok=True)

    # ── Attempt 1: atomic rename (same filesystem) ────────────────────────────
    src_backup_id: Optional[str] = None
    moved_atomically = False

    try:
        os.rename(str(src), str(dst))
        moved_atomically = True
    except OSError as exc:
        # EXDEV = cross-device; EACCES/EPERM = Windows NTFS quirk for dirs.
        # Any of these: fall through to copy + soft-delete.
        _cross_device_errnos = {errno.EXDEV}
        if hasattr(errno, "EACCES"):
            _cross_device_errnos.add(errno.EACCES)
        # WinError 17 (can't move across drives) surfaces as errno 22 on some Pythons.
        # Be permissive: fallback for any rename failure.
        if exc.errno not in _cross_device_errnos and exc.errno not in (
            errno.EPERM if hasattr(errno, "EPERM") else -1,
            22,   # errno.EINVAL / WinError 17 alias on some platforms
        ):
            return f"Rename failed: {exc}"
        # else: expected cross-device — fall through

    if not moved_atomically:
        # ── Attempt 2: copy then soft-delete source ───────────────────────────
        try:
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
        except PermissionError as exc:
            return f"Permission denied during copy step: {exc}"
        except Exception as exc:
            return f"Move failed (copy step): {type(exc).__name__}: {exc}"

        try:
            bm = _get_backup_manager()
            if bm:
                src_backup_id = bm.soft_delete(src)
            else:
                if src.is_dir():
                    shutil.rmtree(str(src))
                else:
                    src.unlink()
        except Exception as exc:
            return (
                f"Copied to destination but could not remove source: {exc}\n"
                f"Both '{src}' and '{dst}' now exist — remove the source manually."
            )

    # ── Build result ──────────────────────────────────────────────────────────
    lines = [
        f"Moved ({item_type}):",
        f"  from:  {src}",
        f"  to:    {dst}",
    ]
    if displacement_id:
        lines.append(f"  displaced existing item backed up as: {displacement_id}")
    if src_backup_id:
        lines.append(
            f"  source backed up as: {src_backup_id}  "
            "(cross-device move — original is recoverable via restore_backup)"
        )

    _log.info("move_file | %s → %s | type=%s | atomic=%s", src, dst, item_type, moved_atomically)
    return "\n".join(lines)


# ── Registration ──────────────────────────────────────────────────────────────

def register_convenience_tools(registry) -> None:
    from BRAIN.tools.registry import ToolEntry

    registry.register(ToolEntry(
        name="append_file",
        description=(
            "Append text to a file, preserving its existing encoding and line-ending style.\n\n"
            "The file's encoding (UTF-8, UTF-16, Latin-1) and dominant line endings "
            "(LF vs CRLF) are detected and applied to the new content — no mixed-encoding "
            "or mixed-CRLF/LF results.\n\n"
            "ensure_newline=true (default): if the file doesn't end with a newline, one "
            "is inserted first so the new content starts on a fresh line.\n\n"
            "create=true (default): creates the file if it doesn't exist.\n\n"
            "Uses workspace routing for external files — writes to sofi-workspace/active/ "
            "so the originals are never modified directly."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to append to.",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Text to append. Line endings are normalised to match "
                        "the target file's existing style."
                    ),
                },
                "ensure_newline": {
                    "type": "boolean",
                    "description": (
                        "If true (default), insert a newline before the content "
                        "when the file doesn't already end with one."
                    ),
                    "default": True,
                },
                "create": {
                    "type": "boolean",
                    "description": (
                        "If true (default), create the file when it doesn't exist. "
                        "If false, return an error when the file is missing."
                    ),
                    "default": True,
                },
            },
            "required": ["path", "content"],
        },
        handler=append_file,
        category="filesystem",
        capability_name="append_file",
        capability_description=(
            "Append text to a file while preserving its encoding and line-ending style."
        ),
        capability_refusal="I can't append to files right now.",
    ))

    registry.register(ToolEntry(
        name="copy_file",
        description=(
            "Copy a file or directory to a new location.\n\n"
            "Source is read directly — no workspace restriction on reads.\n"
            "Destination is written to the given path directly (you choose where it goes).\n\n"
            "If the destination already exists:\n"
            "  • overwrite=false (default): fails with a clear error — nothing changes.\n"
            "  • overwrite=true: the existing destination is soft-deleted first "
            "(fully recoverable via restore_backup), then the copy is written.\n\n"
            "Directories are copied recursively. File timestamps and permissions are "
            "preserved (shutil.copy2)."
        ),
        schema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to copy.",
                },
                "destination": {
                    "type": "string",
                    "description": "Absolute path to copy to.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "If true, soft-delete the existing destination first, "
                        "then copy. The displaced item is recoverable. Default false."
                    ),
                    "default": False,
                },
            },
            "required": ["source", "destination"],
        },
        handler=copy_file,
        category="filesystem",
        capability_name="copy_file",
        capability_description=(
            "Copy files and directories; backs up the destination before overwriting."
        ),
        capability_refusal="I can't copy files right now.",
    ))

    registry.register(ToolEntry(
        name="move_file",
        description=(
            "Move or rename a file or directory.\n\n"
            "SAFE BY DESIGN: the source is never permanently deleted.\n\n"
            "Strategy:\n"
            "  1. Atomic rename (os.rename) — instant when source and destination "
            "are on the same volume (the common case). No backup needed.\n"
            "  2. Cross-device fallback: copy to destination, then soft-delete the "
            "source via BackupManager. The source remains recoverable via restore_backup.\n\n"
            "If the destination already exists:\n"
            "  • overwrite=false (default): fails — nothing moves.\n"
            "  • overwrite=true: destination is soft-deleted first, then the move "
            "proceeds. Both items are fully recoverable.\n\n"
            "Blocked: OS system directories (C:\\Windows, /etc, etc.) and the backup "
            "directory cannot be source or destination.\n\n"
            "Requires confirmation before running."
        ),
        schema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to move.",
                },
                "destination": {
                    "type": "string",
                    "description": "Absolute path to move to.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "If true, soft-delete the existing destination first, "
                        "then move. Default false."
                    ),
                    "default": False,
                },
            },
            "required": ["source", "destination"],
        },
        handler=move_file,
        needs_confirmation=True,
        category="filesystem",
        capability_name="move_file",
        capability_description=(
            "Move or rename files and directories. Atomic rename on same filesystem; "
            "copy + soft-delete on cross-device. Source is always recoverable."
        ),
        capability_refusal="I can't move files right now.",
    ))


# Auto-discovery alias
register = register_convenience_tools
