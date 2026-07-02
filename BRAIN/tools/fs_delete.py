"""
BRAIN/tools/fs_delete.py — Safe file and directory deletion for SOFi

All deletes route through BackupManager — nothing is ever permanently removed.
Each delete creates a timestamped backup slot and returns a backup_id for
full recovery via restore_backup.

Tools:
  delete_file     Soft-delete a file or directory (always recoverable)
  restore_backup  Restore a soft-deleted item to its original path
  list_backups    List recent backup slots with metadata
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

_log = logging.getLogger("sofi.brain.tools.fs_delete")

# OS system directories — always blocked regardless of any policy.
_BLOCKED_DELETE_PREFIXES: tuple = (
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/boot", "/proc", "/sys", "/dev",
    "/System", "/Library", "/Applications",
)


# ── Safety helpers ────────────────────────────────────────────────────────────

def _check_delete_blocked(p: Path) -> Tuple[bool, str]:
    """
    Return (blocked, reason).

    Blocks:
      • OS system directories (catastrophic loss)
      • The backup directory or any path inside it (would corrupt the index)
      • The workspace root itself
    """
    s = str(p).replace("\\", "/")
    for prefix in _BLOCKED_DELETE_PREFIXES:
        if s.lower().startswith(prefix.lower().replace("\\", "/")):
            return True, f"Deleting OS system paths is blocked: {p}"

    try:
        from BRAIN.tools.workspace import get_manager as _gwm
        ws  = _gwm()
        rp  = p.resolve()
        br  = ws.backup_root.resolve()
        wr  = ws.root.resolve()

        if rp == wr:
            return True, "Deleting the workspace root is not allowed."

        if rp == br or br in rp.parents:
            return True, (
                "Deleting inside the backup directory is not allowed — it would "
                "corrupt the backup index. Use list_backups to view slots and "
                "purge to permanently remove individual ones."
            )
    except Exception:
        pass

    return False, ""


def _fmt_ts(iso_str: str) -> str:
    """Parse ISO timestamp → compact human-readable form."""
    try:
        return datetime.fromisoformat(iso_str).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_str[:16] if len(iso_str) >= 16 else iso_str


# ═══════════════════════════════════════════════════════════════════════════
# delete_file
# ═══════════════════════════════════════════════════════════════════════════

async def delete_file(path: str, reason: str = "") -> str:
    """
    Safely delete a file or directory by moving it to a backup slot.

    Never permanently removes anything.  The item is copied to
    sofi-workspace/backup/ and a backup_id is returned for recovery.
    Call restore_backup(backup_id) to undo.

    Works on any path — internal workspace files and external filesystem
    paths both go through BackupManager.
    """
    try:
        return await asyncio.to_thread(_delete_file_impl, path, reason)
    except Exception as exc:
        _log.exception("delete_file | path=%s", path)
        return f"Error deleting: {exc}"


def _delete_file_impl(path: str, reason: str) -> str:
    p = Path(path).expanduser().resolve()

    blocked, block_reason = _check_delete_blocked(p)
    if blocked:
        return f"Blocked: {block_reason}"

    if not p.exists():
        return f"Not found: {p}\nNothing was deleted."

    is_dir    = p.is_dir()
    item_type = "directory" if is_dir else "file"

    try:
        from BRAIN.tools.workspace import get_backup_manager
        backup_id = get_backup_manager().soft_delete(p)
    except FileNotFoundError:
        return f"Not found: {p}\nNothing was deleted."
    except PermissionError as exc:
        return f"Permission denied: {exc}"
    except RuntimeError as exc:
        return f"Backup verification failed — original untouched: {exc}"
    except Exception as exc:
        return f"Delete failed — original untouched: {exc}"

    lines = [
        f"Deleted: {p}",
        f"  type:      {item_type}",
        f"  backup_id: {backup_id}",
    ]
    if reason:
        lines.append(f"  reason:    {reason}")
    lines.append(f"\nTo restore:  restore_backup(backup_id='{backup_id}')")

    _log.info("delete_file | %s | type=%s | backup_id=%s", p, item_type, backup_id)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# restore_backup
# ═══════════════════════════════════════════════════════════════════════════

async def restore_backup(backup_id: str, overwrite: bool = False) -> str:
    """
    Restore a soft-deleted file or directory to its exact original path.

    If the original path is already occupied:
      • overwrite=False (default): fails with a clear error — nothing changes.
      • overwrite=True:  the existing item is soft-deleted first (creating a
        new backup_id that is reported in the result), then the original is
        restored.  Both items remain fully recoverable.

    The backup slot is preserved after restore.  Use list_backups to see
    all slots; the BackupManager's purge() method cleans them up permanently.
    """
    backup_id = backup_id.strip()
    if not backup_id:
        return "Error: backup_id cannot be empty. Use list_backups to find valid IDs."
    try:
        return await asyncio.to_thread(_restore_backup_impl, backup_id, overwrite)
    except Exception as exc:
        _log.exception("restore_backup | backup_id=%s", backup_id)
        return f"Error restoring backup: {exc}"


def _restore_backup_impl(backup_id: str, overwrite: bool) -> str:
    from BRAIN.tools.workspace import get_backup_manager
    bm    = get_backup_manager()
    entry = bm.get(backup_id)

    if entry is None:
        recent  = bm.list_backups(limit=5)
        hint    = ""
        if recent:
            ids  = ", ".join(e["backup_id"] for e in recent[:3])
            hint = f"\nMost recent backup IDs: {ids}"
        return (
            f"Backup '{backup_id}' not found.{hint}\n"
            "Use list_backups to see all available backups."
        )

    original        = Path(entry["original_path"])
    displacement_id: Optional[str] = None

    # If the target is occupied and the caller opted in, move it aside first.
    if overwrite and original.exists():
        try:
            displacement_id = bm.soft_delete(original)
            _log.info(
                "restore_backup | displaced existing item at %s | new_backup_id=%s",
                original, displacement_id,
            )
        except Exception as exc:
            return (
                f"Could not displace the existing item at {original}: {exc}\n"
                "Move or rename it manually, then retry."
            )

    try:
        restored = bm.restore(backup_id)
    except KeyError:
        return f"Backup '{backup_id}' not found in the index."
    except FileExistsError:
        return (
            f"Target path is already occupied: {original}\n"
            "Pass overwrite=true to automatically displace the existing item "
            "(it will be backed up), or move it manually first."
        )
    except FileNotFoundError as exc:
        return (
            f"Backup files are missing for '{backup_id}' — the slot may have been purged.\n"
            f"Detail: {exc}"
        )

    item_type   = "directory" if entry.get("is_dir") else "file"
    backed_up_at = _fmt_ts(entry.get("backed_up_at", ""))

    lines = [
        f"Restored: {restored}",
        f"  type:      {item_type}",
        f"  backup_id: {backup_id}",
        f"  backed up: {backed_up_at}",
    ]
    if displacement_id:
        lines.append(
            f"\n  Displaced item backed up as: {displacement_id}"
        )

    _log.info("restore_backup | id=%s → %s", backup_id, restored)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# list_backups
# ═══════════════════════════════════════════════════════════════════════════

async def list_backups(limit: int = 50) -> str:
    """Return a formatted list of backup slots, newest first."""
    try:
        return await asyncio.to_thread(_list_backups_impl, limit)
    except Exception as exc:
        _log.exception("list_backups")
        return f"Error listing backups: {exc}"


def _list_backups_impl(limit: int) -> str:
    limit = max(1, min(int(limit), 200))

    from BRAIN.tools.workspace import get_backup_manager
    entries = get_backup_manager().list_backups(limit=limit)

    if not entries:
        return "No backups found. Nothing has been soft-deleted yet."

    lines = [f"Backup history ({len(entries)} item(s), newest first):\n"]
    for entry in entries:
        bid  = entry.get("backup_id", "?")
        kind = "dir " if entry.get("is_dir") else "file"
        orig = entry.get("original_path", "?")
        when = _fmt_ts(entry.get("backed_up_at", ""))

        # Truncate very long paths for readability (unicode ellipsis at front).
        if len(orig) > 60:
            orig = "…" + orig[-59:]

        lines.append(f"  {bid}  {kind}  {orig:<60}  {when}")

    if len(entries) == limit:
        lines.append(f"\n(showing {limit} most recent — pass a larger limit for more)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

def register_delete_tools(registry) -> None:
    from BRAIN.tools.registry import ToolEntry

    # ── delete_file ────────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="delete_file",
        description=(
            "Safely delete a file or directory by moving it to a backup slot.\n\n"
            "SAFE BY DESIGN: nothing is ever permanently removed. The item is copied "
            "to sofi-workspace/backup/ and a backup_id is returned. "
            "Use restore_backup(backup_id) to fully undo.\n\n"
            "Works on any path — internal workspace files and external files both "
            "go through BackupManager.\n\n"
            "BLOCKED (always):\n"
            "  • OS system directories (C:\\Windows, /etc, /usr, etc.)\n"
            "  • The backup directory itself (would corrupt the index)\n"
            "  • The workspace root itself\n\n"
            "Requires confirmation before running."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the file or directory to delete. "
                        "The item is moved to a backup slot — nothing is permanently removed."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Optional note about why this item is being deleted (logged in output).",
                    "default": "",
                },
            },
            "required": ["path"],
        },
        handler=delete_file,
        needs_confirmation=True,
        category="filesystem",
        capability_name="delete_file",
        capability_description=(
            "Safely delete files and directories — everything is backed up and "
            "fully recoverable with restore_backup."
        ),
        capability_refusal="I can't delete files right now.",
    ))

    # ── restore_backup ─────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="restore_backup",
        description=(
            "Restore a soft-deleted file or directory to its exact original path.\n\n"
            "Paired with delete_file — use the backup_id returned by delete_file. "
            "Use list_backups to browse IDs for older deletions.\n\n"
            "If the original path is already occupied:\n"
            "  • overwrite=false (default): fails with a clear error — nothing changes.\n"
            "  • overwrite=true: the existing item is soft-deleted first "
            "(a new backup_id is reported in the result), then the original is "
            "restored. Both items remain fully recoverable.\n\n"
            "The backup slot is preserved after restore. Use list_backups to view "
            "slots; they can be purged permanently when no longer needed.\n\n"
            "Requires confirmation before running."
        ),
        schema={
            "type": "object",
            "properties": {
                "backup_id": {
                    "type": "string",
                    "description": (
                        "The backup ID to restore. Returned by delete_file, "
                        "or visible in list_backups. "
                        "Format: YYYY-MM-DD_HHMMSS_<8-char uuid>."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "If true and the target path is occupied, soft-delete the "
                        "existing item first, then restore the backup. "
                        "Both items remain recoverable. Default false."
                    ),
                    "default": False,
                },
            },
            "required": ["backup_id"],
        },
        handler=restore_backup,
        needs_confirmation=True,
        category="filesystem",
        capability_name="restore_backup",
        capability_description=(
            "Restore a soft-deleted file or directory to its original path. "
            "Paired with delete_file."
        ),
        capability_refusal="I can't restore backups right now.",
    ))

    # ── list_backups ───────────────────────────────────────────────────────
    registry.register(ToolEntry(
        name="list_backups",
        description=(
            "List soft-deleted items in the backup store, newest first.\n\n"
            "Shows: backup_id, type (file/dir), original path, and deletion time.\n"
            "Use the backup_id with restore_backup to recover an item.\n\n"
            "Read-only — no confirmation needed."
        ),
        schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum records to return (1–200, default 50).",
                    "default": 50,
                },
            },
            "required": [],
        },
        handler=list_backups,
        category="filesystem",
        capability_name="list_backups",
        capability_description=(
            "List all soft-deleted files and directories in the backup store "
            "with their IDs and original paths."
        ),
        capability_refusal="I can't list backups right now.",
    ))


# Auto-discovery alias — brain.py looks for register(registry)
register = register_delete_tools
