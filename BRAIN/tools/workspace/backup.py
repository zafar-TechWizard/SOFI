"""
BRAIN/tools/workspace/backup.py — BackupManager

Soft-delete implementation for the SOFi tool layer.

Every delete operation anywhere in the system passes through BackupManager.
Nothing is ever permanently removed — files and directories are moved to a
timestamped backup slot that mirrors the original path structure exactly,
making full recovery always possible.

Structure
---------
  <workspace>/backup/
    index.json                     ← maps backup_id → metadata
    <backup_id>/                   ← one slot per delete
      <drive>/<original path>      ← full path preserved

Recovery
--------
  backup_mgr.restore(backup_id)   → puts files/dirs back at exact original path
  backup_mgr.list_backups()       → shows all slots with metadata
  backup_mgr.purge(backup_id)     → permanently removes a slot (irreversible)

Thread safety
-------------
  All public methods are protected by a single RLock.
  Index reads and writes are always done with the lock held.
  File I/O (copy, move, rmtree) is done outside the lock to avoid
  long-held critical sections, but the index is updated atomically
  only after the file operation completes successfully.
"""

import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("sofi.brain.tools.workspace.backup")

_INDEX_FILENAME = "index.json"


class BackupManager:
    """
    Intercepts all delete operations and preserves files/directories in a
    timestamped, path-mirroring backup structure.

    Instantiate once per workspace (the singleton in __init__.py handles this).
    """

    def __init__(self, backup_root: Path) -> None:
        """
        Args:
            backup_root: the backup/ subdirectory of the SOFi workspace.
                         Created if it does not exist.
        """
        self._backup_root: Path = backup_root
        self._backup_root.mkdir(parents=True, exist_ok=True)
        self._index_path: Path  = self._backup_root / _INDEX_FILENAME
        self._lock: threading.RLock = threading.RLock()
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

        _log.info(
            "BackupManager ready | root=%s | %d existing backup(s)",
            backup_root,
            len(self._index),
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def soft_delete(self, path) -> str:
        """
        Safely "delete" a file or directory by moving it to a backup slot.

        The operation is copy-then-verify-then-remove:
          1. Copy source to the backup slot  (if this fails, source is untouched)
          2. Verify the backup copy exists   (if not, raise and clean up partial)
          3. Remove the original             (only after backup is confirmed good)

        This order guarantees that a power loss or crash between steps 2 and 3
        leaves both copies intact rather than losing the file.

        Args:
            path: file or directory to soft-delete (any path-like object).

        Returns:
            backup_id — pass to restore() to recover.

        Raises:
            FileNotFoundError: if *path* does not exist.
            RuntimeError:      if backup copy cannot be verified.
            OSError:           for other I/O failures (original left untouched).
        """
        src = Path(path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"soft_delete: path not found: {src}")

        backup_id  = _make_backup_id()
        slot_root  = self._backup_root / backup_id
        dest       = _slot_path(slot_root, src)
        is_dir     = src.is_dir()

        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            if is_dir:
                shutil.copytree(str(src), str(dest))
                if not dest.exists():
                    raise RuntimeError(f"copytree incomplete: {dest}")
                shutil.rmtree(str(src))
            else:
                shutil.copy2(str(src), str(dest))
                if not dest.exists():
                    raise RuntimeError(f"copy2 incomplete: {dest}")
                src.unlink()

        except Exception:
            # Roll back partial backup slot so we don't leave debris
            try:
                if slot_root.exists():
                    shutil.rmtree(str(slot_root), ignore_errors=True)
            except Exception:
                pass
            raise

        # Record in index only after the file operation fully succeeded
        entry: Dict[str, Any] = {
            "backup_id":     backup_id,
            "original_path": str(src),
            "is_dir":        is_dir,
            "backed_up_at":  datetime.now().isoformat(),
            "slot_root":     str(slot_root),
        }
        with self._lock:
            self._index[backup_id] = entry
            self._save_index(self._index)

        _log.info("soft_delete | id=%s | %s → %s", backup_id, src, dest)
        return backup_id

    def restore(self, backup_id: str) -> Path:
        """
        Restore a backup to its exact original path.

        The restore is also copy-then-verify: the backup copy is preserved in
        its slot until the caller decides to purge it.

        Args:
            backup_id: the string returned by soft_delete().

        Returns:
            The restored path (same as original_path in the index).

        Raises:
            KeyError:       if backup_id is not in the index.
            FileExistsError: if the original path is already occupied.
            FileNotFoundError: if the backup slot files are missing.
            OSError:        for other I/O failures.
        """
        with self._lock:
            entry = self._index.get(backup_id)

        if entry is None:
            raise KeyError(f"restore: backup_id not found: {backup_id!r}")

        original  = Path(entry["original_path"])
        slot_root = Path(entry["slot_root"])
        backup_src = _slot_path(slot_root, original)
        is_dir    = entry["is_dir"]

        if not backup_src.exists():
            raise FileNotFoundError(
                f"restore: backup files missing at {backup_src} "
                f"(slot may have been purged)"
            )

        if original.exists():
            raise FileExistsError(
                f"restore: target path already exists: {original}\n"
                "Move or rename the existing file before restoring."
            )

        original.parent.mkdir(parents=True, exist_ok=True)

        if is_dir:
            shutil.copytree(str(backup_src), str(original))
        else:
            shutil.copy2(str(backup_src), str(original))

        _log.info("restore | id=%s → %s", backup_id, original)
        return original

    def list_backups(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return backup metadata records, newest first.

        Each record contains:
            backup_id, original_path, is_dir, backed_up_at, slot_root

        Args:
            limit: maximum number of records to return.
        """
        with self._lock:
            entries = list(self._index.values())

        entries.sort(key=lambda e: e.get("backed_up_at", ""), reverse=True)
        return entries[:limit]

    def purge(self, backup_id: str) -> bool:
        """
        Permanently delete a backup slot and remove its index entry.

        This is irreversible. Use only when certain the backup is no
        longer needed (e.g. scheduled cleanup of very old backups).

        Args:
            backup_id: the slot to delete.

        Returns:
            True if the slot was found and purged, False if not found.
        """
        with self._lock:
            entry = self._index.pop(backup_id, None)
            if entry is None:
                return False
            self._save_index(self._index)

        slot_root = Path(entry["slot_root"])
        if slot_root.exists():
            shutil.rmtree(str(slot_root), ignore_errors=True)

        _log.info("purge | id=%s | original=%s", backup_id, entry.get("original_path"))
        return True

    def get(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Return the index entry for a backup_id, or None if not found."""
        with self._lock:
            return self._index.get(backup_id)

    # ── Index persistence ─────────────────────────────────────────────────────

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the backup index from disk.

        Returns an empty dict on any failure (corrupt JSON, missing file, etc.)
        so that startup is never blocked by index issues.  A warning is logged
        so the problem is visible without crashing.
        """
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"index root must be a dict, got {type(data).__name__}")
            return data
        except Exception as exc:
            _log.warning(
                "backup index load failed — starting fresh | path=%s | error=%s",
                self._index_path, exc,
            )
            return {}

    def _save_index(self, index: Dict) -> None:
        """
        Write the backup index to disk atomically (tmp + rename).

        Called with self._lock held.  A write failure is logged as an error
        but does not raise — the in-memory index is still correct and will
        be saved on the next successful mutation.
        """
        try:
            tmp = self._index_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=2, ensure_ascii=False)
            tmp.replace(self._index_path)   # atomic on same filesystem
        except Exception as exc:
            _log.error(
                "backup index save failed | path=%s | error=%s",
                self._index_path, exc,
            )


# ── Module-level helpers (shared with manager.py path mirror logic) ───────────

def _make_backup_id() -> str:
    """Generate a unique backup slot name: YYYY-MM-DD_HHMMSS_<8-char uuid>."""
    ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    return f"{ts}_{uid}"


def _slot_path(slot_root: Path, original: Path) -> Path:
    """
    Mirror *original* inside *slot_root*, preserving the full path structure.

    Windows drive letters are folded into the first component (colon stripped):
        C:\\Users\\mdzaf\\foo.py → slot_root/C/Users/mdzaf/foo.py
        /home/mdzaf/foo.py      → slot_root/root/home/mdzaf/foo.py  (no drive)
    """
    drive, tail = os.path.splitdrive(str(original))
    drive_label = drive.replace(":", "").strip("/\\") or "root"
    tail_path   = Path(tail.lstrip("/\\"))
    return slot_root / drive_label / tail_path
