"""
BRAIN/tools/workspace/manager.py — WorkspaceManager

Single source of truth for all path-policy decisions in the SOFi tool layer.

Policy
------
  Internal  (inside workspace root)   → all operations allowed directly
  External  (outside workspace root)
    ├── reads                         → always allowed, read-only carries no risk
    ├── writes / creates              → copy to active/ first; work on the copy
    └── deletes                       → always intercepted by BackupManager

Configuration
-------------
  SOFI_WORKSPACE env var → workspace root directory
  Default: <assistant_root>/sofi-workspace/

Thread safety
-------------
  All mutable state is set exactly once in __init__ and is never written again.
  No locking needed — every public method is purely computational after init.
"""

import logging
import os
from pathlib import Path
from typing import Optional

_log = logging.getLogger("sofi.brain.tools.workspace")

# Two levels up from this file:
#   BRAIN/tools/workspace/manager.py
#   → BRAIN/tools/workspace/
#   → BRAIN/tools/
#   → BRAIN/
#   → assistant/   ← project root
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_WORKSPACE: Path = _PROJECT_ROOT / "sofi-workspace"


class WorkspaceManager:
    """
    Enforces the workspace boundary policy for all SOFi file tool operations.

    Instantiate once at brain startup (or rely on the lazy singleton in
    BRAIN/tools/workspace/__init__.py). All tools share one instance.

    This class is intentionally stateless after __init__ — every public
    method is a pure computation on immutable fields, which means it is
    safe to call from any thread without locking.
    """

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        """
        Args:
            workspace_root: explicit override; if None, reads SOFI_WORKSPACE
                            env var, then falls back to the project default.
        """
        env_val = os.environ.get("SOFI_WORKSPACE", "").strip()

        if workspace_root is not None:
            _root = Path(workspace_root)
        elif env_val:
            _root = Path(env_val)
        else:
            _root = _DEFAULT_WORKSPACE

        self._root: Path = _root.resolve()
        self.ensure_dirs()

        _log.info("WorkspaceManager ready | root=%s", self._root)

    # ── Directory properties ──────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Workspace root directory."""
        return self._root

    @property
    def active_root(self) -> Path:
        """Working copies of external files live here."""
        return self._root / "active"

    @property
    def backup_root(self) -> Path:
        """Soft-deleted items land here (managed by BackupManager)."""
        return self._root / "backup"

    @property
    def scratch_root(self) -> Path:
        """Throwaway temporary files."""
        return self._root / "scratch"

    @property
    def git_root(self) -> Path:
        """Git-backed project copies for code changes."""
        return self._root / "git"

    # ── Policy checks ─────────────────────────────────────────────────────────

    def is_internal(self, path) -> bool:
        """
        Return True if *path* is inside the workspace root.

        Resolves symlinks and normalises before comparing so that paths like
        ``../../sofi-workspace/foo`` are correctly identified as internal.
        Always returns False for paths that cannot be resolved (e.g. the
        path contains characters invalid on this OS).
        """
        try:
            resolved = Path(path).resolve()
            # A path is internal if it IS the root or has the root as a parent
            return resolved == self._root or self._root in resolved.parents
        except (OSError, ValueError):
            return False

    def to_active_copy(self, external_path) -> Path:
        """
        Map an external (outside-workspace) path to its mirror under active/.

        The full original directory structure is preserved so the mapping is
        always reversible.  Drive letters on Windows are folded into the first
        path component (colon stripped):

            C:\\Users\\mdzaf\\foo.py  →  <workspace>/active/C/Users/mdzaf/foo.py
            /home/mdzaf/foo.py       →  <workspace>/active/home/mdzaf/foo.py

        Args:
            external_path: any path-like object; resolved before mapping.

        Returns:
            Absolute Path inside active_root — does not touch the filesystem.
        """
        resolved = Path(external_path).resolve()
        drive, tail = os.path.splitdrive(str(resolved))

        # Normalise drive:  "C:" → "C",  "" (Unix) → "root"
        drive_label = drive.replace(":", "").strip("/\\") or "root"

        # Strip leading separator(s) from the rest of the path, then
        # use Path() to normalise OS-specific separators (backslash on Windows)
        tail_path = Path(tail.lstrip("/\\"))

        return self.active_root / drive_label / tail_path

    def resolve_safe(self, path) -> Path:
        """
        Resolve *path* to an absolute, normalised form.

        This is a thin wrapper around Path.resolve() provided as a single
        place to add future path-safety checks (length limits, forbidden
        characters, etc.) without touching every tool.
        """
        return Path(path).resolve()

    # ── Filesystem setup ──────────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """
        Create all workspace subdirectories if they do not exist.

        Called automatically in __init__; can be called again safely
        (all mkdir calls use exist_ok=True).
        """
        for subdir in (
            self.active_root,
            self.backup_root,
            self.scratch_root,
            self.git_root,
        ):
            try:
                subdir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                _log.error("ensure_dirs failed for %s | %s", subdir, exc)
                raise

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return a status snapshot suitable for brain.inspect() and /status.

        Never raises — disk errors are caught and reported in the dict.
        """
        result: dict = {
            "root":   str(self._root),
            "exists": self._root.exists(),
        }
        try:
            import shutil
            _, _, free = shutil.disk_usage(self._root)
            result["disk_free_mb"] = round(free / 1_048_576, 1)
        except Exception:
            result["disk_free_mb"] = "unknown"
        return result
