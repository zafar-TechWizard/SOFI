"""
BRAIN/tools/workspace/__init__.py

Lazy-initialized singletons for WorkspaceManager and BackupManager.

All file tools import from here — they never instantiate their own copies.
The first call to get_manager() or get_backup_manager() triggers initialization.
brain.py may call initialize() explicitly during setup to eagerly create the
workspace and surface any boot errors early.
"""

import threading
from pathlib import Path
from typing import Optional

from BRAIN.tools.workspace.manager import WorkspaceManager
from BRAIN.tools.workspace.backup import BackupManager

__all__ = ["WorkspaceManager", "BackupManager", "initialize", "get_manager", "get_backup_manager"]

_manager:     Optional[WorkspaceManager] = None
_backup_mgr:  Optional[BackupManager]   = None
_init_lock:   threading.Lock            = threading.Lock()


def initialize(workspace_root: Optional[Path] = None) -> tuple:
    """
    Explicitly initialize the workspace singletons.

    Safe to call multiple times — re-initialization is a no-op if already done.
    Returns (WorkspaceManager, BackupManager).
    """
    global _manager, _backup_mgr
    with _init_lock:
        if _manager is None:
            _manager    = WorkspaceManager(workspace_root)
            _backup_mgr = BackupManager(_manager.backup_root)
    return _manager, _backup_mgr


def get_manager() -> WorkspaceManager:
    """Return the WorkspaceManager singleton, initializing lazily if needed."""
    if _manager is None:
        initialize()
    return _manager  # type: ignore[return-value]


def get_backup_manager() -> BackupManager:
    """Return the BackupManager singleton, initializing lazily if needed."""
    if _backup_mgr is None:
        initialize()
    return _backup_mgr  # type: ignore[return-value]
