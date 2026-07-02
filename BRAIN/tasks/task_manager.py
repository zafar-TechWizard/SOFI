"""
BRAIN/tasks/task_manager.py — Disk-backed task lifecycle manager.

Every delegated task gets a JSON file on disk. Sub-agents update the file
as they work. Main SOFi reads completed deliveries from disk when building
the prompt.

Lifecycle:  pending → in_progress → verifying → completed → delivered
            pending → in_progress → failed

File location: BRAIN/memory/data/tasks/task_{id}_{timestamp}.json
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("sofi.brain.tasks")

_TASKS_DIR = Path(__file__).parent.parent / "memory" / "data" / "tasks"

# Cleanup thresholds
_MAX_TASK_FILES = 50
_DELIVERED_MAX_AGE_DAYS = 7
_FAILED_MAX_AGE_DAYS = 3
_UNDELIVERED_MAX_AGE_DAYS = 2


@dataclass
class TaskStep:
    step: int
    action: str
    status: str = "pending"  # pending | in_progress | done | failed
    detail: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class TaskDelivery:
    status: str = "pending"  # fulfilled | partial | failed
    summary: str = ""
    content: str = ""
    gaps: Optional[str] = None


@dataclass
class TaskFile:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    original_query: str = ""
    brief: str = ""
    agent_type: str = ""
    status: str = "pending"  # pending | in_progress | verifying | completed | delivered | failed
    steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    delivery: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskFile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def _file_path(self) -> Path:
        ts = self.created_at.replace(":", "").replace("-", "").replace("T", "_")[:15]
        return _TASKS_DIR / f"task_{self.task_id}_{ts}.json"


class TaskManager:
    """
    Thread-safe, disk-backed task manager.

    Every mutation flushes to disk immediately. Reads are from disk.
    File locking uses a per-task threading.Lock for in-process safety.
    """

    def __init__(self) -> None:
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        _TASKS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_lock(self, task_id: str) -> threading.Lock:
        with self._global_lock:
            if task_id not in self._locks:
                self._locks[task_id] = threading.Lock()
            return self._locks[task_id]

    # ── Create ───────────────────────────────────────────────────────────

    def create_task(
        self,
        original_query: str,
        brief: str,
        agent_type: str,
    ) -> TaskFile:
        task = TaskFile(
            original_query=original_query,
            brief=brief,
            agent_type=agent_type,
            status="pending",
        )
        self._write(task)
        _log.info(
            "task_create | id=%s agent=%s query_preview=%.60s",
            task.task_id, agent_type, original_query,
        )
        return task

    # ── Update ───────────────────────────────────────────────────────────

    def update_status(self, task_id: str, status: str) -> None:
        lock = self._get_lock(task_id)
        with lock:
            task = self._read(task_id)
            if task is None:
                _log.warning("update_status | task not found | id=%s", task_id)
                return
            task.status = status
            task.updated_at = datetime.now().isoformat(timespec="seconds")
            self._write(task)
        _log.debug("task_update_status | id=%s status=%s", task_id, status)

    def set_steps(self, task_id: str, steps: List[Dict[str, Any]]) -> None:
        lock = self._get_lock(task_id)
        with lock:
            task = self._read(task_id)
            if task is None:
                return
            task.steps = steps
            task.total_steps = len(steps)
            task.current_step = 0
            task.status = "in_progress"
            task.updated_at = datetime.now().isoformat(timespec="seconds")
            self._write(task)
        _log.debug("task_set_steps | id=%s steps=%d", task_id, len(steps))

    def update_step(
        self,
        task_id: str,
        step_index: int,
        status: str,
        detail: str = "",
    ) -> None:
        lock = self._get_lock(task_id)
        with lock:
            task = self._read(task_id)
            if task is None:
                return
            if step_index < len(task.steps):
                task.steps[step_index]["status"] = status
                if detail:
                    task.steps[step_index]["detail"] = detail
                if status == "in_progress":
                    task.steps[step_index]["started_at"] = datetime.now().isoformat(timespec="seconds")
                elif status in ("done", "failed"):
                    task.steps[step_index]["completed_at"] = datetime.now().isoformat(timespec="seconds")
                task.current_step = step_index
            task.updated_at = datetime.now().isoformat(timespec="seconds")
            self._write(task)

    def write_delivery(
        self,
        task_id: str,
        status: str,
        summary: str,
        content: str,
        gaps: Optional[str] = None,
    ) -> None:
        lock = self._get_lock(task_id)
        with lock:
            task = self._read(task_id)
            if task is None:
                return
            task.delivery = {
                "status": status,
                "summary": summary,
                "content": content,
                "gaps": gaps,
            }
            task.status = "completed"
            task.updated_at = datetime.now().isoformat(timespec="seconds")
            self._write(task)
        _log.info(
            "task_delivery | id=%s delivery_status=%s summary_len=%d content_len=%d",
            task_id, status, len(summary), len(content),
        )

    def mark_delivered(self, task_id: str) -> None:
        self.update_status(task_id, "delivered")

    def update_metadata(self, task_id: str, data: Dict[str, Any]) -> None:
        lock = self._get_lock(task_id)
        with lock:
            task = self._read(task_id)
            if task is None:
                return
            if task.metadata is None:
                task.metadata = {}
            task.metadata.update(data)
            self._write(task)

    def mark_failed(self, task_id: str, reason: str = "") -> None:
        lock = self._get_lock(task_id)
        with lock:
            task = self._read(task_id)
            if task is None:
                return
            task.status = "failed"
            task.delivery = {
                "status": "failed",
                "summary": reason or "Task failed",
                "content": "",
                "gaps": None,
            }
            task.updated_at = datetime.now().isoformat(timespec="seconds")
            self._write(task)
        _log.warning("task_failed | id=%s reason=%s", task_id, reason)

    # ── Read ─────────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Optional[TaskFile]:
        return self._read(task_id)

    def get_active_tasks(self) -> List[TaskFile]:
        tasks = self._read_all()
        return [t for t in tasks if t.status in ("pending", "in_progress", "verifying")]

    def get_completed_undelivered(self) -> List[TaskFile]:
        tasks = self._read_all()
        return [t for t in tasks if t.status == "completed"]

    def get_recently_delivered(self, max_age_minutes: int = 30) -> List[TaskFile]:
        """Return tasks marked 'delivered' within the last N minutes."""
        tasks = self._read_all()
        cutoff = time.time() - (max_age_minutes * 60)
        recent = []
        for t in tasks:
            if t.status != "delivered":
                continue
            try:
                updated = datetime.fromisoformat(t.updated_at).timestamp()
                if updated >= cutoff:
                    recent.append(t)
            except (ValueError, OSError):
                pass
        return recent

    def get_all_tasks(self) -> List[TaskFile]:
        return self._read_all()

    def get_task_summary_for_prompt(self) -> Dict[str, Any]:
        """
        Build a compact summary of all tasks for injection into the prompt.
        Active tasks show step-level progress. Completed tasks show delivery.
        """
        active = self.get_active_tasks()
        completed = self.get_completed_undelivered()

        summary: Dict[str, Any] = {}

        if active:
            summary["active"] = []
            for t in active[:5]:
                entry: Dict[str, Any] = {
                    "task_id": t.task_id,
                    "agent": t.agent_type,
                    "query": t.original_query[:120],
                    "status": t.status,
                }
                if t.total_steps > 0:
                    entry["progress"] = f"step {t.current_step + 1}/{t.total_steps}"
                    current = t.steps[t.current_step] if t.current_step < len(t.steps) else None
                    if current:
                        entry["current_action"] = current.get("action", "")
                        entry["detail"] = current.get("detail", "")
                summary["active"].append(entry)

        if completed:
            summary["ready_to_deliver"] = []
            for t in completed[:5]:
                delivery = t.delivery or {}
                summary["ready_to_deliver"].append({
                    "task_id": t.task_id,
                    "agent": t.agent_type,
                    "original_query": t.original_query,
                    "delivery_status": delivery.get("status", "unknown"),
                    "summary": delivery.get("summary", ""),
                    "content": delivery.get("content", ""),
                    "gaps": delivery.get("gaps"),
                })

        return summary

    # ── Cleanup ──────────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """
        Remove old task files. Returns count of files removed.
        - delivered: remove after 7 days
        - failed: remove after 3 days
        - completed but never delivered: remove after 2 days
        - hard cap: 50 files max (oldest delivered pruned first)
        """
        removed = 0
        now = time.time()

        files = sorted(_TASKS_DIR.glob("task_*.json"), key=lambda f: f.stat().st_mtime)

        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                status = data.get("status", "")
                age_days = (now - f.stat().st_mtime) / 86400

                should_remove = False
                if status == "delivered" and age_days > _DELIVERED_MAX_AGE_DAYS:
                    should_remove = True
                elif status == "failed" and age_days > _FAILED_MAX_AGE_DAYS:
                    should_remove = True
                elif status == "completed" and age_days > _UNDELIVERED_MAX_AGE_DAYS:
                    should_remove = True

                if should_remove:
                    f.unlink()
                    removed += 1
            except Exception:
                pass

        # Hard cap
        remaining = sorted(_TASKS_DIR.glob("task_*.json"), key=lambda f: f.stat().st_mtime)
        while len(remaining) > _MAX_TASK_FILES:
            try:
                remaining[0].unlink()
                remaining.pop(0)
                removed += 1
            except Exception:
                break

        if removed:
            _log.info("task_cleanup | removed %d files", removed)
        return removed

    # ── Disk I/O ─────────────────────────────────────────────────────────

    def _write(self, task: TaskFile) -> None:
        _TASKS_DIR.mkdir(parents=True, exist_ok=True)
        path = task._file_path()
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(task.to_dict(), indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            _log.error("task_write | failed | id=%s err=%s", task.task_id, exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _read(self, task_id: str) -> Optional[TaskFile]:
        for f in _TASKS_DIR.glob(f"task_{task_id}_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                return TaskFile.from_dict(data)
            except Exception as exc:
                _log.warning("task_read | parse error | file=%s err=%s", f.name, exc)
        return None

    def _read_all(self) -> List[TaskFile]:
        tasks = []
        if not _TASKS_DIR.exists():
            return tasks
        for f in sorted(_TASKS_DIR.glob("task_*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                tasks.append(TaskFile.from_dict(data))
            except Exception:
                pass
        return tasks
