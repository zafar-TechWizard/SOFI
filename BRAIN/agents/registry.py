"""
BRAIN/agents/registry.py — Active sub-agent registry.

Thread-safe registry of all running sub-agents. Enforces concurrency
limits, supports interrupt propagation, and provides real-time status
for the prompt builder and UI.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_log = logging.getLogger("sofi.brain.agents.registry")

MAX_CONCURRENT = 3


@dataclass
class SubAgentRecord:
    """One live sub-agent tracked in the registry."""

    subagent_id: str
    task_id: str
    agent_type: str
    query: str
    started_at: float = field(default_factory=time.time)

    iteration: int = 0
    current_tool: Optional[str] = None
    last_activity: float = field(default_factory=time.time)

    interrupted: bool = False
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    def touch(self, iteration: int = 0, tool: Optional[str] = None) -> None:
        self.last_activity = time.time()
        if iteration:
            self.iteration = iteration
        if tool is not None:
            self.current_tool = tool

    def interrupt(self) -> None:
        self.interrupted = True
        self._cancel_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()

    @property
    def is_interrupted(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    @property
    def runtime_seconds(self) -> float:
        return time.time() - self.started_at

    def snapshot(self) -> dict:
        return {
            "subagent_id": self.subagent_id,
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "query": self.query[:80],
            "iteration": self.iteration,
            "current_tool": self.current_tool,
            "idle_seconds": round(self.idle_seconds, 1),
            "runtime_seconds": round(self.runtime_seconds, 1),
            "interrupted": self.interrupted,
        }


class ActiveRegistry:
    """
    Thread-safe registry of running sub-agents.

    Enforces MAX_CONCURRENT limit, provides real-time snapshots for
    the prompt builder, and supports bulk interrupt for shutdown.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT) -> None:
        self.max_concurrent = max_concurrent
        self._agents: Dict[str, SubAgentRecord] = {}
        self._lock = threading.Lock()
        self._on_stale: Optional[Callable[[SubAgentRecord], None]] = None
        self._completed_count = 0
        self._total_duration = 0.0

    def register(self, record: SubAgentRecord) -> bool:
        """
        Register a new sub-agent. Returns False if at capacity.
        """
        with self._lock:
            active = len(self._agents)
            if active >= self.max_concurrent:
                _log.warning(
                    "registry_full | %d/%d slots used, rejecting %s",
                    active, self.max_concurrent, record.subagent_id,
                )
                return False
            self._agents[record.subagent_id] = record
            _log.info(
                "registered | %s (%s) task=%s, slots=%d/%d",
                record.subagent_id, record.agent_type, record.task_id,
                active + 1, self.max_concurrent,
            )
            return True

    def unregister(self, subagent_id: str) -> Optional[SubAgentRecord]:
        """Remove a completed/failed sub-agent from the registry."""
        with self._lock:
            record = self._agents.pop(subagent_id, None)
            if record:
                self._completed_count += 1
                self._total_duration += record.runtime_seconds
                _log.info(
                    "unregistered | %s ran %.1fs, slots=%d/%d",
                    subagent_id, record.runtime_seconds,
                    len(self._agents), self.max_concurrent,
                )
            return record

    def get(self, subagent_id: str) -> Optional[SubAgentRecord]:
        with self._lock:
            return self._agents.get(subagent_id)

    def touch(self, subagent_id: str, iteration: int = 0, tool: Optional[str] = None) -> None:
        with self._lock:
            rec = self._agents.get(subagent_id)
            if rec:
                rec.touch(iteration, tool)

    def interrupt(self, subagent_id: str) -> bool:
        with self._lock:
            rec = self._agents.get(subagent_id)
            if rec:
                rec.interrupt()
                _log.info("interrupted | %s", subagent_id)
                return True
            return False

    def interrupt_all(self) -> int:
        """Interrupt all running sub-agents. Returns count interrupted."""
        with self._lock:
            count = 0
            for rec in self._agents.values():
                if not rec.is_interrupted:
                    rec.interrupt()
                    count += 1
            if count:
                _log.info("interrupt_all | %d agents interrupted", count)
            return count

    def list_active(self) -> List[SubAgentRecord]:
        with self._lock:
            return list(self._agents.values())

    def active_count(self) -> int:
        with self._lock:
            return len(self._agents)

    def has_capacity(self) -> bool:
        with self._lock:
            return len(self._agents) < self.max_concurrent

    def find_stale(self, idle_threshold: float = 300.0) -> List[SubAgentRecord]:
        """Find sub-agents that haven't reported activity recently."""
        with self._lock:
            return [r for r in self._agents.values() if r.idle_seconds > idle_threshold]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": [r.snapshot() for r in self._agents.values()],
                "slots": f"{len(self._agents)}/{self.max_concurrent}",
                "completed_total": self._completed_count,
                "total_runtime": round(self._total_duration, 1),
            }
