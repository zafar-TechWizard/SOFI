"""
BRAIN/agents/heartbeat.py — Heartbeat monitor daemon for sub-agents.

Runs as a daemon thread, checks all active sub-agents on a fixed
interval. Auto-interrupts agents that have gone stale (no activity
within thresholds).
"""

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from BRAIN.agents.registry import ActiveRegistry

_log = logging.getLogger("sofi.brain.agents.heartbeat")

HEARTBEAT_INTERVAL = 15.0
IDLE_STALE_THRESHOLD = 300.0
IN_TOOL_STALE_THRESHOLD = 600.0


class HeartbeatMonitor:
    """
    Daemon thread that monitors active sub-agents for staleness.

    Checks every HEARTBEAT_INTERVAL seconds. If a sub-agent hasn't
    reported activity within IDLE_STALE_THRESHOLD, it's auto-interrupted.
    Agents actively running a tool get a longer leash (IN_TOOL_STALE_THRESHOLD).
    """

    def __init__(self, registry: "ActiveRegistry") -> None:
        self._registry = registry
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="sofi-heartbeat",
            daemon=True,
        )
        self._thread.start()
        _log.info("heartbeat_started | interval=%.0fs", HEARTBEAT_INTERVAL)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=HEARTBEAT_INTERVAL + 2)
            self._thread = None
        _log.info("heartbeat_stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self._check()
            except Exception:
                _log.exception("heartbeat_error")

    def _check(self) -> None:
        active = self._registry.list_active()
        if not active:
            return

        now = time.time()
        for rec in active:
            if rec.is_interrupted:
                continue

            idle = now - rec.last_activity
            threshold = (
                IN_TOOL_STALE_THRESHOLD if rec.current_tool
                else IDLE_STALE_THRESHOLD
            )

            if idle > threshold:
                _log.warning(
                    "stale_agent | %s (%s) idle=%.0fs threshold=%.0fs tool=%s — auto-interrupting",
                    rec.subagent_id, rec.agent_type, idle, threshold, rec.current_tool,
                )
                rec.interrupt()
