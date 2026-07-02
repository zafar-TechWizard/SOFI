"""
BRAIN/observability/metrics.py — Lightweight in-memory metrics

Fire-and-forget counters + histograms for SOFi's runtime diagnostics.
Thread-safe. Sub-microsecond hot-path cost (dict increment + deque append).

Usage:
    from BRAIN.observability.metrics import get_metrics
    m = get_metrics()
    m.inc("llm_calls")
    m.observe("response_latency_ms", 1234.5)
    m.snapshot()  # → dict for /inspect

Flush is optional — writes a JSON file post-response via ThreadPoolExecutor.
Never blocks the response path.
"""

import json
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("sofi.brain.metrics")

HISTOGRAM_MAX_SIZE = 1000


class MetricsCollector:
    """
    Thread-safe counters + bounded histograms.

    inc()     — increment a counter (~50ns)
    observe() — record a histogram value (~100ns)
    snapshot() — full state dict for /inspect
    flush()   — async write to disk (post-response only)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self._session_id = session_id or str(uuid.uuid4())[:12]
        self._session_start = time.time()
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, deque] = {}
        self._flush_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrics-flush")

    @property
    def session_id(self) -> str:
        return self._session_id

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = deque(maxlen=HISTOGRAM_MAX_SIZE)
            self._histograms[name].append(value)

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            hist_stats = {}
            for name, values in self._histograms.items():
                if not values:
                    continue
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                hist_stats[name] = {
                    "count": n,
                    "min": round(sorted_vals[0], 1),
                    "max": round(sorted_vals[-1], 1),
                    "avg": round(sum(sorted_vals) / n, 1),
                    "p50": round(sorted_vals[n // 2], 1),
                    "p95": round(sorted_vals[int(n * 0.95)], 1) if n >= 20 else None,
                }

            return {
                "session_id": self._session_id,
                "uptime_s": round(time.time() - self._session_start, 1),
                "counters": dict(self._counters),
                "histograms": hist_stats,
            }

    def flush(self, log_dir: Optional[Path] = None) -> None:
        """
        Fire-and-forget write to disk. Submits to a single-thread pool
        so it never blocks the caller. Safe to call from the response path.
        """
        snap = self.snapshot()
        target = log_dir or Path(__file__).parent.parent / "memory" / "data" / "logs"

        try:
            self._flush_pool.submit(self._write_snapshot, snap, target)
        except RuntimeError:
            pass

    @staticmethod
    def _write_snapshot(snap: Dict, log_dir: Path) -> None:
        try:
            metrics_dir = log_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            filepath = metrics_dir / f"{date_str}_{snap['session_id']}.json"

            filepath.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            _log.debug("metrics flushed to %s", filepath)
        except Exception as exc:
            _log.debug("metrics flush failed: %s", exc)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    def shutdown(self) -> None:
        self._flush_pool.shutdown(wait=False)


_global_metrics: Optional[MetricsCollector] = None
_global_lock = threading.Lock()


def get_metrics(session_id: Optional[str] = None) -> MetricsCollector:
    """Module-level singleton. Created on first call."""
    global _global_metrics
    with _global_lock:
        if _global_metrics is None:
            _global_metrics = MetricsCollector(session_id=session_id)
        return _global_metrics


def reset_metrics(session_id: Optional[str] = None) -> MetricsCollector:
    """Replace the global singleton (used on hot reload to carry session_id)."""
    global _global_metrics
    with _global_lock:
        if _global_metrics is not None:
            _global_metrics.shutdown()
        _global_metrics = MetricsCollector(session_id=session_id)
        return _global_metrics
