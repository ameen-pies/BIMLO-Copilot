"""
Session Monitor — Lightweight in-memory session lifecycle tracker.

Tracks active pipeline sessions for the admin dashboard Monitor tab.
No user content (queries/answers) — only pipeline metadata:
session_id, anonymized user label, route, stage, latency, success/fail.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionInfo:
    session_id: str
    user_label: str          # "user_abc1" (first 4 chars of user_id)
    route: str               # rag, graph, cad, report, etc.
    stage: str               # classify_intent, retrieve_vector, synthesise, etc.
    started_at: float        # time.time()
    last_activity: float     # time.time()
    stages: List[Dict] = field(default_factory=list)  # [{stage, ts, latency_ms}]


@dataclass
class SessionEvent:
    session_id: str
    user_label: str
    route: str
    event: str               # "start", "update", "complete"
    stage: str
    ts: float
    success: Optional[bool] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class SessionMonitor:
    """In-memory session lifecycle tracker. Thread-safe."""

    def __init__(self, max_active: int = 200, max_history: int = 500):
        self._lock = threading.Lock()
        self._active: Dict[str, SessionInfo] = {}
        self._history: deque[SessionEvent] = deque(maxlen=max_history)
        self._subscribers: list = []  # list of queue.Queue for SSE
        self._max_active = max_active

    def start(self, session_id: str, user_label: str, route: str):
        """Called when a query starts processing."""
        now = time.time()
        info = SessionInfo(
            session_id=session_id,
            user_label=user_label,
            route=route,
            stage="starting",
            started_at=now,
            last_activity=now,
        )
        event = SessionEvent(
            session_id=session_id,
            user_label=user_label,
            route=route,
            event="start",
            stage="starting",
            ts=now,
        )
        with self._lock:
            # Evict oldest if at capacity
            if len(self._active) >= self._max_active and session_id not in self._active:
                oldest_key = min(self._active, key=lambda k: self._active[k].last_activity)
                del self._active[oldest_key]
            self._active[session_id] = info
            self._history.append(event)
        self._broadcast(event)

    def update(self, session_id: str, stage: str, latency_ms: Optional[float] = None):
        """Called after each pipeline node completes."""
        now = time.time()
        with self._lock:
            info = self._active.get(session_id)
            if not info:
                return
            info.stage = stage
            info.last_activity = now
            info.stages.append({"stage": stage, "ts": now, "latency_ms": latency_ms})

        event = SessionEvent(
            session_id=session_id,
            user_label=info.user_label,
            route=info.route,
            event="update",
            stage=stage,
            ts=now,
            latency_ms=latency_ms,
        )
        with self._lock:
            self._history.append(event)
        self._broadcast(event)

    def complete(self, session_id: str, success: bool, error: Optional[str] = None):
        """Called when a query finishes (success or failure)."""
        now = time.time()
        with self._lock:
            info = self._active.pop(session_id, None)
            if not info:
                return
            total_ms = (now - info.started_at) * 1000

        event = SessionEvent(
            session_id=session_id,
            user_label=info.user_label,
            route=info.route,
            event="complete",
            stage=info.stage,
            ts=now,
            success=success,
            error=error,
            latency_ms=round(total_ms, 1),
        )
        with self._lock:
            self._history.append(event)
        self._broadcast(event)

    def snapshot(self) -> Dict:
        """Return current state for REST polling fallback."""
        now = time.time()
        with self._lock:
            active = [
                {
                    "session_id": s.session_id,
                    "user_label": s.user_label,
                    "route": s.route,
                    "stage": s.stage,
                    "elapsed_ms": round((now - s.started_at) * 1000, 1),
                    "last_activity": s.last_activity,
                }
                for s in sorted(self._active.values(), key=lambda x: x.last_activity, reverse=True)
            ]
            recent = [
                {
                    "session_id": e.session_id,
                    "user_label": e.user_label,
                    "route": e.route,
                    "event": e.event,
                    "stage": e.stage,
                    "ts": e.ts,
                    "success": e.success,
                    "error": e.error,
                    "latency_ms": e.latency_ms,
                }
                for e in list(self._history)[-50:]
            ]

        # Aggregates from history
        five_min_ago = now - 300
        recent_events = [e for e in self._history if e.ts > five_min_ago]
        total_recent = len(recent_events)
        errors = sum(1 for e in recent_events if e.success is False)
        throughput = total_recent / 5.0 if total_recent > 0 else 0  # events per minute

        return {
            "active": active,
            "recent": recent,
            "aggregates": {
                "active_count": len(active),
                "throughput_per_min": round(throughput, 1),
                "errors_5min": errors,
                "total_5min": total_recent,
            },
        }

    def subscribe(self):
        """Create a subscriber queue for SSE streaming."""
        import queue as _queue
        q = _queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        """Remove a subscriber queue."""
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast(self, event: SessionEvent):
        """Push event to all SSE subscribers."""
        data = {
            "session_id": event.session_id,
            "user_label": event.user_label,
            "route": event.route,
            "event": event.event,
            "stage": event.stage,
            "ts": event.ts,
            "success": event.success,
            "error": event.error,
            "latency_ms": event.latency_ms,
        }
        import json
        msg = json.dumps(data)
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)


# Singleton
session_monitor = SessionMonitor()
