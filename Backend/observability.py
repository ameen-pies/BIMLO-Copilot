"""
observability.py — Structured logging and observability pipeline
for BIMLO Copilot.

Emits JSON log entries for:
  - Agent routing decisions
  - LLM judge outcomes (pass/fail, retry attempts)
  - Retrieval scores (vector + reranker)
  - Ingestion events (per-node timing)
  - End-to-end query latency

Logs go to:
  1. A rotating JSON file  (LOG_DIR/bimlo_structured.jsonl)
  2. stdout (dev-friendly pretty print)

Alert thresholds (configurable via env vars):
  - ALERT_JUDGE_FAIL_THRESHOLD  (default: 3)  — consecutive judge failures before alert
  - ALERT_INGESTION_FAIL_THRESHOLD (default: 2) — consecutive ingestion failures before alert

Usage:
    from observability import obs

    obs.log_routing(session_id, query, route, confidence, latency_ms)
    obs.log_judge(session_id, attempt, passed, score, reason)
    obs.log_retrieval(session_id, query, top_k, scores, reranker_used)
    obs.log_ingestion(doc_id, filename, node, status, latency_ms, details)
    obs.log_query_end(session_id, route, total_latency_ms, success)

    # Context manager for timing a code block:
    with obs.timer("retrieval", session_id=session_id) as t:
        results = vector_store.search(...)
    # t.elapsed_ms is available after the block
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional


# ── Config ────────────────────────────────────────────────────────────────────

LOG_DIR                    = Path(os.getenv("LOG_DIR", "./data/logs"))
LOG_FILE                   = LOG_DIR / "bimlo_structured.jsonl"
MAX_LOG_BYTES              = int(os.getenv("LOG_MAX_BYTES",  str(10 * 1024 * 1024)))  # 10 MB
LOG_BACKUP_COUNT           = int(os.getenv("LOG_BACKUP_COUNT", "5"))
ALERT_JUDGE_FAIL_THRESHOLD = int(os.getenv("ALERT_JUDGE_FAIL_THRESHOLD", "3"))
ALERT_INGEST_FAIL_THRESHOLD= int(os.getenv("ALERT_INGESTION_FAIL_THRESHOLD", "2"))

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Internal logger (writes to rotating JSONL file) ───────────────────────────

_file_logger = logging.getLogger("bimlo.obs")
_file_logger.setLevel(logging.DEBUG)
_file_logger.propagate = False  # don't bleed into root logger

_handler = RotatingFileHandler(
    str(LOG_FILE),
    maxBytes=MAX_LOG_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_handler.setFormatter(logging.Formatter("%(message)s"))   # raw JSON lines
_file_logger.addHandler(_handler)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(event: Dict[str, Any]):
    """Write a structured event to the JSONL file and print a short summary."""
    event.setdefault("ts", _now_iso())
    line = json.dumps(event, ensure_ascii=False, default=str)
    _file_logger.info(line)
    # Dev-friendly stdout (one line, coloured by event type)
    _ICONS = {
        "routing":   "🗺️ ",
        "judge":     "⚖️ ",
        "retrieval": "🔍",
        "ingestion": "📦",
        "query_end": "✅",
        "alert":     "🚨",
        "latency":   "⏱️ ",
    }
    icon = _ICONS.get(event.get("event", ""), "📝")
    summary_parts = [f"{icon} [{event['event']}]"]
    for k in ("session_id", "doc_id", "route", "node", "status", "latency_ms"):
        if k in event:
            summary_parts.append(f"{k}={event[k]}")
    print("  ".join(summary_parts))


# ── Alert state (per-process in-memory counters) ──────────────────────────────

_judge_fail_streak: Dict[str, int]   = {}   # session_id → consecutive failures
_ingest_fail_streak: Dict[str, int]  = {}   # doc_id → consecutive node failures
_lock = threading.Lock()


# ── Timer context manager ─────────────────────────────────────────────────────

class _TimerContext:
    def __init__(self, label: str, **meta):
        self.label     = label
        self.meta      = meta
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        _emit({
            "event":      "latency",
            "label":      self.label,
            "latency_ms": round(self.elapsed_ms, 2),
            **self.meta,
        })


# ── Main observability facade ─────────────────────────────────────────────────

class _Observability:
    """Singleton facade — import as `from observability import obs`."""

    # ── Routing ───────────────────────────────────────────────────────────────

    def log_routing(
        self,
        session_id: str,
        query: str,
        route: str,
        confidence: float,
        latency_ms: float = 0.0,
        intent: Optional[str] = None,
        prev_route: Optional[str] = None,
        forced: bool = False,
    ):
        _emit({
            "event":      "routing",
            "session_id": session_id,
            "query":      query[:200],
            "route":      route,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "intent":     intent,
            "prev_route": prev_route,
            "forced":     forced,
        })

    # ── LLM Judge ─────────────────────────────────────────────────────────────

    def log_judge(
        self,
        session_id: str,
        attempt: int,
        passed: bool,
        score: float,
        reason: Optional[str] = None,
        latency_ms: float = 0.0,
    ):
        _emit({
            "event":      "judge",
            "session_id": session_id,
            "attempt":    attempt,
            "passed":     passed,
            "score":      round(score, 4),
            "reason":     (reason or "")[:300],
            "latency_ms": round(latency_ms, 2),
        })
        # Alert on consecutive failures
        with _lock:
            if not passed:
                _judge_fail_streak[session_id] = _judge_fail_streak.get(session_id, 0) + 1
                streak = _judge_fail_streak[session_id]
                if streak >= ALERT_JUDGE_FAIL_THRESHOLD:
                    self._alert(
                        "judge_failure_streak",
                        f"Session {session_id} has {streak} consecutive judge failures "
                        f"(last score={score:.2f}, reason={reason!r})",
                        session_id=session_id,
                        streak=streak,
                    )
            else:
                _judge_fail_streak[session_id] = 0

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def log_retrieval(
        self,
        session_id: str,
        query: str,
        top_k: int,
        scores: List[float],
        reranker_used: bool = False,
        graph_hits: int = 0,
        latency_ms: float = 0.0,
        source: str = "vector",  # "vector" | "graph" | "hybrid"
    ):
        _emit({
            "event":          "retrieval",
            "session_id":     session_id,
            "query":          query[:200],
            "top_k":          top_k,
            "result_count":   len(scores),
            "min_score":      round(min(scores), 4) if scores else None,
            "max_score":      round(max(scores), 4) if scores else None,
            "avg_score":      round(sum(scores) / len(scores), 4) if scores else None,
            "reranker_used":  reranker_used,
            "graph_hits":     graph_hits,
            "latency_ms":     round(latency_ms, 2),
            "source":         source,
        })

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def log_ingestion(
        self,
        doc_id: str,
        filename: str,
        node: str,            # "chunk_document" | "index_vector_store" | "ingest_graph_rag"
        status: str,          # "started" | "ok" | "skipped" | "failed"
        latency_ms: float = 0.0,
        details: Optional[Dict] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        _emit({
            "event":      "ingestion",
            "doc_id":     doc_id,
            "filename":   filename,
            "node":       node,
            "status":     status,
            "latency_ms": round(latency_ms, 2),
            "session_id": session_id,
            "user_id":    user_id,
            "details":    details or {},
        })
        # Alert on consecutive node failures (same doc)
        with _lock:
            key = f"{doc_id}:{node}"
            if status == "failed":
                _ingest_fail_streak[key] = _ingest_fail_streak.get(key, 0) + 1
                streak = _ingest_fail_streak[key]
                if streak >= ALERT_INGEST_FAIL_THRESHOLD:
                    self._alert(
                        "ingestion_failure_streak",
                        f"Document '{filename}' (doc_id={doc_id}) node '{node}' "
                        f"has failed {streak} times",
                        doc_id=doc_id,
                        node=node,
                        streak=streak,
                    )
            elif status == "ok":
                _ingest_fail_streak.pop(key, None)

    # ── End-to-end query ──────────────────────────────────────────────────────

    def log_query_end(
        self,
        session_id: str,
        route: str,
        total_latency_ms: float,
        success: bool,
        confidence: float = 0.0,
        judge_attempts: int = 1,
        sources_count: int = 0,
        error: Optional[str] = None,
    ):
        _emit({
            "event":            "query_end",
            "session_id":       session_id,
            "route":            route,
            "total_latency_ms": round(total_latency_ms, 2),
            "success":          success,
            "confidence":       round(confidence, 4),
            "judge_attempts":   judge_attempts,
            "sources_count":    sources_count,
            "error":            (error or "")[:300],
        })

    # ── Alert ─────────────────────────────────────────────────────────────────

    def _alert(self, alert_type: str, message: str, **context):
        _emit({
            "event":      "alert",
            "alert_type": alert_type,
            "message":    message,
            **context,
        })
        # In production, extend this to send email / Slack / PagerDuty webhook:
        # e.g. requests.post(os.getenv("ALERT_WEBHOOK_URL"), json={...})

    # ── Timer helper ──────────────────────────────────────────────────────────

    def timer(self, label: str, **meta) -> _TimerContext:
        """Usage: `with obs.timer("retrieval", session_id=sid) as t: ...`"""
        return _TimerContext(label, **meta)

    # ── Admin: read recent logs ───────────────────────────────────────────────

    def tail_logs(self, n: int = 100, event_filter: Optional[str] = None) -> List[Dict]:
        """Return the last n log entries (optionally filtered by event type)."""
        entries: List[Dict] = []
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if event_filter is None or obj.get("event") == event_filter:
                            entries.append(obj)
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            pass
        return entries[-n:]

    def get_stats(self) -> Dict:
        """Aggregate counts by event type from the log file."""
        counts: Dict[str, int] = {}
        judge_pass = judge_fail = 0
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        ev = obj.get("event", "unknown")
                        counts[ev] = counts.get(ev, 0) + 1
                        if ev == "judge":
                            if obj.get("passed"):
                                judge_pass += 1
                            else:
                                judge_fail += 1
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            pass
        return {
            "event_counts":   counts,
            "judge_pass":     judge_pass,
            "judge_fail":     judge_fail,
            "judge_pass_rate": round(judge_pass / (judge_pass + judge_fail), 4)
                               if (judge_pass + judge_fail) else None,
            "log_file":       str(LOG_FILE),
        }


# ── Singleton export ──────────────────────────────────────────────────────────

obs = _Observability()
