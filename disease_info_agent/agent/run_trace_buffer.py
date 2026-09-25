"""In-memory execution trace buffer for a single chat turn (Phase 3)."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_current: ContextVar[dict[str, Any] | None] = ContextVar("disease_trace", default=None)


def start_trace(*, user_message: str) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "trace_id": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        "started_at": datetime.now(UTC).isoformat(),
        "user_message": user_message,
        "events": [],
    }
    _current.set(trace)
    return trace


def get_trace() -> dict[str, Any] | None:
    return _current.get()


def append_event(phase: str, name: str, detail: dict[str, Any] | None = None) -> None:
    trace = _current.get()
    if trace is None:
        return
    trace["events"].append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "phase": phase,
            "name": name,
            "detail": detail or {},
        }
    )


def finish_trace(extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
    trace = _current.get()
    if trace is None:
        return None
    trace["finished_at"] = datetime.now(UTC).isoformat()
    if extra:
        trace.update(extra)
    _current.set(None)
    return trace
