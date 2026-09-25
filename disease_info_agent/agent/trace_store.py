"""Persist chat execution traces to output/traces/."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from agent.run_helpers import AGENT_ROOT

TRACES_DIR = AGENT_ROOT / "output" / "traces"


def save_trace(trace: dict[str, Any]) -> str:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = trace.get("trace_id") or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    trace["trace_id"] = trace_id
    trace.setdefault("saved_at", datetime.now(UTC).isoformat())
    path = TRACES_DIR / f"{trace_id}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    latest = TRACES_DIR / "latest.json"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return trace_id


def load_trace(trace_id: str) -> dict[str, Any]:
    path = TRACES_DIR / f"{trace_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"trace 不存在: {trace_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_trace() -> dict[str, Any] | None:
    latest = TRACES_DIR / "latest.json"
    if not latest.exists():
        return None
    return json.loads(latest.read_text(encoding="utf-8"))
