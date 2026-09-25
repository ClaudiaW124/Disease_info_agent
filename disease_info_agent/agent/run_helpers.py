"""Shared helpers for run directories and summaries."""

from __future__ import annotations

import json
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


def runs_root() -> Path:
    return AGENT_ROOT / "output" / "runs"


def list_run_dirs() -> list[Path]:
    root = runs_root()
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)


def load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {"run_id": run_dir.name, "status": "unknown", "stats": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def summarize_run(run_dir: Path) -> dict:
    manifest = load_manifest(run_dir)
    stats = manifest.get("stats") or {}
    knowledge_db = run_dir / "knowledge_db"
    validated_count = stats.get("validated_fact_count")
    if validated_count is None:
        validation_stats = stats.get("validation_stats") or {}
        validated_count = validation_stats.get("output_count")
    return {
        "run_id": manifest.get("run_id") or run_dir.name,
        "status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "url_count": stats.get("url_count"),
        "fetch_success_count": stats.get("fetch_success_count"),
        "validated_fact_count": validated_count,
        "aggregated_disease_count": stats.get("aggregated_disease_count"),
        "knowledge_db_document_count": stats.get("knowledge_db_document_count"),
        "has_knowledge_db": knowledge_db.exists() and any(knowledge_db.iterdir()) if knowledge_db.exists() else False,
        "output_dir": str(run_dir),
    }


def resolve_run_dir(run_id: str | None = None) -> Path:
    if run_id:
        path = runs_root() / run_id
        if not path.exists():
            raise FileNotFoundError(f"run 不存在: {path}")
        return path
    run_dirs = list_run_dirs()
    if not run_dirs:
        raise FileNotFoundError("还没有任何 run，请先运行 pipeline。")
    return run_dirs[-1]
