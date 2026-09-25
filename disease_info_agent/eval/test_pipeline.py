"""Pytest checks for pipeline run quality (B4-3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_AGENT_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN = Path(__file__).resolve().parent / "golden" / "pipeline_urls.json"


def _runs_root() -> Path:
    return _AGENT_ROOT / "output" / "runs"


def _load_thresholds() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def _resolve_run_dir(run_id: str | None = None) -> Path:
    if run_id:
        path = _runs_root() / run_id
        if not path.exists():
            pytest.skip(f"run 不存在: {path}")
        return path
    run_dirs = sorted([p for p in _runs_root().iterdir() if p.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        pytest.skip("还没有任何 pipeline run")
    return run_dirs[-1]


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"缺少 run_manifest.json: {run_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def latest_run_manifest() -> tuple[Path, dict]:
    run_dir = _resolve_run_dir()
    return run_dir, _load_manifest(run_dir)


def test_pipeline_fetch_success(latest_run_manifest: tuple[Path, dict]) -> None:
    _, manifest = latest_run_manifest
    thresholds = _load_thresholds()
    stats = manifest.get("stats") or {}
    fetch_ok = int(stats.get("fetch_success_count") or 0)
    assert fetch_ok >= int(thresholds["min_fetch_success_count"]), (
        f"fetch_success_count={fetch_ok} < {thresholds['min_fetch_success_count']}"
    )


def test_pipeline_validated_fact_count(latest_run_manifest: tuple[Path, dict]) -> None:
    _, manifest = latest_run_manifest
    thresholds = _load_thresholds()
    stats = manifest.get("stats") or {}
    validated = stats.get("validated_fact_count")
    if validated is None:
        validation_stats = stats.get("validation_stats") or {}
        validated = validation_stats.get("output_count")
    assert validated is not None, "manifest 缺少 validated_fact_count"
    assert int(validated) >= int(thresholds["min_validated_fact_count"]), (
        f"validated_fact_count={validated} < {thresholds['min_validated_fact_count']}"
    )


def test_pipeline_knowledge_db_documents(latest_run_manifest: tuple[Path, dict]) -> None:
    run_dir, manifest = latest_run_manifest
    thresholds = _load_thresholds()
    stats = manifest.get("stats") or {}
    doc_count = stats.get("knowledge_db_document_count")
    knowledge_db = run_dir / "knowledge_db"
    if doc_count is None and knowledge_db.exists():
        manifest_path = knowledge_db / "ingest_manifest.json"
        if manifest_path.exists():
            doc_count = json.loads(manifest_path.read_text(encoding="utf-8")).get("document_count")
    assert doc_count is not None, "manifest 缺少 knowledge_db_document_count"
    assert int(doc_count) >= int(thresholds["min_knowledge_db_document_count"]), (
        f"knowledge_db_document_count={doc_count} < {thresholds['min_knowledge_db_document_count']}"
    )


def test_pipeline_url_coverage(latest_run_manifest: tuple[Path, dict]) -> None:
    _, manifest = latest_run_manifest
    thresholds = _load_thresholds()
    urls = manifest.get("urls") or []
    fragments = thresholds.get("expected_url_fragments") or []
    if not fragments:
        pytest.skip("未配置 expected_url_fragments")
    joined = " ".join(urls).lower()
    matched = [frag for frag in fragments if frag.lower() in joined]
    assert matched, f"urls 未覆盖任何期望域名片段: {fragments}"
