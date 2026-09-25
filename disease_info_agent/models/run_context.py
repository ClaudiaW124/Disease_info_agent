"""Manage output/runs/{run_id}/ directory layout and run_manifest.json."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from models.schemas import PipelineRun


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class RunContext:
    """Creates and tracks one pipeline run under output/runs/{run_id}/."""

    def __init__(self, project_root: Path, run_id: str | None = None) -> None:
        self.project_root = project_root
        self.run_id = run_id or generate_run_id()
        self.runs_root = project_root / "output" / "runs"
        self.run_dir = self.runs_root / self.run_id
        self.raw_pages_dir = self.run_dir / "raw_pages"
        self.facts_dir = self.run_dir / "facts"
        self.validated_facts_dir = self.run_dir / "facts_validated"
        self.scraper_results_dir = self.run_dir / "scraper_results"
        self.visualizations_dir = self.run_dir / "visualizations"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.aggregated_data_path = self.run_dir / "aggregated_data.json"
        self.knowledge_db_dir = self.run_dir / "knowledge_db"
        self._pipeline_run: PipelineRun | None = None

    def ensure_dirs(self) -> None:
        for directory in (
            self.run_dir,
            self.raw_pages_dir,
            self.facts_dir,
            self.validated_facts_dir,
            self.scraper_results_dir,
            self.visualizations_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def start_manifest(
        self,
        *,
        urls_file: str,
        urls: list[str],
        target_diseases: list[str],
        pipeline_version: str = "v1-legacy",
    ) -> PipelineRun:
        self.ensure_dirs()
        self._pipeline_run = PipelineRun(
            run_id=self.run_id,
            started_at=datetime.now(UTC),
            urls_file=urls_file,
            urls=urls,
            target_diseases=target_diseases,
            pipeline_version=pipeline_version,
            status="running",
            output_dir=str(self.run_dir),
        )
        self.write_manifest()
        return self._pipeline_run

    def finish_manifest(self, stats: dict | None = None, status: str = "completed") -> None:
        if self._pipeline_run is None:
            return
        self._pipeline_run.finished_at = datetime.now(UTC)
        self._pipeline_run.status = status
        if stats:
            self._pipeline_run.stats.update(stats)
        self.write_manifest()

    def write_manifest(self) -> None:
        if self._pipeline_run is None:
            return
        payload = self._pipeline_run.to_manifest()
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
