"""Pipeline data schemas — all stages pass these objects."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FactField(str, Enum):
    SYMPTOMS = "symptoms"
    TRANSMISSION = "transmission"
    PREVENTION = "prevention"
    TREATMENT = "treatment"
    EPIDEMIOLOGY = "epidemiology"
    GENERAL = "general"


class RawPage(BaseModel):
    """Fetcher output: one downloaded page."""

    url: str
    text: str = ""
    html: str | None = None
    status_code: int | None = None
    fetch_method: str = "unknown"
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None


class ExtractedFact(BaseModel):
    """Extractor output: one structured fact tied to a source."""

    disease: str
    field: str = FactField.GENERAL.value
    content: str
    source_url: str
    confidence: float | None = None
    context: str | None = None


class DiseaseProfile(BaseModel):
    """Aggregator output: summary for one target disease."""

    disease: str
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    count: int = 0
    items: list[ExtractedFact] = Field(default_factory=list)


class PipelineRun(BaseModel):
    """Run metadata written to output/runs/{run_id}/run_manifest.json."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    urls_file: str | None = None
    urls: list[str] = Field(default_factory=list)
    target_diseases: list[str] = Field(default_factory=list)
    pipeline_version: str = "v1-legacy"
    status: str = "running"
    stats: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = ""

    def to_manifest(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()
