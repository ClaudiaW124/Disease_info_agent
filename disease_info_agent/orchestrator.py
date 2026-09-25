"""Compatibility shim — legacy 4-Agent pipeline lives in legacy/legacy_pipeline.py."""

from legacy.legacy_pipeline import DISEASE_KEYWORDS, TARGET_DISEASES, DiseaseInfoOrchestrator

__all__ = ["DiseaseInfoOrchestrator", "TARGET_DISEASES", "DISEASE_KEYWORDS"]
