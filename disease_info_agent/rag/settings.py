"""Shared RAG configuration loaded from .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
MIN_RELEVANCE_SCORE = 0.25
COLLECTION_NAME = "disease_facts"


def load_env() -> None:
    load_dotenv(find_dotenv(), override=True)


def require_llm_env() -> tuple[str, str, str]:
    load_env()
    base_url = os.getenv("UTU_LLM_BASE_URL")
    api_key = os.getenv("UTU_LLM_API_KEY")
    model = os.getenv("RAG_LLM_MODEL") or os.getenv("UTU_LLM_MODEL")
    if not base_url or not api_key:
        raise ValueError("UTU_LLM_BASE_URL and UTU_LLM_API_KEY must be set in .env")
    if not model:
        raise ValueError("RAG_LLM_MODEL or UTU_LLM_MODEL must be set in .env")
    return base_url, api_key, model


def require_embedding_env() -> tuple[str, str, str]:
    load_env()
    base_url = os.getenv("UTU_LLM_BASE_URL")
    api_key = os.getenv("UTU_LLM_API_KEY")
    model = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if not base_url or not api_key:
        raise ValueError("UTU_LLM_BASE_URL and UTU_LLM_API_KEY must be set in .env")
    return base_url, api_key, model


def knowledge_db_path(project_root: Path, run_id: str) -> Path:
    return project_root / "output" / "runs" / run_id / "knowledge_db"
