"""Orchestrator Agent tools — query / pipeline / eval / run stats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agents import function_tool

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.tool_impl import (  # noqa: E402
    invoke_expand_knowledge_base,
    invoke_get_eval_report,
    invoke_get_latest_run,
    invoke_get_pipeline_status,
    invoke_query_disease,
    invoke_start_pipeline,
)


@function_tool(strict_mode=False)
async def query_disease(question: str, run_id: str = "") -> str:
    """Answer a disease question using the LangGraph RAG knowledge base.

    Use for questions about 登革病毒/猩红热/裂谷热/流感 (symptoms, transmission, prevention, treatment, comparisons).
    Returns JSON with answer, citations, refused flag.

    Args:
        question: User question in Chinese or English.
        run_id: Optional run id; leave empty to use the latest run with knowledge_db.
    """
    payload = await invoke_query_disease(question, run_id)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@function_tool
def get_latest_run() -> str:
    """Get the latest pipeline run summary: run_id, fact counts, knowledge_db status."""
    payload = invoke_get_latest_run()
    return json.dumps(payload, ensure_ascii=False, indent=2)


@function_tool
def get_eval_report() -> str:
    """Get the latest eval harness summary (pass_rate, Ragas faithfulness, hard case stats)."""
    payload = invoke_get_eval_report()
    return json.dumps(payload, ensure_ascii=False, indent=2)


@function_tool(strict_mode=False)
async def start_pipeline(urls_path: str = "urls.txt") -> str:
    """Start the full disease-info pipeline in background (fetch→extract→validate→aggregate→report→ingest).

    Takes several minutes. Returns run_id immediately; use get_latest_run or get_pipeline_status to check progress.
    Only call when the user explicitly asks to rebuild or refresh the knowledge base.

    Args:
        urls_path: Path to URL list relative to disease_info_agent/ (default urls.txt).
    """
    payload = await invoke_start_pipeline(urls_path)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@function_tool(strict_mode=False)
def get_pipeline_status(run_id: str = "") -> str:
    """Check background pipeline job status for a run_id (or latest started job if empty)."""
    payload = invoke_get_pipeline_status(run_id)
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


@function_tool(strict_mode=False)
async def expand_knowledge_base(topic: str, run_id: str = "", max_urls: int = 3) -> str:
    """Search the web for new disease URLs and incrementally expand the knowledge base.

    Use when query_disease refused or user asks to find/update/add sources for a topic.
    Requires search API (SERPER_API_KEY or JINA). Takes 1–3 minutes.

    Args:
        topic: Disease topic or question to search for new sources.
        run_id: Optional run id; default latest run.
        max_urls: Max new URLs to fetch (default 3).
    """
    payload = await invoke_expand_knowledge_base(topic, run_id, max_urls=max_urls)
    return json.dumps(payload, ensure_ascii=False, indent=2)


ORCHESTRATOR_TOOLS = [
    query_disease,
    get_latest_run,
    get_eval_report,
    start_pipeline,
    get_pipeline_status,
    expand_knowledge_base,
]
