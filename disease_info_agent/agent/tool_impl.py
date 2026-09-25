"""Core tool implementations shared by Orchestrator tools and Plan executor."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any

from agent.run_helpers import AGENT_ROOT, resolve_run_dir, summarize_run
from agent.run_trace_buffer import append_event

_pipeline_jobs: dict[str, dict[str, Any]] = {}


def _is_unrelated_refusal(payload: dict[str, Any]) -> bool:
    """True when RAG refused because the question is outside the disease KB scope."""
    steps_str = " ".join(payload.get("steps") or [])
    if "guard=out_of_kb" in steps_str:
        return True
    if "rewrite=UNRELATED" in steps_str:
        return True
    if "UNRELATED" in steps_str:
        return True
    return False


def should_auto_kb_expand(question: str, payload: dict[str, Any]) -> bool:
    """Only expand KB for in-scope disease questions that the KB failed to answer well."""
    from rag.graph import question_mentions_in_kb_disease

    if payload.get("error"):
        return False

    # 无关问题（天气、闲聊等）或库外疾病：拒答即可，不搜索扩展
    if _is_unrelated_refusal(payload):
        return False
    if not question_mentions_in_kb_disease(question):
        return False

    if payload.get("refused"):
        return True
    score = float(payload.get("relevance_score") or 0.0)
    if score < 0.4 and not payload.get("validate_passed"):
        return True
    return False


async def invoke_query_disease(
    question: str,
    run_id: str = "",
    *,
    auto_kb_expand: bool | None = None,
) -> dict[str, Any]:
    from rag.graph import run_rag_graph

    if auto_kb_expand is None:
        auto_kb_expand = os.getenv("AUTO_KB_EXPAND", "true").lower() in {"1", "true", "yes"}

    run_dir = resolve_run_dir(run_id or None)
    knowledge_db = run_dir / "knowledge_db"
    if not knowledge_db.exists():
        return {
            "error": f"run {run_dir.name} 还没有 knowledge_db，请先 start_pipeline。",
            "run_id": run_dir.name,
        }

    append_event("rag", "query_start", {"question": question, "run_id": run_dir.name})
    result = await asyncio.to_thread(run_rag_graph, question.strip(), knowledge_db)
    payload = result.model_dump()
    payload["run_id"] = run_dir.name
    append_event("rag", "query_done", {"steps": payload.get("steps"), "refused": payload.get("refused")})

    if auto_kb_expand and should_auto_kb_expand(question, payload):
        append_event("kb_expand", "auto_trigger", {"reason": "in_kb_disease_but_poor_answer"})
        expand = await invoke_expand_knowledge_base(question, run_dir.name)
        payload["kb_expand_attempt"] = expand
        if expand.get("expanded"):
            append_event("rag", "query_retry", {"question": question})
            retry = await asyncio.to_thread(run_rag_graph, question.strip(), knowledge_db)
            payload = retry.model_dump()
            payload["run_id"] = run_dir.name
            payload["kb_expand"] = expand
            append_event("rag", "query_retry_done", {"steps": payload.get("steps"), "refused": payload.get("refused")})

    return payload


async def invoke_expand_knowledge_base(topic: str, run_id: str = "", max_urls: int = 3) -> dict[str, Any]:
    from agent.kb_expand import expand_knowledge_base

    return await expand_knowledge_base(topic, run_id=run_id, max_urls=max_urls)


def invoke_get_latest_run() -> dict[str, Any]:
    try:
        return summarize_run(resolve_run_dir(None))
    except FileNotFoundError as exc:
        return {"error": str(exc)}


def invoke_get_eval_report() -> dict[str, Any]:
    report_path = AGENT_ROOT / "eval" / "output" / "report.json"
    if not report_path.exists():
        return {
            "error": "尚未运行 eval harness",
            "hint": "uv run python -m disease_info_agent.eval.run_harness",
        }
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rag_summary = (payload.get("rag") or {}).get("summary") or {}
    return {
        "run_id": payload.get("run_id"),
        "generated_at": payload.get("generated_at"),
        "pass_rate": rag_summary.get("pass_rate"),
        "hard_case_pass_rate": rag_summary.get("hard_case_pass_rate"),
        "avg_faithfulness": rag_summary.get("avg_faithfulness"),
        "avg_answer_relevancy": rag_summary.get("avg_answer_relevancy"),
        "ragas_status": rag_summary.get("ragas_status"),
        "refusal_accuracy": rag_summary.get("refusal_accuracy"),
        "report_html": str(AGENT_ROOT / "eval" / "output" / "report.html"),
    }


async def _run_pipeline_job(run_id: str, urls_path: str) -> None:
    from main import run_pipeline as run_pipeline_impl

    args = SimpleNamespace(
        urls=urls_path,
        run_id=run_id,
        concurrency=5,
        no_llm=False,
        skip_rag=False,
        facts_only=False,
        verbose=False,
    )
    try:
        result = await run_pipeline_impl(args, AGENT_ROOT)
        _pipeline_jobs[run_id] = {"status": "completed", "result": result}
    except Exception as exc:  # noqa: BLE001
        _pipeline_jobs[run_id] = {"status": "failed", "error": str(exc)}


async def invoke_start_pipeline(urls_path: str = "urls.txt") -> dict[str, Any]:
    from models.run_context import RunContext

    run_context = RunContext(AGENT_ROOT)
    run_id = run_context.run_id
    existing = _pipeline_jobs.get(run_id)
    if existing and existing.get("status") == "running":
        return {"run_id": run_id, "status": "running", "message": "该 run 已在执行中"}

    _pipeline_jobs[run_id] = {"status": "running"}
    asyncio.create_task(_run_pipeline_job(run_id, urls_path))
    return {
        "run_id": run_id,
        "status": "started",
        "message": "Pipeline 已在后台启动，约需 5–10 分钟。",
        "output_dir": str(run_context.run_dir),
    }


def invoke_get_pipeline_status(run_id: str = "") -> dict[str, Any]:
    if run_id:
        job = _pipeline_jobs.get(run_id)
        if job:
            return {"run_id": run_id, **job}
        run_dir = AGENT_ROOT / "output" / "runs" / run_id
        if run_dir.exists():
            return {"run_id": run_id, "status": "unknown", "manifest": summarize_run(run_dir)}
        return {"error": f"未找到 run 或任务: {run_id}"}

    if not _pipeline_jobs:
        return {"message": "当前没有后台 pipeline 任务"}
    last_run_id = list(_pipeline_jobs.keys())[-1]
    return {"run_id": last_run_id, **_pipeline_jobs[last_run_id]}


TOOL_NAMES = (
    "query_disease",
    "get_latest_run",
    "get_eval_report",
    "start_pipeline",
    "get_pipeline_status",
    "expand_knowledge_base",
)


async def invoke_tool(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch a tool by name for the plan executor."""
    args = args or {}
    if name == "query_disease":
        return await invoke_query_disease(
            question=str(args.get("question", "")),
            run_id=str(args.get("run_id", "") or ""),
        )
    if name == "get_latest_run":
        return invoke_get_latest_run()
    if name == "get_eval_report":
        return invoke_get_eval_report()
    if name == "start_pipeline":
        return await invoke_start_pipeline(urls_path=str(args.get("urls_path", "urls.txt")))
    if name == "get_pipeline_status":
        return invoke_get_pipeline_status(run_id=str(args.get("run_id", "") or ""))
    if name == "expand_knowledge_base":
        return await invoke_expand_knowledge_base(
            topic=str(args.get("topic", args.get("question", ""))),
            run_id=str(args.get("run_id", "") or ""),
            max_urls=int(args.get("max_urls", 3)),
        )
    return {"error": f"未知工具: {name}"}
