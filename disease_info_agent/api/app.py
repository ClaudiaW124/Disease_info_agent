"""B5-2 FastAPI — health / ask / pipeline + web demo UI at /."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_AGENT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _AGENT_ROOT.parent
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

app = FastAPI(
    title="Disease Info Agent",
    description="疾病信息采集 + Orchestrator Agent + LangGraph RAG 问答 API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_rag_warmup_done = False
_coordinator = None
_coordinator_lock = asyncio.Lock()


async def _get_coordinator():
    global _coordinator
    async with _coordinator_lock:
        if _coordinator is None:
            from agent.coordinator import DiseaseInfoCoordinator

            _coordinator = DiseaseInfoCoordinator(_AGENT_ROOT, enable_planner=True)
            await _coordinator.load()
        return _coordinator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    reset_history: bool = Field(default=False, description="清空 Orchestrator 多轮历史")
    no_plan: bool = Field(default=False, description="跳过 Planner，Orchestrator 直连")


class ChatResponse(BaseModel):
    reply: str
    mode: str = Field(default="direct", description="direct 或 plan")
    plan_steps: list[str] | None = Field(default=None, description="Planner 步骤摘要（mode=plan 时）")
    trace: dict[str, Any] | None = Field(default=None, description="Planner 执行轨迹（可视化用）")
    trace_id: str | None = Field(default=None, description="轨迹 ID，对应 output/traces/{id}.json")
    agent: str = "DiseaseInfoCoordinator"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    run_id: str | None = Field(default=None, description="指定 run_id，默认最新")


class PipelineRunRequest(BaseModel):
    urls_path: str = Field(default="urls.txt", description="相对 disease_info_agent/ 的 URL 列表")
    run_id: str | None = Field(default=None, description="可选，复用已有 run 目录")


def _runs_root() -> Path:
    return _AGENT_ROOT / "output" / "runs"


def _list_run_dirs() -> list[Path]:
    root = _runs_root()
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {"run_id": run_dir.name, "status": "unknown", "stats": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(run_dir)
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


def _resolve_run_dir(run_id: str | None) -> Path:
    if run_id:
        path = _runs_root() / run_id
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"run 不存在: {run_id}")
        return path
    run_dirs = _list_run_dirs()
    if not run_dirs:
        raise HTTPException(status_code=404, detail="还没有任何 run，请先运行 pipeline")
    return run_dirs[-1]


def _warmup_rag() -> None:
    global _rag_warmup_done
    if _rag_warmup_done:
        return
    try:
        from rag.graph import get_rag_graph

        run_dir = _resolve_run_dir(None)
        knowledge_db = run_dir / "knowledge_db"
        if knowledge_db.exists():
            get_rag_graph(knowledge_db)
        _rag_warmup_done = True
    except HTTPException:
        pass
    except Exception:
        pass


@app.on_event("startup")
async def _startup_warmup() -> None:
    asyncio.create_task(asyncio.to_thread(_warmup_rag))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    runs = [_summarize_run(path) for path in _list_run_dirs()]
    return {"runs": runs, "count": len(runs)}


@app.get("/api/runs/latest")
async def latest_run() -> dict[str, Any]:
    try:
        return _summarize_run(_resolve_run_dir(None))
    except HTTPException as exc:
        raise exc


@app.get("/api/eval/report")
async def eval_report() -> dict[str, Any]:
    report_path = _AGENT_ROOT / "eval" / "output" / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="尚未运行 eval harness")
    return json.loads(report_path.read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Coordinator — Planner decomposes complex tasks; Orchestrator handles simple routing."""
    coordinator = await _get_coordinator()
    if req.reset_history:
        await coordinator.reset_history()
    result = await coordinator.chat(req.message.strip(), no_plan=req.no_plan)
    trace_payload = result.trace
    plan_steps = None
    if trace_payload and trace_payload.get("steps"):
        plan_steps = [
            f"{step['id']}. {step['tool']}: {step.get('description') or step.get('summary')}"
            for step in trace_payload["steps"]
        ]
    return ChatResponse(
        reply=result.reply,
        mode=result.mode,
        plan_steps=plan_steps,
        trace=trace_payload,
        trace_id=result.trace_id,
    )


@app.get("/api/traces/latest")
async def traces_latest() -> dict[str, Any]:
    from agent.trace_store import load_latest_trace

    payload = load_latest_trace()
    if payload is None:
        raise HTTPException(status_code=404, detail="尚无执行轨迹")
    return payload


@app.get("/api/traces/{trace_id}")
async def traces_get(trace_id: str) -> dict[str, Any]:
    from agent.trace_store import load_trace

    try:
        return load_trace(trace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    from rag.graph import run_rag_graph

    run_dir = _resolve_run_dir(req.run_id)
    knowledge_db = run_dir / "knowledge_db"
    if not knowledge_db.exists():
        raise HTTPException(status_code=400, detail=f"run {run_dir.name} 还没有 knowledge_db，请先 ingest 或 pipeline")

    result = await asyncio.to_thread(run_rag_graph, req.question.strip(), knowledge_db)
    payload = result.model_dump()
    payload["run_id"] = run_dir.name
    return payload


@app.post("/pipeline/run")
async def pipeline_run(req: PipelineRunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    from models.run_context import RunContext

    run_context = RunContext(_AGENT_ROOT, run_id=req.run_id)
    run_id = run_context.run_id

    with _jobs_lock:
        existing = _pipeline_jobs.get(run_id)
        if existing and existing.get("status") == "running":
            return {
                "run_id": run_id,
                "status": "running",
                "message": "该 run 的 pipeline 已在执行中",
                "poll_url": f"/pipeline/status/{run_id}",
            }
        _pipeline_jobs[run_id] = {
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "urls_path": req.urls_path,
        }

    background_tasks.add_task(_run_pipeline_background, run_id, req.urls_path)
    return {
        "run_id": run_id,
        "status": "started",
        "message": "Pipeline 已在后台启动，可通过 /pipeline/status/{run_id} 查询进度",
        "poll_url": f"/pipeline/status/{run_id}",
        "output_dir": str(run_context.run_dir),
    }


@app.get("/pipeline/status/{run_id}")
async def pipeline_status(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _pipeline_jobs.get(run_id)
    if job:
        return job

    run_dir = _runs_root() / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"未找到 run 或任务: {run_id}")

    manifest = _load_manifest(run_dir)
    return {
        "run_id": run_id,
        "status": manifest.get("status", "unknown"),
        "manifest": manifest,
        "summary": _summarize_run(run_dir),
    }


async def _run_pipeline_background(run_id: str, urls_path: str) -> None:
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
        result = await run_pipeline_impl(args, _AGENT_ROOT)
        with _jobs_lock:
            _pipeline_jobs[run_id] = {
                "run_id": run_id,
                "status": "completed",
                "finished_at": datetime.now(UTC).isoformat(),
                "result": result,
                "manifest": _load_manifest(_runs_root() / run_id),
            }
        global _rag_warmup_done
        _rag_warmup_done = False
        _warmup_rag()
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _pipeline_jobs[run_id] = {
                "run_id": run_id,
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            }


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
