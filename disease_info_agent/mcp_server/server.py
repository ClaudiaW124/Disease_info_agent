"""disease-info MCP server — expose pipeline / RAG / eval / run stats as tools.

Run (from repo root, stdio):
    python -m disease_info_agent.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from mcp.server.fastmcp import Context, FastMCP

# MCP uses stdout for JSON-RPC. Keep all logs on stderr.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
for _name in ("httpx", "httpcore", "openai", "langchain", "chromadb", "onnxruntime"):
    logging.getLogger(_name).setLevel(logging.WARNING)

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

mcp = FastMCP("disease-info")
_warmup_lock = threading.Lock()
_warmup_done = False


def _runs_root() -> Path:
    return _AGENT_ROOT / "output" / "runs"


def _list_run_dirs() -> list[Path]:
    root = _runs_root()
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {"run_id": run_dir.name, "status": "unknown", "stats": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _summarize_run(run_dir: Path) -> dict:
    manifest = _load_manifest(run_dir)
    stats = manifest.get("stats") or {}
    knowledge_db = run_dir / "knowledge_db"
    validated_dir = run_dir / "facts_validated"
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
        "extractor_fact_count": stats.get("extractor_fact_count") or stats.get("fact_count"),
        "validated_fact_count": validated_count,
        "aggregated_disease_count": stats.get("aggregated_disease_count"),
        "knowledge_db_document_count": stats.get("knowledge_db_document_count"),
        "has_knowledge_db": knowledge_db.exists() and any(knowledge_db.iterdir()) if knowledge_db.exists() else False,
        "has_validated_facts": validated_dir.exists() and any(validated_dir.glob("*.json")),
        "output_dir": str(run_dir),
    }


def _latest_run_dir() -> Path:
    run_dirs = _list_run_dirs()
    if not run_dirs:
        raise FileNotFoundError("还没有任何 run。请先调用 run_pipeline，或用 CLI 跑一遍 pipeline。")
    return run_dirs[-1]


def _run_dir(run_id: str | None) -> Path:
    if run_id:
        path = _runs_root() / run_id
        if not path.exists():
            raise FileNotFoundError(f"run 不存在: {path}")
        return path
    return _latest_run_dir()


@contextmanager
def _stdout_to_stderr():
    """MCP stdio uses stdout for JSON-RPC; keep pipeline prints off that stream."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _warmup_rag() -> None:
    global _warmup_done
    with _warmup_lock:
        if _warmup_done:
            return
        from rag.graph import get_rag_graph

        run_dir = _latest_run_dir()
        knowledge_db = run_dir / "knowledge_db"
        if knowledge_db.exists():
            get_rag_graph(knowledge_db)
        _warmup_done = True


def _start_warmup_thread() -> None:
    def _run() -> None:
        try:
            _warmup_rag()
        except Exception as exc:  # noqa: BLE001 — warmup must not crash the server
            logging.getLogger("disease-info-mcp").warning("RAG warmup skipped: %s", exc)

    threading.Thread(target=_run, name="rag-warmup", daemon=True).start()


@mcp.tool()
def get_latest_run() -> str:
    """Return the latest pipeline run_id and collection / validation / RAG stats as JSON."""
    summary = _summarize_run(_latest_run_dir())
    return json.dumps(summary, ensure_ascii=False, indent=2)


@mcp.tool()
async def query_disease(question: str, run_id: str | None = None, ctx: Context = None) -> str:
    """Ask a disease question against the RAG knowledge base (LangGraph).

    Covers 登革病毒 / 猩红热 / 裂谷热 / 流感. Returns answer plus source URLs.
    Uses the latest run that has knowledge_db unless run_id is given.
    """
    from rag.graph import run_rag_graph

    async def _progress(step: float, message: str) -> None:
        if ctx is None:
            return
        try:
            await ctx.report_progress(step, 4, message)
        except Exception:  # noqa: BLE001
            return

    await _progress(0, "准备知识库")
    run_dir = _run_dir(run_id)
    knowledge_db = run_dir / "knowledge_db"
    if not knowledge_db.exists():
        return json.dumps(
            {
                "error": f"run {run_dir.name} 还没有 knowledge_db，请先 ingest 或 run_pipeline。",
                "run_id": run_dir.name,
            },
            ensure_ascii=False,
        )

    stop = asyncio.Event()

    async def _heartbeat() -> None:
        tick = 1
        while not stop.is_set():
            await _progress(min(3, 1 + tick * 0.1), "正在检索并生成答案")
            tick += 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=15)
            except TimeoutError:
                continue

    await _progress(1, "开始问答")
    heart = asyncio.create_task(_heartbeat())
    try:
        with _stdout_to_stderr():
            result = await asyncio.to_thread(run_rag_graph, question, knowledge_db)
    finally:
        stop.set()
        heart.cancel()

    await _progress(4, "完成")
    payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    payload["run_id"] = run_dir.name
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def run_pipeline(urls_path: str = "urls.txt", run_id: str | None = None) -> str:
    """Run the full disease-info pipeline (fetch → extract → validate → aggregate → report → ingest).

    urls_path is relative to disease_info_agent/ (default urls.txt). Optional run_id reuses a folder.
    This can take several minutes because it calls the LLM. Returns a JSON summary.
    """
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
    with _stdout_to_stderr():
        result = asyncio.run(run_pipeline_impl(args, _AGENT_ROOT))
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
def get_eval_report() -> str:
    """Return the latest B4 eval harness report (RAG faithfulness / relevancy + pipeline pytest summary) as JSON.

    Run first: python -m disease_info_agent.eval.run_harness
    """
    report_path = _AGENT_ROOT / "eval" / "output" / "report.json"
    if not report_path.exists():
        return json.dumps(
            {
                "error": "尚未运行 eval harness。请执行: uv run python -m disease_info_agent.eval.run_harness",
                "expected_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rag_summary = (payload.get("rag") or {}).get("summary") or {}
    return json.dumps(
        {
            "generated_at": payload.get("generated_at"),
            "run_id": payload.get("run_id"),
            "overall_passed": payload.get("overall_passed"),
            "pass_rate": rag_summary.get("pass_rate"),
            "avg_faithfulness": rag_summary.get("avg_faithfulness"),
            "avg_answer_relevancy": rag_summary.get("avg_answer_relevancy"),
            "hard_case_pass_rate": rag_summary.get("hard_case_pass_rate"),
            "ragas_status": rag_summary.get("ragas_status"),
            "avg_ragas_faithfulness": rag_summary.get("avg_ragas_faithfulness"),
            "pipeline_pytest_passed": (payload.get("pipeline_pytest") or {}).get("passed"),
            "report_json": str(report_path),
            "report_html": str(_AGENT_ROOT / "eval" / "output" / "report.html"),
        },
        ensure_ascii=False,
        indent=2,
    )


_start_warmup_thread()

if __name__ == "__main__":
    mcp.run()
