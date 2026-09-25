"""Build planner execution traces for API / Web visualization (Phase 3)."""

from __future__ import annotations

from typing import Any, Literal

from agent.plan_models import ChatResult, StepResult
from pydantic import BaseModel, Field

RAG_NODE_LABELS = {
    "guard": "库内守卫",
    "retrieve": "向量检索",
    "grade": "相关度评分",
    "rewrite": "问题改写",
    "generate": "生成答案",
    "validate": "答案校验",
    "refuse": "拒答",
}


class RagStepView(BaseModel):
    raw: str
    node: str
    label: str
    status: Literal["ok", "warn", "error"] = "ok"


class TraceStepView(BaseModel):
    id: int
    tool: str
    description: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "pending"] = "pending"
    summary: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    rag_steps: list[RagStepView] = Field(default_factory=list)


class FlowNodeView(BaseModel):
    id: str
    label: str
    kind: str = "agent"


class FlowEdgeView(BaseModel):
    source: str
    target: str


class PlannerTraceView(BaseModel):
    mode: Literal["direct", "plan"] = "direct"
    goal: str = ""
    reason: str = ""
    phases: list[str] = Field(default_factory=list)
    steps: list[TraceStepView] = Field(default_factory=list)
    flowchart: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str | None = None


def parse_rag_steps(steps: list[str] | None) -> list[RagStepView]:
    views: list[RagStepView] = []
    for raw in steps or []:
        node = raw.split("(")[0].split("=")[0].strip()
        if "->" in raw:
            node = "rewrite"
        label = RAG_NODE_LABELS.get(node, node)
        status: Literal["ok", "warn", "error"] = "ok"
        lowered = raw.lower()
        if "refuse" in lowered or "unrelated" in lowered or "out_of_kb" in lowered:
            status = "error"
        elif "grade=no" in lowered or "validate=no" in lowered:
            status = "warn"
        views.append(RagStepView(raw=raw, node=node, label=label, status=status))
    return views


def build_flowchart(mode: str, steps: list[TraceStepView]) -> dict[str, Any]:
    nodes = [
        FlowNodeView(id="user", label="用户", kind="user"),
        FlowNodeView(id="coordinator", label="Coordinator", kind="agent"),
        FlowNodeView(id="planner", label="Planner", kind="agent"),
    ]
    edges = [
        FlowEdgeView(source="user", target="coordinator"),
        FlowEdgeView(source="coordinator", target="planner"),
    ]

    if mode == "direct":
        nodes.append(FlowNodeView(id="orchestrator", label="Orchestrator", kind="agent"))
        edges.append(FlowEdgeView(source="planner", target="orchestrator"))
        nodes.extend(
            [
                FlowNodeView(id="rag", label="LangGraph RAG", kind="pipeline"),
                FlowNodeView(id="validate", label="Validate", kind="pipeline"),
                FlowNodeView(id="reply", label="回复", kind="output"),
            ]
        )
        edges.extend(
            [
                FlowEdgeView(source="orchestrator", target="rag"),
                FlowEdgeView(source="rag", target="validate"),
                FlowEdgeView(source="validate", target="reply"),
            ]
        )
    else:
        nodes.append(FlowNodeView(id="executor", label="Executor", kind="agent"))
        edges.append(FlowEdgeView(source="planner", target="executor"))
        for step in steps:
            node_id = f"tool_{step.id}"
            nodes.append(FlowNodeView(id=node_id, label=step.tool, kind="tool"))
            edges.append(FlowEdgeView(source="executor", target=node_id))
            if step.tool == "query_disease" and step.rag_steps:
                rag_id = f"rag_{step.id}"
                val_id = f"validate_{step.id}"
                nodes.append(FlowNodeView(id=rag_id, label="RAG", kind="pipeline"))
                nodes.append(FlowNodeView(id=val_id, label="Validate", kind="pipeline"))
                edges.extend(
                    [
                        FlowEdgeView(source=node_id, target=rag_id),
                        FlowEdgeView(source=rag_id, target=val_id),
                    ]
                )
        nodes.append(FlowNodeView(id="synthesizer", label="Synthesizer", kind="agent"))
        nodes.append(FlowNodeView(id="reply", label="回复", kind="output"))
        if steps:
            edges.append(FlowEdgeView(source=f"tool_{steps[-1].id}", target="synthesizer"))
        else:
            edges.append(FlowEdgeView(source="executor", target="synthesizer"))
        edges.append(FlowEdgeView(source="synthesizer", target="reply"))

    return {
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }


def _summarize_tool_result(tool: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any], list[RagStepView]]:
    rag_steps = parse_rag_steps(payload.get("steps"))

    if payload.get("error"):
        return f"错误: {payload['error']}", {"error": payload["error"]}, rag_steps

    if tool == "query_disease":
        answer = str(payload.get("answer") or "")
        citations = payload.get("citations") or []
        refused = bool(payload.get("refused"))
        score = float(payload.get("relevance_score") or 0.0)
        label = "已拒答" if refused else "已回答"
        expand_note = ""
        if payload.get("kb_expand", {}).get("expanded"):
            expand_note = " · 已自动扩展 KB"
        elif payload.get("kb_expand_attempt"):
            expand_note = " · 尝试过 KB 扩展"
        summary = f"{label} · 相关度 {score:.2f} · {len(citations)} 个来源{expand_note}"
        return summary, {
            "refused": refused,
            "relevance_score": score,
            "validate_passed": payload.get("validate_passed"),
            "answer_preview": answer[:280] + ("…" if len(answer) > 280 else ""),
            "citations": citations[:8],
            "run_id": payload.get("run_id"),
            "kb_expand": payload.get("kb_expand") or payload.get("kb_expand_attempt"),
        }, rag_steps

    if tool == "expand_knowledge_base":
        expanded = bool(payload.get("expanded"))
        summary = payload.get("message") or ("已扩展" if expanded else "未扩展")
        return summary, payload, rag_steps

    if tool == "get_latest_run":
        docs = payload.get("knowledge_db_document_count")
        facts = payload.get("validated_fact_count")
        summary = f"run={payload.get('run_id')} · {docs} 篇文档 · {facts} 条 facts"
        return summary, {
            "run_id": payload.get("run_id"),
            "status": payload.get("status"),
            "knowledge_db_document_count": docs,
            "validated_fact_count": facts,
            "has_knowledge_db": payload.get("has_knowledge_db"),
        }, rag_steps

    if tool == "get_eval_report":
        rate = payload.get("pass_rate")
        summary = f"pass_rate={rate}" if rate is not None else "评测报告已读取"
        return summary, {
            "pass_rate": rate,
            "hard_case_pass_rate": payload.get("hard_case_pass_rate"),
            "avg_faithfulness": payload.get("avg_faithfulness"),
            "refusal_accuracy": payload.get("refusal_accuracy"),
            "report_html": payload.get("report_html"),
        }, rag_steps

    if tool == "start_pipeline":
        summary = f"Pipeline {payload.get('status')} · run={payload.get('run_id')}"
        return summary, {
            "run_id": payload.get("run_id"),
            "status": payload.get("status"),
            "message": payload.get("message"),
            "output_dir": payload.get("output_dir"),
        }, rag_steps

    if tool == "get_pipeline_status":
        summary = f"任务状态: {payload.get('status') or payload.get('message') or 'unknown'}"
        return summary, payload, rag_steps

    preview = str(payload)[:120]
    return preview, payload, rag_steps


def _step_from_result(step_result: StepResult) -> TraceStepView:
    step = step_result.step
    summary, detail, rag_steps = _summarize_tool_result(step.tool, step_result.result)
    return TraceStepView(
        id=step.id,
        tool=step.tool,
        description=step.description,
        args=step.args,
        status="ok" if step_result.ok else "error",
        summary=summary,
        detail=detail,
        error=step_result.error,
        rag_steps=rag_steps,
    )


def build_planner_trace(
    result: ChatResult,
    *,
    events: list[dict[str, Any]] | None = None,
    trace_id: str | None = None,
) -> PlannerTraceView:
    plan = result.plan
    goal = plan.goal if plan else ""
    reason = plan.reason if plan else ""

    if result.mode == "direct":
        trace = PlannerTraceView(
            mode="direct",
            goal=goal,
            reason=reason or "单一意图，无需多步计划",
            phases=["Planner", "Orchestrator", "RAG", "Validate"],
            steps=[],
            events=events or [],
            trace_id=trace_id,
        )
        trace.flowchart = build_flowchart("direct", [])
        return trace

    phases = ["Planner", "Executor", "RAG", "Validate", "Synthesizer"]
    steps: list[TraceStepView] = []

    if result.execution:
        steps = [_step_from_result(item) for item in result.execution.step_results]
    elif plan:
        for step in plan.steps:
            steps.append(
                TraceStepView(
                    id=step.id,
                    tool=step.tool,
                    description=step.description,
                    args=step.args,
                    status="pending",
                    summary="已规划，未执行",
                )
            )

    trace = PlannerTraceView(
        mode="plan",
        goal=goal,
        reason=reason,
        phases=phases,
        steps=steps,
        events=events or [],
        trace_id=trace_id,
    )
    trace.flowchart = build_flowchart("plan", steps)
    return trace
