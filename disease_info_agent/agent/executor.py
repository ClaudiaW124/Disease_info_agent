"""Phase 2 Plan Executor — run TaskPlan steps via shared tool implementations."""

from __future__ import annotations

from typing import Any

from agent.plan_models import ExecutionResult, PlanStep, StepResult, TaskPlan
from agent.tool_impl import invoke_tool


class PlanExecutor:
    """Deterministic executor: runs planner steps in order."""

    async def execute(self, plan: TaskPlan) -> ExecutionResult:
        results: list[StepResult] = []
        for step in plan.steps:
            results.append(await self._run_step(step))
        return ExecutionResult(plan=plan, step_results=results)

    async def _run_step(self, step: PlanStep) -> StepResult:
        try:
            payload = await invoke_tool(step.tool, step.args)
            ok = "error" not in payload
            return StepResult(step=step, ok=ok, result=payload, error=payload.get("error") if not ok else None)
        except Exception as exc:  # noqa: BLE001
            return StepResult(step=step, ok=False, result={}, error=str(exc))


def format_execution_for_synthesis(execution: ExecutionResult) -> str:
    """Compact JSON-ish summary for the synthesizer prompt."""
    blocks: list[dict[str, Any]] = []
    for item in execution.step_results:
        blocks.append(
            {
                "step_id": item.step.id,
                "tool": item.step.tool,
                "description": item.step.description,
                "ok": item.ok,
                "error": item.error,
                "result": item.result,
            }
        )
    import json

    return json.dumps(blocks, ensure_ascii=False, indent=2)
