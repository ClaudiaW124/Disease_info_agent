"""Phase 2 Task Planner — decompose complex user goals into executable steps."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from utu.agents import SimpleAgent

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.plan_models import PlanStep, TaskPlan  # noqa: E402
from agent.tool_impl import TOOL_NAMES  # noqa: E402


def _extract_json_blob(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("empty planner output")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("no JSON object found in planner output")


def parse_task_plan(raw: str, *, fallback_goal: str = "") -> TaskPlan:
    payload = _extract_json_blob(raw)
    mode = payload.get("mode", "direct")
    if mode not in {"direct", "plan"}:
        mode = "plan" if payload.get("steps") else "direct"

    steps: list[PlanStep] = []
    for item in payload.get("steps") or []:
        tool = str(item.get("tool", "")).strip()
        if tool not in TOOL_NAMES:
            continue
        steps.append(
            PlanStep(
                id=int(item.get("id") or len(steps) + 1),
                tool=tool,
                args=dict(item.get("args") or {}),
                description=str(item.get("description") or ""),
            )
        )

    if mode == "plan" and len(steps) < 2:
        mode = "direct"
        steps = []

    return TaskPlan(
        mode=mode,  # type: ignore[arg-type]
        goal=str(payload.get("goal") or fallback_goal),
        reason=str(payload.get("reason") or ""),
        steps=steps,
    )


class DiseaseInfoPlanner:
    """LLM planner that emits structured TaskPlan JSON."""

    def __init__(self) -> None:
        self._agent: SimpleAgent | None = None

    async def load(self) -> None:
        model_name = os.getenv("RAG_LLM_MODEL") or os.getenv("UTU_LLM_MODEL")
        self._agent = SimpleAgent(
            config="disease_info/task_planner",
            model=model_name,
        )
        await self._agent.build()

    async def cleanup(self) -> None:
        if self._agent and self._agent._initialized:
            await self._agent.cleanup()

    async def create_plan(self, user_message: str) -> TaskPlan:
        if not self._agent:
            await self.load()
        assert self._agent is not None
        recorder = await self._agent.run(user_message.strip(), save=True)
        raw = str(recorder.get_run_result().final_output)
        try:
            return parse_task_plan(raw, fallback_goal=user_message.strip())
        except (json.JSONDecodeError, ValueError):
            return TaskPlan(mode="direct", goal=user_message.strip(), reason="planner_parse_fallback", steps=[])

    async def __aenter__(self) -> DiseaseInfoPlanner:
        await self.load()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.cleanup()
