"""Structured plan models for Phase 2 Planner → Executor flow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class TaskPlan(BaseModel):
    mode: Literal["direct", "plan"] = "direct"
    goal: str = ""
    reason: str = ""
    steps: list[PlanStep] = Field(default_factory=list)


class StepResult(BaseModel):
    step: PlanStep
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ExecutionResult(BaseModel):
    plan: TaskPlan
    step_results: list[StepResult] = Field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(item.ok for item in self.step_results)


class ChatResult(BaseModel):
    reply: str
    mode: Literal["direct", "plan"] = "direct"
    plan: TaskPlan | None = None
    execution: ExecutionResult | None = None
    trace_id: str | None = None
    trace: dict | None = None
