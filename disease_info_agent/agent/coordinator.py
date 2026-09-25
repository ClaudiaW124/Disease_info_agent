"""Phase 2 Coordinator — Planner → Executor → Synthesizer, with Orchestrator fallback."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from utu.agents import SimpleAgent

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.executor import PlanExecutor, format_execution_for_synthesis  # noqa: E402
from agent.orchestrator import DiseaseInfoOrchestrator  # noqa: E402
from agent.plan_models import ChatResult, TaskPlan  # noqa: E402
from agent.planner import DiseaseInfoPlanner  # noqa: E402
from agent.run_trace_buffer import finish_trace, start_trace  # noqa: E402
from agent.trace_store import save_trace  # noqa: E402


class DiseaseInfoSynthesizer:
    def __init__(self) -> None:
        self._agent: SimpleAgent | None = None

    async def load(self) -> None:
        model_name = os.getenv("RAG_LLM_MODEL") or os.getenv("UTU_LLM_MODEL")
        self._agent = SimpleAgent(config="disease_info/synthesizer", model=model_name)
        await self._agent.build()

    async def cleanup(self) -> None:
        if self._agent and self._agent._initialized:
            await self._agent.cleanup()

    async def synthesize(self, user_message: str, plan: TaskPlan, execution_json: str) -> str:
        if not self._agent:
            await self.load()
        assert self._agent is not None
        prompt = (
            f"## 用户问题\n{user_message.strip()}\n\n"
            f"## 计划目标\n{plan.goal}\n\n"
            f"## 计划步骤\n"
            + "\n".join(f"{s.id}. [{s.tool}] {s.description or s.args}" for s in plan.steps)
            + f"\n\n## 各步执行结果（JSON）\n{execution_json}\n"
        )
        recorder = await self._agent.run(prompt, save=True)
        return str(recorder.get_run_result().final_output)


class DiseaseInfoCoordinator:
    """Top-level Phase 2 entry: plan multi-step tasks or delegate to Orchestrator."""

    def __init__(self, project_root: Path | None = None, *, enable_planner: bool = True) -> None:
        self.project_root = project_root or _AGENT_ROOT
        self.enable_planner = enable_planner
        self._planner: DiseaseInfoPlanner | None = None
        self._executor = PlanExecutor()
        self._synthesizer: DiseaseInfoSynthesizer | None = None
        self._orchestrator: DiseaseInfoOrchestrator | None = None

    async def load(self) -> None:
        self._orchestrator = DiseaseInfoOrchestrator(self.project_root)
        await self._orchestrator.load()
        if self.enable_planner:
            self._planner = DiseaseInfoPlanner()
            await self._planner.load()
            self._synthesizer = DiseaseInfoSynthesizer()
            await self._synthesizer.load()

    async def cleanup(self) -> None:
        if self._orchestrator:
            await self._orchestrator.cleanup()
        if self._planner:
            await self._planner.cleanup()
        if self._synthesizer:
            await self._synthesizer.cleanup()

    async def reset_history(self) -> None:
        if self._orchestrator:
            await self._orchestrator.reset_history()

    async def chat(
        self,
        message: str,
        *,
        no_plan: bool = False,
        force_plan: bool = False,
        save_trace_file: bool = True,
    ) -> ChatResult:
        if not self._orchestrator:
            await self.load()
        assert self._orchestrator is not None

        trace_ctx = start_trace(user_message=message.strip())

        try:
            use_planner = self.enable_planner and not no_plan and self._planner is not None
            if not use_planner:
                reply = await self._orchestrator.chat(message)
                result = ChatResult(reply=reply, mode="direct")
            else:
                assert self._planner is not None
                assert self._synthesizer is not None

                plan = await self._planner.create_plan(message)
                if plan.mode == "direct" and not force_plan:
                    reply = await self._orchestrator.chat(message)
                    result = ChatResult(reply=reply, mode="direct", plan=plan)
                elif not plan.steps:
                    reply = await self._orchestrator.chat(message)
                    result = ChatResult(reply=reply, mode="direct", plan=plan)
                else:
                    execution = await self._executor.execute(plan)
                    execution_json = format_execution_for_synthesis(execution)
                    reply = await self._synthesizer.synthesize(message, plan, execution_json)
                    result = ChatResult(reply=reply, mode="plan", plan=plan, execution=execution)

            from agent.trace import build_planner_trace

            full_trace = build_planner_trace(
                result,
                events=list(trace_ctx.get("events") or []),
                trace_id=trace_ctx.get("trace_id"),
            )
            trace_payload = full_trace.model_dump()
            trace_payload["reply_preview"] = result.reply[:500]

            if save_trace_file:
                save_trace(trace_payload)

            return ChatResult(
                reply=result.reply,
                mode=result.mode,
                plan=result.plan,
                execution=result.execution,
                trace_id=full_trace.trace_id,
                trace=trace_payload,
            )
        finally:
            finish_trace()

    async def __aenter__(self) -> DiseaseInfoCoordinator:
        await self.load()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.cleanup()


async def run_chat_loop(project_root: Path | None = None, *, no_plan: bool = False) -> None:
    print("=" * 50)
    print("  Disease Info Coordinator (Planner + Orchestrator)")
    print("  输入 exit / quit 退出 · /reset 清空历史 · /noplan 切换直连模式")
    print("=" * 50)
    planner_on = not no_plan
    async with DiseaseInfoCoordinator(project_root, enable_planner=planner_on) as coordinator:
        while True:
            try:
                user_input = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("再见。")
                break
            if user_input.lower() == "/reset":
                await coordinator.reset_history()
                print("（对话历史已清空）")
                continue
            if user_input.lower() == "/noplan":
                planner_on = not planner_on
                coordinator.enable_planner = planner_on
                print(f"Planner {'开启' if planner_on else '关闭'}（Orchestrator 直连）")
                continue

            print("\nAgent> ", end="", flush=True)
            result = await coordinator.chat(user_input, no_plan=not planner_on)
            if result.mode == "plan" and result.plan:
                steps = " → ".join(f"{s.tool}" for s in result.plan.steps)
                print(f"[Planner: {steps}]\n", end="")
            print(result.reply)
