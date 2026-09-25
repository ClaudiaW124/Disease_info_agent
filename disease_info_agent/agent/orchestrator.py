"""Phase 1 Orchestrator Agent — top-level coordinator with tool calling."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from utu.agents import SimpleAgent

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.tools import ORCHESTRATOR_TOOLS  # noqa: E402


class DiseaseInfoOrchestrator:
    """Top-level agent: routes user intent to RAG / pipeline / eval tools."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or _AGENT_ROOT
        self._agent: SimpleAgent | None = None

    async def load(self) -> None:
        model_name = os.getenv("RAG_LLM_MODEL") or os.getenv("UTU_LLM_MODEL")
        self._agent = SimpleAgent(
            config="disease_info/orchestrator",
            model=model_name,
            tools=ORCHESTRATOR_TOOLS,
        )
        await self._agent.build()

    async def cleanup(self) -> None:
        if self._agent and self._agent._initialized:
            await self._agent.cleanup()

    async def chat(self, message: str) -> str:
        if not self._agent:
            await self.load()
        assert self._agent is not None
        recorder = await self._agent.run(message, save=True)
        return str(recorder.get_run_result().final_output)

    async def reset_history(self) -> None:
        if self._agent:
            self._agent.clear_input_items()

    async def __aenter__(self) -> DiseaseInfoOrchestrator:
        await self.load()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.cleanup()


async def run_chat_loop(project_root: Path | None = None, *, no_plan: bool = False) -> None:
    """Interactive multi-turn chat — delegates to Phase 2 Coordinator."""
    from agent.coordinator import run_chat_loop as coordinator_loop

    await coordinator_loop(project_root, no_plan=no_plan)
