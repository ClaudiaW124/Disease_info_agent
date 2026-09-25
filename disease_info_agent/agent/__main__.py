"""CLI: uv run python -m disease_info_agent.agent"""

import asyncio
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.orchestrator import run_chat_loop

if __name__ == "__main__":
    asyncio.run(run_chat_loop())
