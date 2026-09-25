"""Launch FastAPI: uv run python -m disease_info_agent.api"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "disease_info_agent.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
