"""
server/app.py — OpenEnv server entrypoint.
The [project.scripts] `server` command resolves here.
Starts the FastAPI backend on 0.0.0.0:7860.
"""

import os
import uvicorn

# Import the FastAPI app instance from the backend module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend_api import app  # noqa: E402


def main() -> None:
    """Main entrypoint — called by `server` CLI command defined in pyproject.toml."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
