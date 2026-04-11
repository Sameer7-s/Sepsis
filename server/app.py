"""
server/app.py — OpenEnv server entrypoint.
The [project.scripts] `server` command resolves here.
Starts the FastAPI backend on 0.0.0.0:7860.
"""

import os
import sys
import uvicorn

# Import the FastAPI app instance from the backend module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend_api import app  # noqa: E402


def validate_port(port_str: str) -> int:
    """Validate and convert port string to integer."""
    try:
        port = int(port_str)
        if port < 1 or port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {port}")
        return port
    except ValueError as e:
        raise ValueError(f"Invalid PORT value '{port_str}': {e}") from e


def main() -> None:
    """Main entrypoint — called by `server` CLI command defined in pyproject.toml."""
    try:
        port_str = os.environ.get("PORT", "7860").strip()
        port = validate_port(port_str)
        
        print(f"[INFO] Starting server on 0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
