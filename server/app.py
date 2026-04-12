"""
server/app.py — OpenEnv server entrypoint.
The [project.scripts] `server` command resolves here.
Starts the FastAPI backend on 0.0.0.0:PORT (default 7860).

Usage:
  python server/app.py              # runs on 0.0.0.0:7860
  PORT=8000 python server/app.py   # runs on 0.0.0.0:8000

This is the ONLY place where uvicorn.run() should be called.
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
    """Main entrypoint — starts the FastAPI server."""
    try:
        # Get port from environment, default to 7860
        port_str = os.environ.get("PORT", "7860").strip()
        port = validate_port(port_str)

        print(f"[INFO] Starting FastAPI server on 0.0.0.0:{port}")

        # Simple, clean uvicorn startup
        # No retry logic, port cleanup, or pre-checks - these cause more problems than they solve
        # in containerized environments (HuggingFace Spaces, Docker, etc.)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
        )

    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        if "Address already in use" in str(e) or "98" in str(e) or "48" in str(e):
            print(
                f"[ERROR] Port {port} is already in use by another process.",
                file=sys.stderr,
            )
            print(
                f"[ERROR] Set PORT environment variable to use a different port.",
                file=sys.stderr,
            )
        else:
            print(f"[ERROR] OS error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("[INFO] Server shutdown by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
