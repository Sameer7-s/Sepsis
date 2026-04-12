"""
server/app.py — canonical backend entrypoint.
"""

import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backend_api import app  # noqa: F401


def _port_bound(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        return probe.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False
    finally:
        probe.close()


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    if _port_bound(port):
        return

    uvicorn.run(
        "backend_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()