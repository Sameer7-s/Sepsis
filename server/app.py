"""
server/app.py — canonical backend entrypoint.

Validator multi-mode deployment requirements (BOTH must pass):
  ✓  A callable  def main()  function must exist in this file
  ✓  Must be guarded with:  if __name__ == '__main__': main()

The platform can start this file in two ways:
  1. Docker ENTRYPOINT → python -u server/app.py  → __main__ → main()
  2. openenv validator → imports server.app; calls server.app.main() directly

main() is fully idempotent: if another process (e.g. the Docker container's
ENTRYPOINT) has already bound port 7860, main() exits cleanly instead of
crashing with [Errno 98] EADDRINUSE.
"""

import os
import socket
import sys

# Allow 'from backend_api import app' when this file lives in server/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backend_api import app  # noqa: F401  — re-exported so HF can discover it


# ---------------------------------------------------------------------------
# Port probe helper
# ---------------------------------------------------------------------------

def _port_bound(host: str, port: int) -> bool:
    """Return True if something is already listening on host:port."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        return probe.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False
    finally:
        probe.close()


# ---------------------------------------------------------------------------
# main() — REQUIRED by the openenv validator
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Start the uvicorn server on PORT (default 7860).

    This function MUST:
      - be named exactly  main
      - be defined at module level (not nested)
      - be callable without arguments
      - be guarded by  if __name__ == '__main__': main()  below

    Idempotency: if port is already bound (Docker ENTRYPOINT already started
    the server), this function logs a notice and returns cleanly so the
    validator's second call does not cause EADDRINUSE.
    """
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    if _port_bound(host, port):
        print(
            f"[INFO] server/app.main(): port {port} already occupied — "
            "backend already running, skipping duplicate startup."
        )
        return  # ← clean exit, no EADDRINUSE crash

    uvicorn.run(
        "backend_api:app",   # string import — prevents accidental double-import
        host=host,
        port=port,
        reload=False,        # NEVER True — reload spawns a second watcher process
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Required guard — validator checks this exists
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
