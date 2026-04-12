"""
server/app.py — single, canonical server entrypoint.

On Hugging Face Spaces the platform starts this file as the main process.
Locally you can run:  python server/app.py

NEVER import this file from inference.py or anywhere else.
"""

import os
import sys

# Allow `from backend_api import app` when this file lives in server/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backend_api import app  # noqa: F401  — re-exported so HF can discover it

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "backend_api:app",   # string form — no accidental double-import
        host=host,
        port=port,
        reload=False,        # reload=True would start a second watcher process
        log_level="info",
    )
