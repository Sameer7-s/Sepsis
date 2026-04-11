"""
server/app.py — OpenEnv server entrypoint.
The [project.scripts] `server` command resolves here.
Starts the FastAPI backend on 0.0.0.0:7860.
"""

import os
import signal
import socket
import sys
import time
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


def force_cleanup_port(port: int, max_retries: int = 3) -> None:
    """Attempt to forcefully clean up any processes using the port."""
    import subprocess
    import platform
    
    for attempt in range(max_retries):
        try:
            if platform.system() == "Windows":
                # On Windows, use netstat and taskkill
                result = subprocess.run(
                    f"netstat -ano | findstr :{port}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.stdout:
                    print(f"[DEBUG] Found process on port {port}: {result.stdout}")
                    # Extract PID and kill
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split()
                        if parts:
                            pid = parts[-1]
                            try:
                                subprocess.run(f"taskkill /PID {pid} /F", shell=True, timeout=2)
                                print(f"[INFO] Killed process {pid}")
                            except Exception as e:
                                print(f"[DEBUG] Could not kill {pid}: {e}")
            else:
                # On Linux/Mac, use lsof and kill
                result = subprocess.run(
                    f"lsof -i :{port}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.stdout:
                    print(f"[DEBUG] Found process on port {port}: {result.stdout}")
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if parts:
                            pid = parts[1]
                            try:
                                subprocess.run(f"kill -9 {pid}", shell=True, timeout=2)
                                print(f"[INFO] Killed process {pid}")
                            except Exception as e:
                                print(f"[DEBUG] Could not kill {pid}: {e}")
        except Exception as e:
            print(f"[DEBUG] Port cleanup attempt {attempt + 1} failed: {e}")
        
        time.sleep(0.5)


def wait_for_port_available(port: int, timeout: float = 5.0) -> bool:
    """Check if port is available, with retries."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, 'SO_REUSEPORT'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result != 0:  # Port is available
                print(f"[INFO] Port {port} is available")
                return True
            else:
                print(f"[DEBUG] Port {port} still in use, waiting...")
        except Exception as e:
            print(f"[DEBUG] Port check error: {e}")
        time.sleep(0.2)
    return False


def main() -> None:
    """Main entrypoint — called by `server` CLI command defined in pyproject.toml."""
    try:
        port_str = os.environ.get("PORT", "7860").strip()
        port = validate_port(port_str)
        
        print(f"[INFO] Checking port {port} availability...")
        
        # Try to clean up if port is in use
        if not wait_for_port_available(port, timeout=2.0):
            print(f"[WARNING] Port {port} appears to be in use, attempting cleanup...")
            force_cleanup_port(port)
            time.sleep(1)
            
            # Try one more time
            if not wait_for_port_available(port, timeout=2.0):
                print(f"[WARNING] Port {port} still in use, will attempt to start anyway with SO_REUSEADDR")
        
        print(f"[INFO] Starting server on 0.0.0.0:{port}")
        
        # Run uvicorn with proper configuration
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            server_header=False,
            lifespan="on",
        )
        server = uvicorn.Server(config)
        server.run()
        
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        if "Address already in use" in str(e) or "98" in str(e) or "48" in str(e):
            print(f"[ERROR] Port {port} is already in use by another process.", file=sys.stderr)
            print(f"[ERROR] Try waiting a moment or checking for zombie processes.", file=sys.stderr)
        print(f"[ERROR] OS error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("[INFO] Server interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
