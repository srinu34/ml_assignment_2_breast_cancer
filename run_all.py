"""Start the ML app and API together for local development."""

from __future__ import annotations

import subprocess
import sys
import time


def start_process(command: list[str], name: str) -> subprocess.Popen[str]:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"Started {name} with pid {proc.pid}")
    return proc


if __name__ == "__main__":
    app_proc = start_process(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"],
        "streamlit",
    )
    api_proc = start_process(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        "fastapi",
    )

    try:
        print("\nBoth services started. Open the app at http://localhost:8501 and the API docs at http://localhost:8000/docs")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        app_proc.terminate()
        api_proc.terminate()
        app_proc.wait(timeout=10)
        api_proc.wait(timeout=10)
        print("Stopped")
