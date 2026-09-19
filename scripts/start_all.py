"""Start API (FastAPI :8000) and UI (Streamlit :8501) together and open the browser.

    python scripts/start_all.py        (Ctrl+C or close the window to stop)
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API, UI = "http://localhost:8000", "http://localhost:8501"


def wait_for(url: str, seconds: int = 90) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


if __name__ == "__main__":
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
                           cwd=ROOT)
    if not wait_for(f"{API}/health"):
        print("ERROR: the API did not start. Read the messages above (and logs/error.log).")
        api.terminate()
        sys.exit(1)
    # headless = no first-run e-mail prompt from Streamlit; we open the browser ourselves
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", "8501",
                           "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT)
    if wait_for(UI, 60):
        webbrowser.open(UI)
    print(f"\n  App     -> {UI}\n  Swagger -> {API}/docs\n  Health  -> {API}/ready\n  (Ctrl+C or close this window to stop)\n")
    try:
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            p.terminate()
