#!/usr/bin/env bash
# Policy Assistant - START THE APP (macOS / Linux).  Run:  bash run.sh   (Ctrl+C to stop)
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run  bash setup.sh  first."; exit 1; }
source .venv/bin/activate
echo "Opening http://localhost:8501   (admin / Admin@123  ·  employee / Employee@123)"
echo "Swagger: http://localhost:8000/docs"
python scripts/start_all.py
