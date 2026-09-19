#!/usr/bin/env bash
# Policy Assistant - ONE-TIME SETUP (macOS / Linux).  Run:  bash setup.sh
set -e
cd "$(dirname "$0")"

command -v python3 >/dev/null || { echo "ERROR: install Python 3.11 first."; exit 1; }
command -v ollama  >/dev/null || { echo "ERROR: install Ollama from https://ollama.com first."; exit 1; }

echo "[1/4] Creating the Python environment (.venv)..."
[ -x .venv/bin/python ] || python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Installing Python packages (slow the first time)..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[3/4] Downloading the AI models into Ollama..."
ollama pull llama3.1:8b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b

echo "[4/4] Downloading the search models and checking everything..."
python scripts/create_sample_policies.py
python scripts/warmup_models.py

read -r -p "Load the 4 demo policies now? [y/N] " ans
if [[ "$ans" =~ ^[Yy] ]]; then
  python scripts/ingest_folder.py --folder data/sample_policies --department HR
fi
echo "Setup complete. Start the app with:  bash run.sh"
