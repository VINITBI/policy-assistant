@echo off
REM ==============================================================
REM  Policy Assistant - ONE-TIME SETUP (Windows)
REM  Double-click this file. It takes 15-30 minutes the first time.
REM ==============================================================
cd /d "%~dp0"
title Policy Assistant - Setup

echo.
echo [1/6] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
    echo    ERROR: Python was not found. Install Python 3.11 from https://www.python.org/downloads/
    echo           and tick "Add python.exe to PATH" during installation. Then run this file again.
    pause & exit /b 1
)
python --version

echo.
echo [2/6] Checking Ollama...
where ollama >nul 2>nul
if errorlevel 1 (
    echo    ERROR: Ollama was not found. Install it from https://ollama.com and run this file again.
    pause & exit /b 1
)

echo.
echo [3/6] Creating the Python environment (.venv)...
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call ".venv\Scripts\activate.bat"

echo.
echo [4/6] Installing Python packages (this is the slow part)...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo    ERROR: package installation failed. Check your internet connection and run this file again.
    pause & exit /b 1
)

echo.
echo [5/6] Downloading the AI models into Ollama...
ollama pull llama3.1:8b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b

echo.
echo [6/6] Downloading the search models and checking everything...
python scripts\create_sample_policies.py
python scripts\warmup_models.py

echo.
choice /M "Load the 4 demo policies now so you can try the app straight away"
if errorlevel 2 goto done
python scripts\ingest_folder.py --folder data\sample_policies --department HR

:done
echo.
echo ==============================================================
echo  Setup complete. Now double-click run.bat to start the app.
echo ==============================================================
pause
