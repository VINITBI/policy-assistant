@echo off
REM ==============================================================
REM  Policy Assistant - START THE APP (Windows)
REM  Double-click this file every time you want to use the app.
REM  Keep this window open while you use it; close it to stop.
REM ==============================================================
cd /d "%~dp0"
title Policy Assistant - Running

if not exist ".venv\Scripts\python.exe" (
    echo Setup has not been run yet. Double-click setup.bat first.
    pause & exit /b 1
)
call ".venv\Scripts\activate.bat"

echo Starting... your browser will open at http://localhost:8501
echo   Admin login   : admin / Admin@123
echo   Employee login: employee / Employee@123
echo   API + Swagger : http://localhost:8000/docs
echo.
python scripts\start_all.py
pause
