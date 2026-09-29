@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment is missing. Follow the setup steps in README.md first.
    exit /b 1
)
set PYTHONPATH=src
".venv\Scripts\python.exe" scripts\train_model.py
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m uvicorn retentioniq.api:app --host 127.0.0.1 --port 8000