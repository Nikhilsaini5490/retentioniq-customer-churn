@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment is missing. Follow the setup steps in README.md first.
    exit /b 1
)
".venv\Scripts\python.exe" -m pytest -q