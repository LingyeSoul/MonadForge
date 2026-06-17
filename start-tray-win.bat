@echo off
setlocal
cd /d "%~dp0"
echo Starting MonadForge tray...
REM pythonw.exe = no console window. The tray brings up the daemon itself
REM (ensure_daemon on startup), so no separate daemon launch is needed here.
start "" .venv\Scripts\pythonw.exe -m scripts.tray
