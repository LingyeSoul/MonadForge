@echo off
setlocal
cd /d "%~dp0"
echo Starting MonadForge (daemon + WebUI + tray)...
echo WebUI at http://127.0.0.1:8000  (daemon brings it up)
REM The tray also ensures the daemon is up; it can run alongside safely.
start "" .venv\Scripts\pythonw.exe -m scripts.tray
REM Launch the daemon in the foreground — it spawns the WebUI as a
REM supervised sidecar and owns it until the daemon stops.
.venv\Scripts\python.exe tasks.py daemon %*
timeout /t 4 /nobreak >nul
start http://127.0.0.1:8000
