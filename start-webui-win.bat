@echo off
setlocal
cd /d "%~dp0"
echo Starting MonadForge WebUI...
echo Access at http://127.0.0.1:8000
start /B "" .venv\Scripts\python.exe -m webui %*
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000
pause