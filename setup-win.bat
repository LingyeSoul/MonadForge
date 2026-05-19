@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   MonadForge - Windows One-Click Setup
echo   Forked from https://github.com/sorryhyun/anima_lora
echo   Python 3.13 / PyTorch 2.11 + CUDA 13.0
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM 0. Check Python
REM ---------------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.13 from https://www.python.org/downloads/
    echo         and make sure "Add to PATH" is checked.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [INFO] Python version: %PY_VER%

REM ---------------------------------------------------------------
REM 1. Install / update uv
REM ---------------------------------------------------------------
where uv >nul 2>&1
if errorlevel 1 (
    echo [STEP 1/2] Installing uv ...
    pip install -U uv
) else (
    echo [STEP 1/2] uv found, upgrading ...
    pip install -U uv
)

REM ---------------------------------------------------------------
REM 2. Sync project dependencies (torch, triton, flash-attn, etc.)
REM ---------------------------------------------------------------
echo.
echo [STEP 2/2] Syncing project dependencies via uv ...
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM Done
REM ---------------------------------------------------------------
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Quick start:
echo     python tasks.py lora          - train LoRA
echo     python tasks.py test          - inference test
echo     python tasks.py gui           - launch GUI
echo     python tasks.py --help        - all commands
echo.
echo   Model downloads:
echo     python tasks.py download-models
echo ============================================================
pause
