@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   MonadForge - Windows One-Click Setup
echo   Forked from https://github.com/sorryhyun/anima_lora
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
    echo [STEP 1/4] Installing uv ...
    pip install -U uv
) else (
    echo [STEP 1/4] uv found, upgrading ...
    pip install -U uv
)

REM ---------------------------------------------------------------
REM 2. Sync project dependencies (torch, triton, flash-attn, etc.)
REM ---------------------------------------------------------------
echo.
echo [STEP 2/4] Syncing project dependencies via uv ...

REM Best-effort Defender exclusions so the install dir + uv cache are skipped by
REM real-time scanning. uv populates .venv\Scripts\ with unsigned trampoline
REM .exe launchers that Defender may quarantine mid-write.
set "UV_CACHE=%LOCALAPPDATA%\uv"
powershell -NoProfile -Command ^
    "try { Add-MpPreference -ExclusionPath '%~dp0','%UV_CACHE%' -ErrorAction Stop; Write-Host '  [INFO] Added Windows Defender exclusions.' } catch { Write-Host '  [INFO] Defender exclusion skipped (not elevated or Defender inactive).' }"

set "SYNC_OK=0"
for /L %%i in (1,1,3) do (
    if "!SYNC_OK!"=="0" (
        uv sync
        if not errorlevel 1 (
            set "SYNC_OK=1"
        ) else (
            if %%i lss 3 (
                echo [WARN] uv sync failed. Retrying (attempt %%i/2) in 3s ...
                timeout /t 3 /nobreak >nul
            )
        )
    )
)
if "!SYNC_OK!"=="0" (
    echo.
    echo [ERROR] uv sync did not complete after 3 attempts.
    echo         This is often caused by Windows Defender blocking uv's trampoline .exe files.
    echo         Add folder exclusions in Windows Security for:
    echo           %~dp0
    echo           %UV_CACHE%
    echo         Then re-run this script.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 3. Ensure portable Node.js (for WebUI frontend build)
REM ---------------------------------------------------------------
echo.
set "PORTABLE_NODE=%~dp0tools\node"

REM Check system Node.js first
where node >nul 2>&1
if not errorlevel 1 (
    echo [STEP 3/4] Node.js found in PATH, skipping portable setup.
    goto :build_frontend
)

REM System Node.js not found — check portable
if exist "%PORTABLE_NODE%\node.exe" (
    echo [STEP 3/4] Using portable Node.js from tools\node\
    set "PATH=%PORTABLE_NODE%;%PATH%"
    goto :build_frontend
)

REM Neither found — download portable
echo [STEP 3/4] Node.js not found. Setting up portable Node.js ...
set "NODE_ZIP_URL=https://nodejs.org/dist/v24.16.0/node-v24.16.0-win-x64.zip"
set "NODE_ZIP=%~dp0node.zip"
set "NODE_EXTRACT_DIR=%~dp0tools\node-v24.16.0-win-x64"

if not exist "%~dp0tools" mkdir "%~dp0tools"

echo        Downloading from %NODE_ZIP_URL% ...
powershell -NoProfile -Command ^
    "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%NODE_ZIP_URL%' -OutFile '%NODE_ZIP%'"
if errorlevel 1 (
    echo [ERROR] Failed to download Node.js.
    echo        Please install Node.js manually from https://nodejs.org/
    pause
    exit /b 1
)

echo        Extracting ...
powershell -NoProfile -Command ^
    "Expand-Archive -Path '%NODE_ZIP%' -DestinationPath '%~dp0tools' -Force"
if errorlevel 1 (
    echo [ERROR] Failed to extract Node.js zip.
    pause
    exit /b 1
)

move "%NODE_EXTRACT_DIR%" "%PORTABLE_NODE%" >nul 2>&1
del "%NODE_ZIP%" >nul 2>&1

if not exist "%PORTABLE_NODE%\node.exe" (
    echo [ERROR] Portable Node.js setup failed. node.exe not found after extraction.
    echo        Please install Node.js manually from https://nodejs.org/
    pause
    exit /b 1
)

set "PATH=%PORTABLE_NODE%;%PATH%"
for /f "tokens=*" %%v in ('node --version 2^>^&1') do set NODE_VER=%%v
echo        Portable Node.js !NODE_VER! installed to tools\node\

REM ---------------------------------------------------------------
REM 4. Build WebUI frontend
REM ---------------------------------------------------------------
:build_frontend
echo.
echo [STEP 4/4] Building WebUI frontend ...
call "%~dp0build-webui-win.bat"
if errorlevel 1 (
    echo [ERROR] WebUI frontend build failed.
    pause
    exit /b 1
)
echo [STEP 4/4] WebUI frontend built successfully.

REM ---------------------------------------------------------------
REM Done
REM ---------------------------------------------------------------
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Quick start:
echo     start-webui-win.bat          - launch WebUI (browser)
echo     python tasks.py lora         - train LoRA
echo     python tasks.py test         - inference test
echo     python tasks.py --help       - all commands
echo.
echo   Model downloads:
echo     python tasks.py download-models
echo ============================================================
pause
