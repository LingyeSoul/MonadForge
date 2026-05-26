@echo off
setlocal

REM Add portable Node.js to PATH if system Node.js is missing
set "PORTABLE_NODE=%~dp0tools\node"
where node >nul 2>&1
if errorlevel 1 (
    if exist "%PORTABLE_NODE%\node.exe" (
        set "PATH=%PORTABLE_NODE%;%PATH%"
        echo Using portable Node.js from tools\node\
    ) else (
        echo [ERROR] Node.js not found. Run setup-win.bat first or install Node.js from https://nodejs.org/
        pause
        exit /b 1
    )
)

cd /d "%~dp0\webui\frontend"

if not exist node_modules (
    echo Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo npm install failed.
        pause
        exit /b 1
    )
)

echo Building frontend...
call npm run build
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)

echo Done. Output: webui\frontend\dist\
