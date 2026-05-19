@echo off
setlocal
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
