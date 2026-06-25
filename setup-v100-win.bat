@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   MonadForge - V100专用安装脚本 (Windows)
echo   使用 torch==2.10.0+cu129，不安装 flash-attn
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM 0. 检查Python
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
REM 1. 安装 / 更新 uv
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
REM 2. 创建V100专用虚拟环境
REM ---------------------------------------------------------------
echo.
echo [STEP 2/4] Creating virtual environment (.venv) ...

if exist ".venv" (
    echo [INFO] .venv already exists, skipping creation.
) else (
    uv venv .venv --python 3.13
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Created .venv with Python 3.13.
)

REM ---------------------------------------------------------------
REM 3. 安装V100兼容依赖
REM ---------------------------------------------------------------
echo.
echo [STEP 3/4] Installing V100-compatible dependencies ...

REM 激活虚拟环境
call .venv\Scripts\activate.bat

REM 配置uv使用PyTorch CUDA 12.9索引
set "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu129"
set "UV_INDEX_STRATEGY=unsafe-best-match"

REM 安装torch和torchvision（V100兼容版本）
echo [INFO] Installing torch==2.10.0+cu129 and torchvision==0.21.0+cu129 ...
uv pip install torch==2.10.0+cu129 torchvision==0.21.0+cu129 --index-url https://download.pytorch.org/whl/cu129
if errorlevel 1 (
    echo [ERROR] Failed to install torch.
    pause
    exit /b 1
)

REM 安装其他依赖（不包含flash-attn）
echo [INFO] Installing other dependencies (excluding flash-attn) ...
uv pip install -r requirements-v100.txt --index-strategy unsafe-best-match
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 4. 构建WebUI前端
REM ---------------------------------------------------------------
echo.
echo [STEP 4/4] Building WebUI frontend ...

REM 检查Node.js
where node >nul 2>&1
if errorlevel 1 (
    echo [WARN] Node.js not found. WebUI frontend will not be built.
    echo        Install Node.js from https://nodejs.org/ and run build-webui-win.bat manually.
    goto :done
)

call build-webui-win.bat
if errorlevel 1 (
    echo [WARN] WebUI frontend build failed. You can build it later with build-webui-win.bat.
) else (
    echo [INFO] WebUI frontend built successfully.
)

REM ---------------------------------------------------------------
REM 完成
REM ---------------------------------------------------------------
:done
echo.
echo ============================================================
echo   V100安装完成！
echo.
echo   重要配置说明：
echo   ---------------------------------------------------------
echo   1. 使用 attn_mode="torch" (SDPA) 进行注意力计算
echo      - flash-attn 在V100上不稳定，会产生NaN
echo      - torch_compile=true 可以正常启用
echo.
echo   2. 推荐的V100训练配置：
echo      mixed_precision = "fp16"
echo      attn_mode = "torch"
echo      torch_compile = true
echo      gradient_checkpointing = true
echo      lora_fp32_compute = true  (自动启用)
echo.
echo   3. 启动训练：
echo      .venv\Scripts\activate
echo      python train.py --config your_config.toml
echo.
echo   4. 或使用WebUI：
echo      start-webui-win.bat
echo ============================================================
pause
