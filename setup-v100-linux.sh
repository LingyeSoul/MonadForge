#!/usr/bin/env bash
# ============================================================
# MonadForge V100 专用安装脚本 (Linux)
# ============================================================
# 使用 torch==2.10.0+cu129，不安装 flash-attn
# V100 (SM 7.0) 不支持 flash-attn，生产训练请使用 attn_mode="torch"
#
# 用法:
#   chmod +x setup-v100-linux.sh
#   ./setup-v100-linux.sh
# ============================================================

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

say()  { echo -e "${CYAN}==>${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}警告:${NC} $*"; }
die()  { echo -e "${RED}错误:${NC} $*" >&2; exit 1; }

# ============================================================
# 1. 检查系统依赖
# ============================================================
say "检查系统依赖 ..."

command -v python3 >/dev/null 2>&1 || die "需要 python3"
command -v curl >/dev/null 2>&1 || die "需要 curl"
command -v tar >/dev/null 2>&1 || die "需要 tar"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
say "Python 版本: $PYTHON_VERSION"

# 检查Python 3.13
if [[ ! "$PYTHON_VERSION" =~ ^3\.13\. ]]; then
    warn "推荐使用 Python 3.13，当前版本: $PYTHON_VERSION"
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || die "安装已取消"
fi

# ============================================================
# 2. 安装 uv
# ============================================================
say "检查 uv ..."

if ! command -v uv >/dev/null 2>&1; then
    say "安装 uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 使 uv 在当前 shell 可用
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
fi

if command -v uv >/dev/null 2>&1; then
    UV_VERSION=$(uv --version 2>&1 | awk '{print $2}')
    ok "uv 已安装: $UV_VERSION"
else
    die "uv 安装失败；请打开新终端并重新运行"
fi

# ============================================================
# 3. 检查CUDA
# ============================================================
say "检查CUDA兼容性 ..."

cuda_ok() {
    for n in nvcc /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda/bin/nvcc /usr/local/cuda-12.*/bin/nvcc; do
        { command -v "$n" >/dev/null 2>&1 || [ -x "$n" ]; } || continue
        "$n" --version 2>/dev/null | grep -qE 'release 12\.[0-9]' && return 0
    done
    return 1
}

if cuda_ok; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1)
    ok "CUDA toolkit 检测成功: $CUDA_VERSION"
elif [ -n "${ANIMA_SKIP_CUDA:-}" ]; then
    say "ANIMA_SKIP_CUDA 已设置 — 跳过CUDA检查"
else
    warn "未检测到CUDA toolkit"
    warn "torch==2.10.0+cu129 需要CUDA 12.9 runtime"
    warn "torch wheels 已捆绑CUDA runtime，但torch.compile/Triton需要系统toolkit"
    warn ""
    warn "建议安装CUDA 12.9:"
    warn "  https://developer.nvidia.com/cuda-12-9-0-download-archive"
    warn ""
    warn "或设置 ANIMA_SKIP_CUDA=1 跳过此检查继续安装"
    read -p "是否继续安装？(y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || die "安装已取消"
fi

# ============================================================
# 4. 创建V100专用虚拟环境
# ============================================================
say "创建虚拟环境 (.venv) ..."

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    warn "$VENV_DIR 已存在"
    read -p "是否删除并重新创建？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        say "已删除旧的虚拟环境"
    else
        say "使用现有虚拟环境"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.13
    ok "已创建 $VENV_DIR (Python 3.13)"
fi

# 激活虚拟环境
say "激活虚拟环境 ..."
source "$VENV_DIR/bin/activate"
ok "虚拟环境已激活: $VIRTUAL_ENV"

# ============================================================
# 5. 安装V100兼容依赖
# ============================================================
say "安装V100兼容依赖 ..."

# 配置uv使用PyTorch CUDA 12.9索引
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu129"
export UV_INDEX_STRATEGY="unsafe-best-match"

# 安装torch和torchvision（V100兼容版本）
say "安装 torch==2.10.0+cu129 和 torchvision==0.25.0+cu129 ..."
uv pip install torch==2.10.0+cu129 torchvision==0.25.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129 \
    --quiet
ok "torch 和 torchvision 安装完成"

# 验证torch安装
say "验证torch安装 ..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 安装其他依赖（不包含flash-attn）
say "安装其他依赖 (不包含flash-attn) ..."
if [ -f "requirements-v100.txt" ]; then
    uv pip install -r requirements-v100.txt \
        --index-strategy unsafe-best-match \
        --quiet
    ok "依赖安装完成"
else
    die "未找到 requirements-v100.txt，请确保文件存在"
fi

# ============================================================
# 6. 配置环境变量（可选）
# ============================================================
say "配置V100优化环境变量 ..."

ENV_FILE="$VENV_DIR/bin/activate"

# 检查是否已经添加过
if ! grep -q "# MonadForge V100 optimizations" "$ENV_FILE" 2>/dev/null; then
    cat >> "$ENV_FILE" << 'EOF'

# MonadForge V100 optimizations
export ANIMA_V100_FLASH_STABILITY="off"
export ANIMA_DEBUG_FINITE="0"
EOF
    ok "已添加V100优化环境变量"
else
    say "环境变量已配置"
fi

# ============================================================
# 7. 验证安装
# ============================================================
say "验证安装 ..."

# 检查关键包
python3 << 'EOF'
import sys
import importlib

packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('accelerate', 'Accelerate'),
    ('transformers', 'Transformers'),
    ('diffusers', 'Diffusers'),
    ('safetensors', 'SafeTensors'),
    ('einops', 'Einops'),
    ('bitsandbytes', 'BitsAndBytes'),
]

print("\n已安装的包:")
print("-" * 50)
all_ok = True
for pkg, name in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {name:20} {version}")
    except ImportError:
        print(f"  ✗ {name:20} 未安装")
        all_ok = False

print("-" * 50)
if all_ok:
    print("所有关键包已安装成功！")
else:
    print("警告: 部分包未安装")
    sys.exit(1)
EOF

# ============================================================
# 完成
# ============================================================
echo
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  V100 安装完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo
echo "  重要配置说明："
echo "  ---------------------------------------------------------"
echo "  1. 使用 attn_mode=\"torch\" (SDPA) 进行注意力计算"
echo "     - flash-attn 在V100上不稳定，会产生NaN"
echo "     - torch_compile=true 可以正常启用"
echo ""
echo "  2. 推荐的V100训练配置："
echo "     mixed_precision = \"fp16\""
echo "     attn_mode = \"torch\""
echo "     torch_compile = true"
echo "     gradient_checkpointing = true"
echo "     lora_fp32_compute = true  (自动启用)"
echo ""
echo "  3. 启动训练："
echo "     source .venv/bin/activate"
echo "     python train.py --config your_config.toml"
echo ""
echo "  4. 或使用WebUI："
echo "     bash start-webui-Linux.sh"
echo "  ---------------------------------------------------------"
echo
