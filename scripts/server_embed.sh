#!/usr/bin/env bash
# 在**服务器**上准备环境并跑嵌入：装依赖 → 加载 bge-large-zh-v1.5 → 把 chunks.jsonl 编码成向量文件。
#
# 前置：把仓库和切块文件放到服务器上（上行只传这两样）
#   rsync -a --exclude data --exclude .git ./ user@host:~/dubbing/
#   rsync -a data/poetry/chunks.jsonl user@host:~/dubbing/data/poetry/
#
# 用法：
#   bash scripts/server_embed.sh --chunks data/poetry/chunks.jsonl --out data/poetry/vectors.npy \
#       --device cuda --batch 64
#
#   # 先跑一小段验证（2000 条，约一分钟）
#   bash scripts/server_embed.sh --chunks data/poetry/chunks.jsonl --limit 2000 \
#       --out data/poetry/vectors.head.npy --device cuda
#
#   # 国内网络：走镜像下模型
#   bash scripts/server_embed.sh --mirror https://hf-mirror.com --chunks ... --out ...
#
#   # 模型已经在本地目录（离线机器）
#   bash scripts/server_embed.sh --model /data/models/bge-large-zh-v1.5 --device cuda ...
#
# 本脚本自己的选项（写在最前面，其余参数原样转给 scripts/embed_poetry.py）：
#   --env-dir DIR    虚拟环境目录，默认 <仓库>/.venv-embed
#   --model NAME     模型名或本地模型目录，默认 BAAI/bge-large-zh-v1.5
#   --mirror URL     HuggingFace 镜像，例如 https://hf-mirror.com
#   --skip-install   跳过装依赖（环境已就绪时）
#
# 跑完把 --out 指定的 .npy 和同名的 .json 拷回本机（下行只传这两个文件）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${EMBED_ENV_DIR:-$REPO_ROOT/.venv-embed}"
MODEL="${EMBED_MODEL:-BAAI/bge-large-zh-v1.5}"
MIRROR="${HF_ENDPOINT:-}"
SKIP_INSTALL=0
PASSTHRU=()

usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-dir)      ENV_DIR="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --mirror)       MIRROR="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              PASSTHRU+=("$1"); shift ;;
    esac
done

step() { printf '\n== %s\n' "$*" >&2; }

step "检查仓库"
if [[ ! -f "$REPO_ROOT/dubbing/rag/vectors.py" || ! -f "$REPO_ROOT/scripts/embed_poetry.py" ]]; then
    echo "找不到 dubbing/rag/vectors.py 或 scripts/embed_poetry.py。" >&2
    echo "把整个仓库放到服务器上再跑（rsync -a ./ user@host:~/dubbing/），只传 chunks.jsonl 是不够的。" >&2
    exit 1
fi
echo "仓库：$REPO_ROOT"

step "准备 Python 环境"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "创建虚拟环境：$ENV_DIR"
    "$PYTHON_BIN" -m venv "$ENV_DIR"
else
    echo "复用已有环境：$ENV_DIR"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    "$ENV_DIR/bin/python" -m pip install --quiet --upgrade pip
    echo "安装 sentence-transformers（会带上 torch，第一次比较慢）"
    "$ENV_DIR/bin/python" -m pip install --quiet sentence-transformers
else
    echo "--skip-install：跳过装依赖"
fi

if [[ -n "$MIRROR" ]]; then
    export HF_ENDPOINT="$MIRROR"
    echo "HF 镜像：$HF_ENDPOINT"
fi

step "GPU 自检"
"$ENV_DIR/bin/python" - <<'PY' || true
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  cuda 可用：{name}（torch {torch.__version__}）")
    else:
        print(f"  没有可用的 cuda，将退回 CPU（torch {torch.__version__}）")
        print("  想用 GPU：确认驱动与 CUDA 版本匹配，并让 --device cuda")
except ImportError:
    print("  没装 torch？sentence-transformers 应当会带上它")
PY

step "开始编码"
echo "模型：$MODEL"
echo "（第一次运行会把模型下到 ~/.cache/huggingface，bge-large-zh-v1.5 约 1.3 GB）"

exec "$ENV_DIR/bin/python" "$REPO_ROOT/scripts/embed_poetry.py" --model "$MODEL" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
