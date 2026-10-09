#!/usr/bin/env bash
# 可在任意工作目录调用；模型路径始终相对于代码目录解析。
set -euo pipefail
TRACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACE_VENV="${TRACE_VENV:-$TRACE_DIR/.venv}"
TRACE_PYTHON="${TRACE_PYTHON:-$TRACE_VENV/bin/python}"
if ! TRACE_PYTHON="$(command -v -- "$TRACE_PYTHON")"; then
    echo "找不到可用的环境 Python，请先执行 bash $TRACE_DIR/install.sh；共享 NAS 上的环境需在当前机器创建。" >&2
    exit 1
fi
# 帮助与文件存在性检查只需要标准库，允许在安装模型依赖之前使用。
if [[ $# -eq 1 && ( "$1" == '--help' || "$1" == '-h' || "$1" == '--check-models' ) ]]; then
    exec "$TRACE_PYTHON" "$TRACE_DIR/run.py" "$@"
fi
if ! "$TRACE_PYTHON" -c '
import importlib, sys
errors = []
for name in ("torch", "transformers", "peft", "numpy", "scipy", "accelerate"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append(f"{name}: {exc}")
if errors:
    print("环境依赖未安装完整或加载失败：\n" + "\n".join(errors), file=sys.stderr)
    sys.exit(1)
'; then
    echo "请先成功执行 install.sh。当前不会启动推理，也不会创建或覆盖输出目录。" >&2
    exit 1
fi
exec "$TRACE_PYTHON" "$TRACE_DIR/run.py" "$@"
