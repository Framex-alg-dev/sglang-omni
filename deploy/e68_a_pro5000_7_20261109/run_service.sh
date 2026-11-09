#!/usr/bin/env bash
set -euo pipefail
deployment_root=/data/xingmt/e68_a_action_20261109
python_bin=/data/fanshide/sglang-omni-dev/.venv/bin/python
cd "$deployment_root/code"
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="$deployment_root/code"
export CUDA_HOME=/usr/local/cuda
export PATH="/data/fanshide/sglang-omni-dev/.venv/bin:/usr/local/cuda/bin:$PATH"
export CPATH="/data/fanshide/.tools/uv-python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12:/usr/include"
export XDG_CACHE_HOME="$deployment_root/cache"
export TMPDIR="$deployment_root/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export SGLANG_OMNI_STRICT_PORT=1
export SGLANG_OMNI_SERVICE_ROLE=action-decision
export SGLANG_OMNI_SERVICE_INSTANCE_ID=e68-a-pro5000-7-1000045-20261109
export SGLANG_OMNI_MODEL_VERSION=e68-a-v116-step563-fp8
export SGLANG_OMNI_ACTION_ARTIFACT_ROOT="$deployment_root/assets"
export SGLANG_OMNI_QWEN3_CHAT_TEMPLATE_MODEL="$deployment_root/model"
export SGLANG_OMNI_SCOPED_RADIX_CACHE=1
export SGLANG_OMNI_GPU_RUNTIME_RESERVE_GIB=1.0
export SGLANG_OMNI_REALTIME_LOG_DIR="$deployment_root/logs/realtime"
export SGLANG_OMNI_REALTIME_LOG_TIMEZONE=Asia/Shanghai
export SGLANG_OMNI_REALTIME_LOG_MAX_FILE_MB=128
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
exec "$python_bin" -m sglang_omni.cli serve \
  --config "$deployment_root/config_e68_a_gpu.yaml" \
  --host 0.0.0.0 --port 18048 --log-level info
