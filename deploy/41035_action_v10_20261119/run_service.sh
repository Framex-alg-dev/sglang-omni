#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
PYTHON_BIN="${SGLANG_OMNI_PYTHON:-python3}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SGLANG_OMNI_SERVICE_ROLE=action-decision
export SGLANG_OMNI_MODEL_VERSION=v10-boundary-ep1-final-fp8
export SGLANG_OMNI_QWEN3_CHAT_TEMPLATE_MODEL=/data/xingmt/action_recall_v10_20261119/model
export SGLANG_OMNI_ACTION_ARTIFACT_ROOT="$DEPLOY_DIR/assets"
export SGLANG_OMNI_ACTION_USAGE_SCENARIO_ROOT="$DEPLOY_DIR/assets/usage_scenarios"
export SGLANG_OMNI_GPU_RUNTIME_RESERVE_GIB="${SGLANG_OMNI_GPU_RUNTIME_RESERVE_GIB:-1.0}"
export SGLANG_OMNI_SCOPED_RADIX_CACHE=1
export SGLANG_OMNI_STRICT_PORT=1
export SGLANG_OMNI_SERVICE_INSTANCE_ID="${SGLANG_OMNI_SERVICE_INSTANCE_ID:-41035-action-v10}"

FRAME_LOG_DIR=/var/log/framex/omni-action-worker
if [[ ! -d "$FRAME_LOG_DIR" || ! -w "$FRAME_LOG_DIR" ]]; then
  echo "FrameX log directory is missing or not writable: $FRAME_LOG_DIR" >&2
  echo "Provision it before launch: sudo install -d -o $(id -un) -g $(id -gn) $FRAME_LOG_DIR" >&2
  exit 1
fi

exec "$PYTHON_BIN" -m sglang_omni.cli serve \
  --config "$DEPLOY_DIR/config.yaml" \
  --host "${ACTION_V10_HOST:-0.0.0.0}" \
  --port "${ACTION_V10_PORT:-18049}" \
  --log-level "${ACTION_V10_LOG_LEVEL:-info}"
