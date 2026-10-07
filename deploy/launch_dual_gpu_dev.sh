#!/usr/bin/env bash
set -euo pipefail

project_root=/data/fanshide/sglang-omni-dev
runtime_home=/data/fanshide/.runtime-home
cache_root=/data/fanshide/.cache
log_root="$runtime_home/sglang-omni-dev"
python_bin="$project_root/.venv/bin/python3"
gpu1_pid=""
gpu0_pid=""

mkdir -p "$log_root/realtime"

stop_children() {
    trap - EXIT INT TERM
    if [[ -n "$gpu0_pid" ]] && kill -0 "$gpu0_pid" 2>/dev/null; then
        kill -TERM "$gpu0_pid" 2>/dev/null || true
    fi
    if [[ -n "$gpu1_pid" ]] && kill -0 "$gpu1_pid" 2>/dev/null; then
        kill -TERM "$gpu1_pid" 2>/dev/null || true
    fi
    [[ -z "$gpu0_pid" ]] || wait "$gpu0_pid" 2>/dev/null || true
    [[ -z "$gpu1_pid" ]] || wait "$gpu1_pid" 2>/dev/null || true
}
trap stop_children EXIT INT TERM

env \
    CUDA_VISIBLE_DEVICES=1 \
    SGLANG_OMNI_SERVICE_INSTANCE_ID=sglang-omni-GPU0_1-dev-gpu1 \
    SGLANG_OMNI_INTERNAL_MODEL_API=1 \
    SGLANG_OMNI_ACTION_WARMUP=0 \
    SGLANG_OMNI_GLOBAL_ACTION_PREWARM_BUDGET_S=0.001 \
    "$python_bin" -m sglang_omni.cli serve \
        --config "$project_root/deploy/config_reply_gpu.yaml" \
        --host 127.0.0.1 \
        --port 18005 \
        --log-level info &
gpu1_pid=$!

while ! bash -c '</dev/tcp/127.0.0.1/18005' 2>/dev/null; do
    if ! kill -0 "$gpu1_pid" 2>/dev/null; then
        wait "$gpu1_pid"
        exit $?
    fi
    sleep 1
done

env \
    CUDA_VISIBLE_DEVICES=0 \
    SGLANG_OMNI_SERVICE_INSTANCE_ID=sglang-omni-GPU0_1-dev-gpu0 \
    SGLANG_OMNI_SERVICE_ROLE=action-decision \
    SGLANG_OMNI_MODEL_VERSION=e57a \
    SGLANG_OMNI_ACTION_DECISION_TOKEN="${SGLANG_OMNI_ACTION_DECISION_TOKEN:-${SGLANG_OMNI_INTERNAL_MODEL_TOKEN:-${SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN:?set an internal service token}}}" \
    SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN="${SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN:-${SGLANG_OMNI_ACTION_DECISION_TOKEN:-${SGLANG_OMNI_INTERNAL_MODEL_TOKEN:?set an internal service token}}}" \
    SGLANG_OMNI_ACTION_ARTIFACT_ROOT=/data/xingmt/model_repo/action_prediction_model/assets \
    "$python_bin" -m sglang_omni.cli serve \
        --config "$project_root/deploy/config_action_gpu.yaml" \
        --host 0.0.0.0 \
        --port 18004 \
        --log-level info &
gpu0_pid=$!

wait -n "$gpu0_pid" "$gpu1_pid"
