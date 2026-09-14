#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
script_path="${project_root}/scripts/start_luozhijie_single_gpu.sh"
unit_name="luozhijie-sglang-omni.service"
gpu_index="${SGLANG_GPU_INDEX:-3}"
service_port="${SGLANG_SERVICE_PORT:-18101}"
service_host="${SGLANG_SERVICE_HOST:-127.0.0.1}"
config_file="${project_root}/deploy/config_single_gpu.yaml"
knowledge_env_file="${project_root}/deploy/realtime-knowledge.env"
python_bin="${project_root}/.venv/bin/python"
cache_root="/data/luozhijie/.cache"
log_root="${project_root}/logs/realtime"

run_server() {
  export XDG_CACHE_HOME="${cache_root}"
  export FLASHINFER_WORKSPACE_DIR="${cache_root}/flashinfer"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
  export CUDA_HOME="/usr/local/cuda"
  export CUDA_VISIBLE_DEVICES="${gpu_index}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export SGLANG_OMNI_QWEN3_CHAT_TEMPLATE_MODEL="/data/models/Qwen3-Omni-30B-A3B-Instruct"
  export SGLANG_OMNI_REALTIME_LOG_DIR="${log_root}"
  export SGLANG_OMNI_REALTIME_LOG_TIMEZONE="Asia/Shanghai"
  export SGLANG_OMNI_REALTIME_LOG_MAX_FILE_MB=128
  export SGLANG_OMNI_REALTIME_LOG_FULL_INSTRUCTIONS=1
  export SGLANG_OMNI_RESOURCE_LOG_INTERVAL_S=20
  export SGLANG_OMNI_SERVICE_INSTANCE_ID="sglang-omni-luozhijie-gpu${gpu_index}"
  export SGLANG_OMNI_STRICT_PORT=1
  export PYTHONUNBUFFERED=1
  export PATH="${project_root}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

  if [[ -f "${knowledge_env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${knowledge_env_file}"
    set +a
  fi
  # Match the currently deployed reference unit's timeout override.
  export SGLANG_OMNI_REALTIME_KNOWLEDGE_TURN_TIMEOUT_MS=1500

  exec "${python_bin}" -m sglang_omni.cli serve \
    --config "${config_file}" \
    --host "${service_host}" \
    --port "${service_port}" \
    --log-level info \
    --realtime-tts-url ws://127.0.0.1:40001/api-ws/v1/realtime \
    --realtime-tts-voice benchmark_qwen_cherry_zh
}

if [[ "${1:-}" == "--run-server" ]]; then
  run_server
fi

for required in "${python_bin}" "${config_file}" "${knowledge_env_file}"; do
  if [[ ! -e "${required}" ]]; then
    echo "error: required file does not exist: ${required}" >&2
    exit 1
  fi
done
for command_name in systemctl systemd-run nvidia-smi ss curl; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "error: required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done
if [[ ! "${gpu_index}" =~ ^[0-9]+$ ]] || [[ ! "${service_port}" =~ ^[0-9]+$ ]]; then
  echo "error: SGLANG_GPU_INDEX and SGLANG_SERVICE_PORT must be integers" >&2
  exit 1
fi

# This script may replace only its own transient unit. It never stops a foreign unit.
if systemctl --user is-active --quiet "${unit_name}"; then
  echo "Stopping our existing ${unit_name} before restart ..."
  systemctl --user stop "${unit_name}"
fi
systemctl --user reset-failed "${unit_name}" >/dev/null 2>&1 || true

for _ in $(seq 1 30); do
  gpu_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_index}" | tr -d ' ')"
  if [[ "${gpu_used_mib}" =~ ^[0-9]+$ ]] && (( gpu_used_mib < 1024 )); then
    break
  fi
  sleep 1
done
if [[ ! "${gpu_used_mib:-}" =~ ^[0-9]+$ ]] || (( gpu_used_mib >= 1024 )); then
  echo "error: GPU ${gpu_index} is not free (${gpu_used_mib:-unknown} MiB used); refusing to start" >&2
  exit 1
fi
if ss -H -ltn "sport = :${service_port}" | head -n 1 | grep -q .; then
  echo "error: port ${service_port} is already listening; refusing to start" >&2
  exit 1
fi

mkdir -p \
  "${cache_root}/flashinfer" \
  "${cache_root}/torchinductor" \
  "${log_root}"

echo "Starting ${unit_name}: GPU=${gpu_index}, ${service_host}:${service_port}"
echo "Config: ${config_file}"
systemd-run --user \
  --unit="${unit_name}" \
  --property="WorkingDirectory=${project_root}" \
  --property=Restart=on-failure \
  --property=RestartSec=10 \
  --property=TimeoutStartSec=infinity \
  --property=TimeoutStopSec=120 \
  --property=KillMode=control-group \
  --setenv="SGLANG_GPU_INDEX=${gpu_index}" \
  --setenv="SGLANG_SERVICE_PORT=${service_port}" \
  --setenv="SGLANG_SERVICE_HOST=${service_host}" \
  "${script_path}" --run-server

echo "Waiting for health (first CUDA/JIT initialization can take several minutes) ..."
for attempt in $(seq 1 180); do
  if health="$(curl -fsS --max-time 2 "http://127.0.0.1:${service_port}/health" 2>/dev/null)"; then
    echo "Healthy: ${health}"
    echo "WebSocket: ws://127.0.0.1:${service_port}/v1/session/realtime"
    exit 0
  fi
  if ! systemctl --user is-active --quiet "${unit_name}"; then
    echo "error: ${unit_name} exited during startup" >&2
    journalctl --user -u "${unit_name}" --no-pager -n 120 >&2
    exit 1
  fi
  if (( attempt % 10 == 0 )); then
    echo "Still starting ... $((attempt * 2))s"
  fi
  sleep 2
done

echo "error: timed out waiting for http://127.0.0.1:${service_port}/health" >&2
echo "Inspect logs: journalctl --user -u ${unit_name} -f" >&2
exit 1
