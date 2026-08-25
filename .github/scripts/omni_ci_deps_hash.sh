#!/usr/bin/env bash
# Shared dependency fingerprint for Omni CI venv reuse.
set -euo pipefail

omni_ci_deps_hash() {
  local dependency_file
  for dependency_file in pyproject.toml uv.lock; do
    if [ ! -f "${dependency_file}" ]; then
      echo "${dependency_file} not found in $(pwd)" >&2
      return 1
    fi
  done
  sha256sum pyproject.toml uv.lock | sha256sum | awk '{print $1}'
}
