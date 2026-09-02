# Runtime Environment and Compatibility Matrix

This page is the source of truth for the environment used to install, validate,
and operate SGLang-Omni. Model-specific GPU placement and memory requirements
remain in the corresponding cookbook or example configuration.

## Support levels

- **Recommended**: the default environment for new installations.
- **CI validated**: exercised by the repository's automated GPU or documentation
  workflows.
- **Documented**: supported by project metadata or an in-repository launch
  profile, but not necessarily covered by the complete GPU CI suite.
- **Unvalidated**: may work, but maintainers should not rely on it without a
  model-specific smoke test.

## System compatibility

| Component | Compatibility | Status | Notes |
| --- | --- | --- | --- |
| Operating system | Linux | Recommended | GPU serving depends on the NVIDIA Linux driver and CUDA-specific packages. |
| CPU architecture | x86_64 | CI validated | Other architectures are not currently covered by GPU CI. |
| Container runtime | Docker with NVIDIA Container Toolkit | Recommended | Use `--gpus all`, host IPC, and sufficient shared memory as shown in [Installation](../get_started/installation.md). |
| Bare-metal installation | Linux with the CUDA toolchain and UCX 1.20.x | Documented | UCX must be built with CUDA and verbs support; reuse the flags in `docker/Dockerfile`. |
| CPU-only or non-NVIDIA serving | Not supported | Unvalidated | The default dependency set contains CUDA 13-specific NIXL and Mooncake wheels. |

The repository does not currently pin an exact Linux distribution, Docker
version, or NVIDIA Container Toolkit version. The recommended container is the
most reliable way to inherit a compatible user-space stack.

## CUDA and communication stack

| Component | Required or validated value | Status | Maintenance rule |
| --- | --- | --- | --- |
| CUDA user-space stack | CUDA 13 / cu130 | Recommended | Keep aligned with the `nixl-cu13` and `mooncake-transfer-engine-cuda13` dependencies. |
| NVIDIA driver | A Linux driver compatible with the installed CUDA 13 runtime | Required | A recorded H100 validation used driver `580.126.20`; this is validation evidence, not a repository-wide minimum-driver guarantee. |
| PyTorch | `2.11.0` | Locked | Keep `torch`, `torchvision`, `torchaudio`, and `torchcodec` versions aligned. |
| SGLang | `0.5.16` | Locked | `transformers`, `flash-attn-4`, and `kernels` are selected for this stack. |
| UCX | `1.20.x`, built with CUDA and verbs | Required for UCX/RDMA paths | The Docker build currently tracks the `v1.20.x` ref. Pin a commit when producing a release image. |
| NCCL | Supplied by the PyTorch/container CUDA stack | CI validated | Do not mix NCCL libraries from a different CUDA major version. |

Check the effective environment before diagnosing runtime failures:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
ucx_info -v
```

`ucx_info` is only required when using a UCX-backed data path or validating a
manual installation.

## GPU compatibility

GPU memory requirements are model- and topology-specific. A GPU listed here
does not imply that every model fits on one device.

| GPU family | Status | Current evidence and guidance |
| --- | --- | --- |
| NVIDIA H100 80 GB | CI validated | Used by the self-hosted GPU CI and multiple model benchmarks. Qwen3-Omni provides single-GPU FP8 profiles; larger models may require tensor parallelism. |
| NVIDIA H200 141 GB | Documented and benchmark validated | H200-specific Qwen3-Omni and MPS profiles are included. Recheck memory fractions after changing model, concurrency, or sequence limits. |
| NVIDIA H20 / H20-3e | Documented | Qwen3-Omni FP8/INT4 launch profiles are documented. Run model-specific startup and inference smoke tests before production use. |
| Other NVIDIA CUDA GPUs | Unvalidated | They may work when the CUDA 13 stack and model memory requirements are satisfied, but no compatibility guarantee is maintained. |
| AMD, Apple GPU, or CPU-only | Not supported for serving | The default runtime and relay dependency set is NVIDIA CUDA-specific. |

For exact placement and capacity, consult the selected model's cookbook and the
YAML files under `examples/configs/` or `deploy/`.

## Python compatibility

| Python version | Status | Use |
| --- | --- | --- |
| 3.12 | Recommended | Used by the installation guide and documentation CI. |
| 3.11 | CI validated | Used to build the reusable Omni GPU CI environment. |
| 3.10 | Documented minimum | Allowed by project metadata, but not the primary full GPU-serving CI environment. |
| 3.13 and newer | Unvalidated | Do not use for production until the CUDA wheels and a model smoke test pass. |

The package metadata remains `requires-python = ">=3.10"`. Narrow that range if
a future dependency stops supporting one of the versions above.

## Reproducible dependency installation

`pyproject.toml` declares direct dependencies and `uv.lock` records the resolved
dependency graph and artifact hashes. The current lock was generated with
`uv 0.12.5`.

For a new checkout, install exactly the locked environment:

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv sync --locked
```

Install the optional Audar TTS dependencies only when needed:

```bash
uv sync --locked --extra audar-tts
```

When changing dependencies:

1. Edit `pyproject.toml`.
2. Run `uv lock` and review both `pyproject.toml` and `uv.lock`.
3. Run `uv lock --check` and the relevant import, unit, and GPU smoke tests.
4. Commit the declaration and lock-file changes together.

Do not run an unconstrained upgrade merely to refresh the lock file. Dependency
upgrades should be intentional and reviewed against the CUDA compatibility rows
above.

## Maintainer verification checklist

Before declaring a new environment supported, record all of the following in
the pull request or release notes:

- Linux distribution and architecture
- Python and `uv` versions
- NVIDIA driver and CUDA runtime versions
- GPU model and memory size
- container image tag and digest, when applicable
- model repository ID and immutable revision
- launch configuration and topology
- successful startup, health check, and one representative inference
