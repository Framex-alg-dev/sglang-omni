# RTX PRO 5000 72GB Blackwell kernel configs

These five JSON files are measured Triton tuning results for Qwen3-Omni on an
NVIDIA RTX PRO 5000 72GB Blackwell GPU. They are hardware- and shape-specific.

`sgl-omni serve` installs them automatically when that exact GPU is visible:

- `N=*.json` goes to the active SGLang dense W8A8 FP8 config directory.
- `E=*.json` goes to the active SGLang MoE directory for the installed Triton
  version.

The runtime installer is shared by editable development environments and the
production Docker image. It is idempotent, skips other GPU models, and logs
when files are installed. Retune these files after material SGLang, Triton,
CUDA, model-shape, or GPU changes instead of assuming the old result remains
optimal.
