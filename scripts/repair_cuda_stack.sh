#!/usr/bin/env bash
set -euo pipefail

# Repair the common GPU-environment failure mode where a CUDA 12 PyTorch wheel is mixed
# with CUDA 13 pip packages or broken optional torch media packages.

echo "[finpost] Removing optional torch media packages that can break transformers imports..."
python -m pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true

echo "[finpost] Removing CUDA 13 pip packages when present..."
python - <<'PY'
import importlib.metadata as metadata
import subprocess
import sys

packages = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name")
    version = dist.version
    if not name:
        continue
    lower = name.lower()
    if lower.endswith("-cu13") or (
        lower.startswith(("cuda-", "nvidia-")) and version.startswith("13.")
    ):
        packages.append(name)

if packages:
    print("[finpost] uninstalling:", " ".join(sorted(packages)))
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *packages])
else:
    print("[finpost] no CUDA 13 pip packages detected")
PY

echo "[finpost] Installing Torch 2.6.0 CUDA 12.4 wheel..."
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0+cu124"

echo "[finpost] Rechecking CUDA stack..."
python scripts/check_cuda_stack.py
