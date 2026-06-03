"""Fail-fast CUDA stack check for GPU training environments.

This script catches the common expensive failure mode where a working PyTorch
CUDA 12 install is polluted by CUDA 13 pip packages. Run it before distributed
training:

    python scripts/check_cuda_stack.py
"""

from __future__ import annotations

import importlib.metadata as metadata

MIN_TORCH_VERSION = (2, 6)


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = dist.version
    return packages


def _torch_major_minor(version: str) -> tuple[int, int] | None:
    base = version.split("+", 1)[0]
    parts = base.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"FAIL: could not import torch: {exc!r}")
        print("Repair hint: reinstall the Torch CUDA wheel and its cu12 dependencies.")
        return 1

    packages = _installed_packages()
    torch_cuda = torch.version.cuda or "cpu"
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    nccl_version = None
    if cuda_available and hasattr(torch.cuda, "nccl"):
        try:
            nccl_version = torch.cuda.nccl.version()
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            nccl_version = f"unavailable: {exc!r}"

    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch_cuda}")
    print(f"cuda available: {cuda_available}")
    print(f"device count: {device_count}")
    print(f"nccl: {nccl_version}")

    torch_major_minor = _torch_major_minor(torch.__version__)
    if torch_major_minor is None or torch_major_minor < MIN_TORCH_VERSION:
        print(
            "FAIL: torch>=2.6 is required for the active GPU stack. "
            "ChainEval/BERTScore uses transformers model loading paths that "
            "reject older torch versions."
        )
        print("Repair hint: bash scripts/repair_cuda_stack.sh")
        return 1

    cu13_packages = sorted(
        name
        for name in packages
        if name.endswith("-cu13")
        or (name.startswith(("cuda-", "nvidia-")) and packages[name].startswith("13."))
    )
    if cu13_packages and torch_cuda.startswith("12."):
        print("FAIL: CUDA 13 pip packages are installed next to a Torch CUDA 12 build:")
        for name in cu13_packages:
            print(f"  {name}=={packages[name]}")
        print("Remove the cu13 packages before launching Accelerate/NCCL training.")
        return 1

    optional_torch_packages = ("torchvision", "torchaudio")
    for package in optional_torch_packages:
        if package not in packages:
            continue
        try:
            __import__(package)
        except Exception as exc:
            print(
                f"FAIL: optional package {package}=={packages[package]} is installed "
                f"but cannot be imported: {exc!r}"
            )
            print(
                "Repair hint: uninstall torchvision/torchaudio or run "
                "bash scripts/repair_cuda_stack.sh. finpost does not use "
                "these packages, but transformers may detect a broken install."
            )
            return 1

    if not cuda_available:
        print("FAIL: torch.cuda.is_available() is false.")
        return 1

    if device_count < 1:
        print("FAIL: no CUDA devices visible.")
        return 1

    print("PASS: CUDA stack is importable and no CUDA 13 package conflict was detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
