"""One-shot edit: replace the WANDB_MODE cell with auto-detect logic.

If WANDB_API_KEY is set in the environment (e.g., from RunPod's pod env
vars or a local .env file the user has sourced), switch wandb to online
mode. Otherwise, stay offline. Auto-detection means the same notebook
works on a fresh pod with no wandb auth AND on a pod where the user has
set the key.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path("notebooks/sft_phase1_runpod_ablation_2000.ipynb")

NEW_SRC = """# WANDB mode auto-detect.
#
# If WANDB_API_KEY is in the environment, switch wandb to ONLINE mode
# (writes runs to wandb.ai under your account). Otherwise, stay OFFLINE
# (writes to wandb/offline-run-*/ which you can sync later if you want).
#
# Set WANDB_API_KEY one of three ways before running this cell:
#   1. RunPod pod env var (set when creating the pod; persists for pod lifetime)
#   2. A .env file (gitignored; manually source before launching Jupyter)
#   3. Manual `export WANDB_API_KEY=...` in the JupyterLab terminal
#
# The wandb library itself looks up WANDB_API_KEY automatically; no extra
# auth call needed inside this notebook.
if os.environ.get('WANDB_API_KEY'):
    os.environ['WANDB_MODE'] = 'online'
    masked = os.environ['WANDB_API_KEY'][:4] + '...' + os.environ['WANDB_API_KEY'][-4:]
    print(f'WANDB_MODE = online   (using WANDB_API_KEY = {masked})')
else:
    os.environ['WANDB_MODE'] = 'offline'
    print('WANDB_MODE = offline  (no WANDB_API_KEY in env; runs land in wandb/offline-run-*/)')
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    # Find the cell currently containing WANDB_MODE = 'offline'.
    target_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        if "WANDB_MODE" in "".join(c["source"]) and "offline" in "".join(c["source"]):
            target_idx = i
            break
    if target_idx is None:
        raise SystemExit("could not find existing WANDB_MODE cell")
    nb["cells"][target_idx]["source"] = NEW_SRC.splitlines(keepends=True)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated cell {target_idx} in {NB}")


if __name__ == "__main__":
    main()
