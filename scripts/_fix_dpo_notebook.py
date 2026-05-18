"""One-shot edit script for notebooks/dpo_phase1_runpod.ipynb.

Two fixes:

1. WANDB env scoping. The canary cell currently does ``os.environ['WANDB_MODE']
   = 'offline'`` which persists in the kernel and silently makes the *full* DPO
   run offline as well (subprocesses inherit parent env). Fix: extend
   ``run_and_tail`` with an ``env_overrides`` kwarg and pass the offline flag
   only to the canary subprocess.

2. Canary cleanup cell. Insert a new step between the canary and the full run
   that removes the canary's checkpoints from ``DPO_CHECKPOINT_DIR``. Without
   this, the full run writes into the same dir on top of canary leftovers,
   wasting ~1-2 GB and risking the find-final-checkpoint logic picking a stale
   canary step if the full run dies.

This script is idempotent: if either fix is already present, it does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path('notebooks/dpo_phase1_runpod.ipynb')

NEW_RUN_HELPER = '''\
def run_and_tail(
    cmd: list[str],
    *,
    label: str,
    tail_lines: int = 40,
    env_overrides: dict[str, str] | None = None,
) -> None:
    """Run a subprocess, print its stdout/stderr tail, and raise on non-zero exit.

    ``env_overrides`` lets a caller scope env vars to *this one subprocess*
    without leaking them into the kernel via ``os.environ``. The canary cell
    uses this to set ``WANDB_MODE=offline`` for the canary only, so the full
    run that follows still syncs live to wandb.
    """
    print(f'=== {label} ===')
    print(' '.join(str(part) for part in cmd))
    env = None
    if env_overrides:
        env = {**os.environ, **env_overrides}
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    print('\\n--- stdout tail ---')
    print('\\n'.join(result.stdout.splitlines()[-tail_lines:]))
    if result.returncode != 0:
        print('\\n--- stderr tail ---')
        print('\\n'.join(result.stderr.splitlines()[-tail_lines:]))
        raise RuntimeError(f'{label} failed with exit code {result.returncode}')
    print(f'=== {label} passed ===')
'''

NEW_CANARY_CELL = '''\
# Canary is intentionally wandb-offline: 50 steps is a smoke test, not a run
# worth syncing. The env override is scoped to this subprocess only via
# run_and_tail's env_overrides arg, so the full run cell below still picks up
# WANDB_API_KEY from the pod env and syncs live.
canary_cmd = [
    sys.executable,
    '-m', 'finpost.training.dpo_train',
    '--config', str(DPO_YAML),
    '--device', 'cuda',
    '--max-steps', str(DPO_CANARY_STEPS),
]
run_and_tail(
    canary_cmd,
    label='Qwen DPO canary',
    env_overrides={'WANDB_MODE': 'offline'},
)
'''

CLEANUP_HEADER_SOURCE = '''\
### Drop canary checkpoints before the full run

The canary writes into the same ``DPO_CHECKPOINT_DIR`` as the full run. If we
do not clean up first, two things go wrong:

1. The canary's final checkpoint (~1-2 GB) sits on disk during the full run.
2. The ``dpo-find-checkpoint`` cell after training picks the highest-step
   directory under ``DPO_CHECKPOINT_DIR``. If the full run dies partway, that
   could be a stale canary step, and we would convert and evaluate the wrong
   model.

Dropping the canary checkpoints here makes the full run start from a clean
directory. The canary's logs in ``wandb/offline-run-*`` are tiny and left
alone for post-mortem if you ever need them.
'''

CLEANUP_CELL_SOURCE = '''\
import shutil

if DPO_CHECKPOINT_DIR.exists():
    print(f'canary leftovers under {DPO_CHECKPOINT_DIR}:')
    for path in sorted(DPO_CHECKPOINT_DIR.iterdir()):
        print(' ', path.name)
    shutil.rmtree(DPO_CHECKPOINT_DIR)
    print(f'\\nremoved {DPO_CHECKPOINT_DIR}')
else:
    print(f'{DPO_CHECKPOINT_DIR} does not exist yet - nothing to clean')

print('\\ndisk after cleanup:')
subprocess.run(['df', '-h', '/workspace'], check=False)
'''


def _split_source(text: str) -> list[str]:
    """nbformat stores source as a list of lines, each ending with \\n
    except the last. Mirror that to keep diffs clean."""
    lines = text.splitlines(keepends=True)
    return lines if lines else ['']


def _find_cell(cells: list[dict], cell_id: str) -> int:
    for i, c in enumerate(cells):
        if c.get('id') == cell_id:
            return i
    raise KeyError(f'cell id {cell_id!r} not found')


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    cells = nb['cells']

    # -- Fix 1: run_and_tail and canary cell --------------------------------
    helper_idx = _find_cell(cells, 'dpo-run-helper')
    helper_cell = cells[helper_idx]
    current_helper = ''.join(helper_cell['source'])
    if 'env_overrides' in current_helper:
        print('fix 1 (run_and_tail): already applied, skipping')
    else:
        helper_cell['source'] = _split_source(NEW_RUN_HELPER)
        helper_cell['outputs'] = []
        helper_cell['execution_count'] = None
        print('fix 1a (run_and_tail): updated')

    canary_idx = _find_cell(cells, 'dpo-canary')
    canary_cell = cells[canary_idx]
    current_canary = ''.join(canary_cell['source'])
    if "os.environ['WANDB_MODE']" in current_canary:
        canary_cell['source'] = _split_source(NEW_CANARY_CELL)
        canary_cell['outputs'] = []
        canary_cell['execution_count'] = None
        print('fix 1b (canary cell): updated')
    else:
        print('fix 1b (canary cell): already applied, skipping')

    # -- Fix 2: insert canary cleanup cells ---------------------------------
    cleanup_md_id = 'dpo-canary-cleanup-md'
    cleanup_code_id = 'dpo-canary-cleanup'

    already_present = any(c.get('id') == cleanup_code_id for c in cells)
    if already_present:
        print('fix 2 (cleanup cells): already inserted, skipping')
    else:
        # Insert right after dpo-post-canary-gpu, before the dpo-step-7 header
        anchor_idx = _find_cell(cells, 'dpo-post-canary-gpu')
        insert_at = anchor_idx + 1

        cleanup_md = {
            'cell_type': 'markdown',
            'id': cleanup_md_id,
            'metadata': {},
            'source': _split_source(CLEANUP_HEADER_SOURCE),
        }
        cleanup_code = {
            'cell_type': 'code',
            'id': cleanup_code_id,
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': _split_source(CLEANUP_CELL_SOURCE),
        }
        cells.insert(insert_at, cleanup_md)
        cells.insert(insert_at + 1, cleanup_code)
        print(f'fix 2 (cleanup cells): inserted at index {insert_at}')

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'wrote {NB_PATH}')


if __name__ == '__main__':
    main()
