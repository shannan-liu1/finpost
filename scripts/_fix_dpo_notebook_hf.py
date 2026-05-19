"""Update dpo_phase1_runpod.ipynb to pull combined-500 from HuggingFace.

Three edits:
1. dpo-step-2 markdown — update text from step 2000 to step 500.
2. dpo-constants — add SFT_HF_REPO / SFT_HF_SUBFOLDER / SFT_CHECKPOINT_STEP;
   replace local SFT_HF_CHECKPOINT path; update run names to reflect step 500.
3. New cell dpo-fetch-sft-ckpt — inserted between dpo-constants and dpo-step-3;
   downloads combined/step-0500 from HF Hub into a local staging dir, then
   reassigns SFT_HF_CHECKPOINT to the exact subdir containing the model files.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path('notebooks/dpo_phase1_runpod.ipynb')

NEW_STEP2_MD = '''\
## Step 2 - Study constants

Chosen SFT checkpoint: `combined` at step **500**, pulled from
`sl891/finpost-phase1-sft-ablation`. This was determined empirically by the
Phase 1 SFT ablation — combined-500 achieved the highest GSM8K accuracy (0.276)
of any arm or step, and was the only checkpoint that generalized well across
both GSM8K and MATH. All three arms overfit past step 500; 2000 steps was
overkill. See `docs/phase1-sft-study.html` for the full findings.

If you want to use a different arm or step, update `SFT_ARM`,
`SFT_CHECKPOINT_STEP`, and `SFT_HF_SUBFOLDER` below.
'''

NEW_CONSTANTS = '''\
SFT_ARM = 'combined'
SFT_CHECKPOINT_STEP = 500          # step within the SFT run; empirically best on GSM8K + MATH
SFT_HF_REPO = 'sl891/finpost-phase1-sft-ablation'
SFT_HF_SUBFOLDER = f'{SFT_ARM}/step-{SFT_CHECKPOINT_STEP:04d}'  # combined/step-0500

# SFT_HF_CHECKPOINT is set to the downloaded local path by the fetch cell below.
# Do not edit this line; edit SFT_ARM / SFT_CHECKPOINT_STEP above instead.
SFT_HF_CHECKPOINT = None

PAIR_RUN_NAME = f'qwen_{SFT_ARM}_{SFT_CHECKPOINT_STEP}s_k8_v1'
PAIR_OUT_DIR = Path(f'results/dpo_pairs/{PAIR_RUN_NAME}')

DPO_RUN_NAME = f'qwen-{SFT_ARM}-step{SFT_CHECKPOINT_STEP}-dpo'
DPO_EXPERIMENT_DIR = Path('experiments/dpo')
DPO_YAML = DPO_EXPERIMENT_DIR / 'qwen_dpo_baseline.yaml'
DPO_CHECKPOINT_DIR = Path(f'results/checkpoints/{DPO_RUN_NAME}')
DPO_HF_DIR = Path(f'results/checkpoints/{DPO_RUN_NAME}-hf')

EVAL_OUT_DIR = Path('results/evals/dpo_phase1_run_1')
EVAL_RUN_NAME = EVAL_OUT_DIR.name

SOURCES = ['gsm8k', 'math']
HELDOUT_TRAIN_N = 2000
SAMPLES_PER_PROMPT = 8
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 768
SEED = 42

DTYPE = 'bfloat16'
MAX_SEQ_LEN = 1024
DPO_MAX_STEPS = 1000
DPO_CANARY_STEPS = 50
DPO_LR = 5.0e-6
DPO_BETA = 0.1
PAIR_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
WARMUP_STEPS = 100
CHECKPOINT_EVERY_N_STEPS = 250
CHECKPOINT_RETENTION_LAST_N = 4

EVAL_N = 500
EVAL_SEED = 42
BATCH_SIZE_GSM8K = 128
BATCH_SIZE_MATH = 128
GPU_COST_PER_HOUR = 0.44

print('SFT HF repo:       ', SFT_HF_REPO)
print('SFT subfolder:     ', SFT_HF_SUBFOLDER)
print('pair out dir:      ', PAIR_OUT_DIR)
print('DPO run name:      ', DPO_RUN_NAME)
print('DPO yaml:          ', DPO_YAML)
print('DPO ckpt dir:      ', DPO_CHECKPOINT_DIR)
print('DPO HF dir:        ', DPO_HF_DIR)
print('eval out dir:      ', EVAL_OUT_DIR)
print('effective pair batch:', PAIR_BATCH_SIZE * GRAD_ACCUM_STEPS)
'''

FETCH_CELL = '''\
# Download the chosen SFT checkpoint from HuggingFace Hub to the pod volume.
# snapshot_download is idempotent: if the files are already present it skips
# the network transfer. The write token in your pod env vars has read access
# too, so no re-authentication is needed.
#
# The repo mirrors the subfolder structure, so files land at:
#   _dl_root / combined / step-0500 / config.json
#   _dl_root / combined / step-0500 / model.safetensors
#   ...
# We then reassign SFT_HF_CHECKPOINT to that leaf directory.
from huggingface_hub import snapshot_download

_dl_root = Path('results/checkpoints/_sft_hf_cache')
print(f'Fetching {SFT_HF_REPO}/{SFT_HF_SUBFOLDER} ...')
snapshot_download(
    repo_id=SFT_HF_REPO,
    allow_patterns=f'{SFT_HF_SUBFOLDER}/**',
    local_dir=str(_dl_root),
    local_dir_use_symlinks=False,
)

SFT_HF_CHECKPOINT = _dl_root / SFT_HF_SUBFOLDER
print('SFT checkpoint ready at:', SFT_HF_CHECKPOINT)
for p in sorted(SFT_HF_CHECKPOINT.iterdir()):
    print(' ', p.name)
'''


def _split_source(text: str) -> list[str]:
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

    # 1. Update step-2 markdown
    md_idx = _find_cell(cells, 'dpo-step-2')
    cells[md_idx]['source'] = _split_source(NEW_STEP2_MD)
    print('updated dpo-step-2 markdown')

    # 2. Update constants cell
    const_idx = _find_cell(cells, 'dpo-constants')
    cells[const_idx]['source'] = _split_source(NEW_CONSTANTS)
    cells[const_idx]['outputs'] = []
    cells[const_idx]['execution_count'] = None
    print('updated dpo-constants')

    # 3. Insert fetch cell between dpo-constants and dpo-step-3
    fetch_id = 'dpo-fetch-sft-ckpt'
    if any(c.get('id') == fetch_id for c in cells):
        print('dpo-fetch-sft-ckpt already present, skipping insert')
    else:
        insert_at = const_idx + 1
        fetch_cell = {
            'cell_type': 'code',
            'id': fetch_id,
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': _split_source(FETCH_CELL),
        }
        cells.insert(insert_at, fetch_cell)
        print(f'inserted dpo-fetch-sft-ckpt at index {insert_at}')

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'wrote {NB_PATH}')


if __name__ == '__main__':
    main()
