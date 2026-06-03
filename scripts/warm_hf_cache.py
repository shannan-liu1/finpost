"""Pre-download Hugging Face model snapshots into HF_HOME.

This avoids the expensive ambiguity where a distributed launch appears hung but
is actually downloading model weights. The script downloads tokenizer/config
files and safetensors without loading the model onto GPU.

Example:
    HF_HOME=/workspace/hf-cache python scripts/warm_hf_cache.py \
      Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ids", nargs="+", help="HF model ids to cache")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="Cache root. Defaults to HF_HOME when set.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    allow_patterns = [
        "*.json",
        "*.safetensors",
        "tokenizer*",
        "vocab.json",
        "merges.txt",
        "*.model",
    ]
    for model_id in args.model_ids:
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=args.cache_dir,
            allow_patterns=allow_patterns,
        )
        print(f"cached {model_id} -> {path}")


if __name__ == "__main__":
    main()
