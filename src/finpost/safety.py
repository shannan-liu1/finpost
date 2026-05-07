"""Safe wrappers around external loaders.

The functions in this module enforce defensive defaults — primarily,
disabling execution of remote loader scripts and pickle-format model
weights. Always call these instead of the underlying library APIs
directly. Defaults can be overridden by passing explicit keyword
arguments, but the override should be deliberate and recorded in the
PRD or commit that introduces it.

See SECURITY.md for the full policy and the incident history that
motivates these defaults.
"""

from __future__ import annotations

from typing import Any

# Imported under a private alias so callers cannot accidentally bypass
# the wrapper by writing `from finpost.safety import load_dataset`.
from datasets import load_dataset as _hf_load_dataset


def safe_load_dataset(
    path: str,
    name: str | None = None,
    *,
    split: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load a Hugging Face dataset with defensive defaults.

    Parameters
    ----------
    path
        The dataset identifier on the Hugging Face Hub
        (e.g. "openai/gsm8k").
    name
        The dataset configuration name (e.g. "main" for GSM8K).
    split
        The split to load ("train", "test", or None for all).
    revision
        Git ref to pin against. For datasets that ship a Python loader
        script, pass "refs/convert/parquet" to use the auto-converted
        parquet branch instead of the script-based main branch.
    trust_remote_code
        Defaults to False — script-based datasets fail to load. Set to
        True only after auditing the loader script. Passing True must
        be deliberate; a justification belongs in the PRD or commit.
    **kwargs
        Forwarded unchanged to ``datasets.load_dataset``.

    Notes
    -----
    Both ``datasets >= 2.16`` defaults ``trust_remote_code`` to False,
    but we still pass it explicitly here. Defense in depth: if the
    upstream default ever changes, our policy is unaffected.
    """
    return _hf_load_dataset(
        path=path,
        name=name,
        split=split,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
