# Security

FinPost loads external datasets, tokenizers, and model weights through defensive
wrappers in `src/finpost/safety.py`.

Public-facing defaults:

- `trust_remote_code=False` for Hugging Face datasets, tokenizers, and models.
- `use_safetensors=True` for model loads by default.
- HF tokens, W&B keys, `.env` files, caches, model weights, checkpoints,
  private source catalogs, and intermediate generated datasets must stay out of
  git. The audited FinChain v2 split under `data/finchain_v2_template_disjoint/`
  is intentionally public.

If a future run needs remote code or pickle-format weights, document the reason
in the change that introduces it and keep the exception scoped to that run.
