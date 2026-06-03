# Public Eval Summaries

Compact result summaries mirrored from experiment artifacts. Hugging Face is the source for model weights and any eval paths not mirrored here.

**Included per experiment:**
- `selection_metadata.json` — which checkpoint was selected on validation and why
- `accuracy_summary.json` / `accuracy_summary.csv` — aggregate accuracy, parse success, and step metrics
- `cost_summary.json` — GPU time and throughput

**Not included:**
- Per-example `details_*.csv` generation dumps
- Model weights and checkpoint folders
- DPO pair/completion JSONLs

The DPO summaries are kept as diagnostic evidence for an underperforming run — they document the evaluation, not an improvement. See `docs/hf_public_metrics.md` for the full metrics table.
