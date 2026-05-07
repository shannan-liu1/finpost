# results/

Training checkpoints and evaluation outputs live here. The contents of this directory are gitignored — only this README is tracked.

Conventional layout (created on demand by training and eval code):

```
results/
├── checkpoints/
│   └── <experiment_name>/<step>.safetensors
├── eval/
│   └── <experiment_name>/<dataset>.jsonl
└── tables/
    └── <experiment_name>.csv
```

Nothing here should be irreproducible. Reproducibility is via the configuration in `experiments/`, the seed recorded in the run, and the code at the commit recorded in the run.
