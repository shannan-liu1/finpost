# data/

Raw and processed data lives here. The contents of this directory are gitignored — only this README is tracked, so the directory survives a fresh clone with its purpose documented.

Conventional layout (created on demand by data-loading code):

```
data/
├── raw/
│   └── <ticker>/<period>/<form>.html     # raw EDGAR filings (Phase 2)
├── processed/
│   ├── gsm8k/                             # cached normalized records
│   ├── math/
│   └── finance/
└── synth/                                 # teacher-distilled examples (Phase 2)
```

Nothing here should be irreproducible. Anything you cannot re-download or re-generate from code does not belong in this directory.
