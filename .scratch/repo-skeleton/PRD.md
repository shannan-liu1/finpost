# 0001. Repository skeleton and environment

- **Status:** Done (verified 2026-05-05)
- **Created:** 2026-05-05
- **Owner:** Shannan
- **Estimated time:** ~30 minutes
- **Depends on:** none

## Goal

Establish the directory layout, dependency manifest, and packaging metadata so the rest of the project has a stable foundation. After this PRD, anyone with a Python 3.11+ environment can clone the repo, run one install command, and import `finpost`.

## Scope

**In scope:**
- Top-level directory structure (`src/finpost/`, `tests/`, `scripts/`, `experiments/`, `notebooks/`, `data/`, `results/`, `docs/`).
- `pyproject.toml` declaring the package, Python version, and dependencies (with optional extras for parameter-efficient fine-tuning, the Direct Preference Optimization reference library, and EDGAR tooling).
- `.gitignore` excluding bulky and ephemeral content (`data/`, `results/`, virtual environments, caches).
- A minimal `src/finpost/__init__.py` exposing `__version__`.
- A `README.md` at the repo root with one-paragraph overview, install instructions, and a tree of the directory layout.
- Placeholder `.gitkeep` files (or `README.md` files) inside otherwise-empty tracked directories so they survive a fresh clone.

**Out of scope:**
- Any source code beyond `__init__.py`. Real modules are added in later PRDs.
- Continuous integration configuration. Add later if needed.
- Pre-commit hooks. Add later if needed.

## Deliverables

```
finpost/
├── README.md                           # overview, install, layout
├── pyproject.toml                      # package metadata + dependencies
├── .gitignore                          # excludes data/, results/, venvs, caches
├── CONTEXT.md                          # already exists
├── PLAN.md                             # already exists
├── PRDs/                               # already exists (this PRD lives here)
├── src/
│   └── finpost/
│       └── __init__.py                 # exposes __version__
├── tests/
│   └── .gitkeep
├── scripts/
│   └── .gitkeep
├── experiments/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── data/                               # gitignored content; only README.md tracked
│   └── README.md
├── results/                            # gitignored content; only README.md tracked
│   └── README.md
└── docs/
    └── .gitkeep
```

## Acceptance criteria

Run from the repo root in a fresh Python 3.11 virtual environment:

1. `pip install -e .` exits with status 0.
2. `python -c "import finpost; print(finpost.__version__)"` prints `0.0.1`.
3. `python -c "import finpost; print(finpost.__file__)"` prints a path under `src/finpost/`.
4. `git status` after install is clean (no new tracked files appear unexpectedly; `*.egg-info/` is gitignored).
5. The directory tree above exists exactly as specified.

## Notes / open questions

- `bitsandbytes` (in the `peft` optional extra) is known to be finicky on Windows. Local development on Windows is fine without the extra; install it on the rented A100 instance instead.
- We may add a `pre-commit` config later for automatic formatting (`ruff`). Defer until we have enough code to make it useful.
