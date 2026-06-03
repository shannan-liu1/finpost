from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_core_modules_import() -> None:
    import finpost
    import finpost.data.finchain_dataset
    import finpost.evals.finchain_metrics
    import finpost.training.finchain_rlvr

    assert finpost.__version__
