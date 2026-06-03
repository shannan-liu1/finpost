"""Vendored ChainEval scorer for FinChain reasoning traces.

Provenance
----------
The scoring functions in this module are derived from upstream FinChain:

- Paper: "FinChain: A Symbolic Benchmark of Verifiable Chain-of-Thought
  Financial Reasoning", Hossain et al., 2025 — arXiv:2506.02515
- Repo:  https://github.com/mbzuai-nlp/finchain
- File:  ``chaineval/evaluate_predictions.py``
- Pinned commit on file: ``146eaa8225bf867fe7386c2d4727b59e03170235``
  (fetched 2026-05-21)

The upstream repository has NO LICENSE file and the readme's "License"
section is empty as of the fetched commit. We vendor here under a
good-faith MIT assumption per the spec's "if uncertain, attribute clearly
and assume MIT" guidance. If the upstream authors clarify their license
in a way that conflicts, this module needs to be re-licensed or removed.

What is vendored verbatim
-------------------------
- Module-level constants: ``ALIGN_THRESHOLD``, ``VALUE_REL_TOL``,
  ``FINAL_REL_TOL``, ``DTW_ALPHA_SIM``, ``DTW_BETA_NUM``,
  ``DTW_SIM_ACCEPT``, ``DTW_GAP_PENALTY``.
- The step-splitting regex patterns in ``split_trace_into_steps``.
- The final-value extraction patterns in ``extract_final_value``.
- The value-match mask logic in ``build_value_match_mask``.
- The DP alignment math in ``align_with_stats`` and the score / cost
  matrix construction in ``compute_dtw_metrics``.

What is intentionally different from upstream
---------------------------------------------
- **Lazy heavy imports.** Upstream loads ``sentence_transformers``,
  ``bert_score``, the Longformer tokenizer, and ``rouge_score`` at
  module-import time. That makes simply importing the module pull
  ~300MB of model weights. Here, those imports happen inside helper
  getters (``_get_sbert``, ``_get_bertscorer``, ``_get_rouge``,
  ``_get_longformer_tokenizer``) and the loaded objects are cached in
  module-level globals so subsequent calls are O(1).
- **Optional-extra gating.** A try/import block at the top sets
  ``CHAINEVAL_AVAILABLE``. If the extras are missing, ``score_trace``
  raises ``ChainEvalNotAvailable`` with the pip install hint instead
  of crashing inside an internal import.
- **No scikit-learn dependency.** Upstream imports
  ``sklearn.metrics.pairwise.cosine_similarity``; we use a 2-line numpy
  implementation (normalize rows, then matrix-multiply) so the extra
  stays narrow.
- **Field renames in the returned dict.** Upstream returns ``recall`` /
  ``precision`` / ``final_answer_match``; we rename to ``step_recall``
  / ``step_precision`` / ``chaineval_final_match`` so downstream eval
  CSVs (which already carry FinChain "recall"-flavored columns
  in the future) cannot collide.
- **``score_trace`` returns a flat dict** with all ~17 fields including
  DTW. Upstream returns a tuple from ``score_trace`` and a separate
  dict from ``compute_dtw_metrics``; we fold both so the eval CLI can
  call one function per row.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

# numpy is required for the mask matrix and DP alignment math. It is part of
# the core finpost dependencies, so we import unconditionally.
import numpy as np

# =============================================================================
# Optional-extra availability gate.
# =============================================================================
#
# These three packages are only needed inside ``score_trace`` (rouge scoring,
# sentence embeddings, BERTScore). The light helpers
# (``split_trace_into_steps``, ``extract_final_value``,
# ``build_value_match_mask``, ``align_with_stats``) never touch them and run
# on plain numpy. We gate the heavy imports here so simply importing
# ``finpost.evals.chaineval`` does not require the extras.
#
# Set CHAINEVAL_AVAILABLE based on whether the three packages can be
# imported. score_trace raises ChainEvalNotAvailable when the flag is False.

try:
    import bert_score  # noqa: F401
    import rouge_score  # noqa: F401
    import sentence_transformers  # noqa: F401

    CHAINEVAL_AVAILABLE: bool = True
except ImportError:
    CHAINEVAL_AVAILABLE = False


class ChainEvalNotAvailable(RuntimeError):
    """Raised when score_trace is called without the [chaineval] extras.

    Tells the user how to install the extras so they do not have to
    cross-reference pyproject.toml.
    """

    DEFAULT_MESSAGE = (
        "ChainEval scoring requires extra dependencies that are not installed. "
        'Run: pip install -e ".[chaineval]" '
        "(or, for the full eval stack: "
        'pip install -e ".[dev,rlvr,chaineval]")'
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


# =============================================================================
# Constants — pinned verbatim to upstream values.
#
# These thresholds are part of the ChainEval contract; changing any of them
# changes the published metric definition. Bump only when intentionally
# tracking an upstream change and document the bump in the docstring above.
# =============================================================================

ALIGN_THRESHOLD = 0.45  # min cosine similarity to count a step as aligned
VALUE_REL_TOL = 0.15  # relative tolerance for intermediate-value match
FINAL_REL_TOL = 0.05  # relative tolerance for final-answer match
DTW_ALPHA_SIM = 0.85  # bonus DTW: weight on cosine similarity
DTW_BETA_NUM = 0.15  # bonus DTW: weight on numeric agreement
DTW_SIM_ACCEPT = 0.45  # DTW path-score above this counts as a matched pair
DTW_GAP_PENALTY = 0.25  # DP gap cost (insert/delete a step)

# Longformer tokenizer truncation budget; matches upstream MAX_TOKENS.
_MAX_TOKENS = 4096
_SENTENCE_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_LONGFORMER_MODEL = "allenai/longformer-base-4096"


# =============================================================================
# Lazy model loaders. Cache loaded objects so subsequent calls are O(1).
#
# These functions all raise ChainEvalNotAvailable when CHAINEVAL_AVAILABLE is
# False so the caller sees a single, informative error instead of an
# ImportError from deep inside the import chain.
# =============================================================================

_sbert_cache: Any = None
_bertscorer_cache: Any = None
_rouge_cache: Any = None
_longformer_tokenizer_cache: Any = None


def _get_sbert() -> Any:
    """Return a cached SentenceTransformer all-MiniLM-L6-v2 embedder."""
    global _sbert_cache
    if not CHAINEVAL_AVAILABLE:
        raise ChainEvalNotAvailable()
    if _sbert_cache is None:
        from sentence_transformers import SentenceTransformer

        _sbert_cache = SentenceTransformer(_SENTENCE_EMBEDDER_NAME)
    return _sbert_cache


def _get_bertscorer() -> Any:
    """Return a cached BERTScorer using the Longformer model."""
    global _bertscorer_cache
    if not CHAINEVAL_AVAILABLE:
        raise ChainEvalNotAvailable()
    if _bertscorer_cache is None:
        # Pick CUDA when available — BERTScorer is slow on CPU. We import
        # torch lazily here too; torch is a core finpost dep so the import
        # itself is cheap, but defer it so the module is fully usable
        # without touching CUDA when only the light helpers are called.
        import torch
        from bert_score import BERTScorer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bertscorer_cache = BERTScorer(model_type=_LONGFORMER_MODEL, device=device)
    return _bertscorer_cache


def _get_rouge() -> Any:
    """Return a cached rouge_score RougeScorer over (rouge1, rouge2, rougeL, rougeLsum)."""
    global _rouge_cache
    if not CHAINEVAL_AVAILABLE:
        raise ChainEvalNotAvailable()
    if _rouge_cache is None:
        from rouge_score import rouge_scorer as rouge_lib

        _rouge_cache = rouge_lib.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
        )
    return _rouge_cache


def _get_longformer_tokenizer() -> Any:
    """Return a cached Longformer AutoTokenizer for input truncation."""
    global _longformer_tokenizer_cache
    if not CHAINEVAL_AVAILABLE:
        raise ChainEvalNotAvailable()
    if _longformer_tokenizer_cache is None:
        from finpost.safety import safe_load_tokenizer

        _longformer_tokenizer_cache = safe_load_tokenizer(_LONGFORMER_MODEL)
    return _longformer_tokenizer_cache


# =============================================================================
# Light helpers: pure numpy / regex, no heavy deps.
# =============================================================================


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-pair cosine similarity between matrix ``a`` (n, d) and ``b`` (m, d).

    Replaces ``sklearn.metrics.pairwise.cosine_similarity`` so this module
    does not pull scikit-learn into the optional extra. Mathematically
    equivalent: normalize each row to unit L2 norm, then matrix-multiply.

    A tiny epsilon (1e-12) is added to the denominator to avoid division
    by zero on a zero-vector row — that case yields a row of zeros,
    matching sklearn's documented behavior (norm=0 returns a zero row).
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / (a_norm + 1e-12)
    b_unit = b / (b_norm + 1e-12)
    return a_unit @ b_unit.T


def bag_of_words_cosine(
    text_a: str,
    text_b: str,
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float | None:
    """Bag-of-words cosine similarity between two whitespace-tokenized strings.

    Returns the cosine in ``[0, 1]`` if it falls inside ``[min_val, max_val]``,
    else ``None``. ``None`` is also returned when either side has zero
    L2 norm (empty string after whitespace split).

    Verbatim port of the upstream helper. Used by ``build_value_match_mask``
    and by ``score_trace`` when comparing string-valued final answers.
    """
    counts_a, counts_b = Counter(text_a.split()), Counter(text_b.split())
    vocab = set(counts_a) | set(counts_b)
    dot = sum(counts_a[token] * counts_b[token] for token in vocab)
    norm_a = math.sqrt(sum(counts_a[token] ** 2 for token in vocab))
    norm_b = math.sqrt(sum(counts_b[token] ** 2 for token in vocab))
    if not norm_a or not norm_b:
        return None
    similarity = dot / (norm_a * norm_b)
    if min_val <= similarity <= max_val:
        return similarity
    return None


def parse_numeric_value(text: str) -> float | None:
    """Extract the first number from ``text`` and apply M/B/K magnitude scaling.

    Verbatim port of the upstream helper. Strips ``$``, ``~``, and ``,``,
    then takes the first ``[\\d.]+`` match. The magnitude keyword
    (``million``, ``billion``, ``thousand``) is searched in the lowercased
    raw text.
    """
    cleaned = text.replace("~", "").replace("$", "").replace(",", "").strip()
    matches = re.findall(r"[\d.]+", cleaned)
    if not matches:
        return None
    try:
        value = float(matches[0])
    except ValueError:
        return None

    lowered = text.lower()
    if "billion" in lowered:
        value *= 1_000_000_000
    elif "million" in lowered:
        value *= 1_000_000
    elif "thousand" in lowered:
        value *= 1_000
    return value


def extract_final_value(step_text: str) -> Any:
    """Pull the most-final-looking numeric value out of a step.

    Tries four regex patterns in order:

    1. ``**Answer:** <number>`` (markdown-emphasized answer label)
    2. ``= <number>`` (equation rightmost equality)
    3. Any bare number with an optional magnitude keyword
    4. Inline ``answer:`` / ``final answer:`` markers

    Returns ``float`` when ``parse_numeric_value`` succeeds; the raw
    matched substring (``str``) when the regex hit a value
    ``parse_numeric_value`` could not turn into a float; ``None`` when
    no pattern hit at all.

    Verbatim port of the upstream helper.
    """
    normalized = re.sub(r"\s+", " ", step_text).strip()

    patterns = [
        r"\**Answer:\**\s*.*?(?:USD|\$)?\s*([\d,]+(?:\.\d{1,2})?)",
        r"=\s*(?!.*=)(?:USD|\$)?\s*[\d,]+(?:\.\d{1,2})?\s*(million|billion|thousand)?",
        r"\d[\d,]+(?:\.\d{1,2})?\s*(million|billion|thousand)?",
        r"(?<=\n|\.|:)\s*(answer: |final answer: |final answer is: )?"
        r"([\d,]+(?:\.\d{1,2})?\s*(million|billion|thousand)?)",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, normalized.lower()))
        if matches:
            candidate = matches[-1].group(0)
            value = parse_numeric_value(candidate)
            return value if value is not None else candidate.strip()
    return None


def split_trace_into_steps(
    trace_text: str | None,
) -> tuple[list[str], list[Any], Any]:
    """Split a reasoning trace into ordered step strings plus values.

    Tries three regex patterns in order:

    1. ``Step N`` with optional asterisks/hashes/dashes (default FinChain).
    2. ``\\n<num>`` numeric-prefix steps.
    3. ``Step:`` unnumbered markers.

    Falls back to splitting by lines when no pattern matches at all so
    callers never receive an empty list on a non-empty trace.

    Returns
    -------
    (steps, values, final_value)
        ``steps`` is the cleaned step strings (whitespace collapsed,
        ``Step N:`` prefix stripped). ``values`` is ``extract_final_value``
        applied to each step. ``final_value`` is the last non-None entry
        in ``values`` (or None when every step had no extractable value).

    Verbatim port of the upstream helper. Returns three empty/None
    results when ``trace_text`` is not a string, matching upstream.
    """
    if not isinstance(trace_text, str):
        return [], [], None

    # Upstream strips a trailing ``\nuser\n`` section that some chat-templated
    # generations include. We preserve that behaviour.
    content = trace_text.split("\nuser\n")[0] if "\nuser\n" in trace_text else trace_text

    step_matches = list(
        re.finditer(r"(?:^|\n)(\s*)(\**)(#*)(\s*)Step(-*)(\s*)(\d+)", content)
    )
    if not step_matches:
        step_matches = list(re.finditer(r"(?:^|\n)\s*(\**)(\d+)(.*)", content))
    if not step_matches:
        step_matches = list(re.finditer(r"(?:^|\n)(\s*)Step\s*:", content, re.IGNORECASE))

    if step_matches:
        indices = [match.span()[0] for match in step_matches]
        spans = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
        spans.append((indices[-1], len(content)))
        raw_steps = [content[start:end].strip() for start, end in spans]
    else:
        # Last-resort fallback: one step per non-empty line.
        raw_steps = [ln.strip() for ln in content.splitlines() if ln.strip()]

    cleaned_steps = [re.sub(r"(?i)Step\s*\d*\s*:?", "", step).strip() for step in raw_steps]
    steps = [re.sub(r"\s+", " ", step) for step in cleaned_steps if step]

    values = [extract_final_value(step) for step in steps]
    final_value = next((value for value in reversed(values) if value is not None), None)
    return steps, values, final_value


def build_value_match_mask(
    gold_values: list[Any],
    pred_values: list[Any],
) -> np.ndarray:
    """Build the value-match mask matrix used to gate the cosine similarity.

    Note: the input is the list of *extracted values* per step (from
    ``split_trace_into_steps``), not the step text. The spec uses
    ``gold_steps``/``pred_steps`` colloquially but the upstream
    semantics — preserved here — takes values.

    Returns a float matrix of shape ``(len(gold_values), len(pred_values))``
    where entry ``[i, j]`` is 1.0 when:

    - ``gold_values[i] is None``, OR
    - ``pred_values[j] is None``, OR
    - both are strings and their bag-of-words cosine >= 0.3, OR
    - both are numeric and ``|gold - pred| / (|gold| + 1e-4) < VALUE_REL_TOL``

    and 0.0 otherwise.

    The "None means allow" branch lets the cosine alone decide alignment
    on steps whose numeric content the parser could not extract.
    """
    mask = np.zeros((len(gold_values), len(pred_values)), dtype=float)
    for i, gold in enumerate(gold_values):
        if gold is None:
            mask[i, :] = 1.0
            continue
        for j, pred in enumerate(pred_values):
            if pred is None:
                mask[i, j] = 1.0
                continue
            if isinstance(gold, str) or isinstance(pred, str):
                if bag_of_words_cosine(str(gold), str(pred), min_val=0.3):
                    mask[i, j] = 1.0
            else:
                denominator = abs(gold) + 1e-4
                if abs(gold - pred) / denominator < VALUE_REL_TOL:
                    mask[i, j] = 1.0
    return mask


# =============================================================================
# BERTScore wrapper. Heavy.
# =============================================================================


def _compute_bert_score(reference: Any, candidate: Any) -> float:
    """Compute the BERTScore F1 between two strings, truncating to MAX_TOKENS.

    Returns 0.0 when either side is empty or not a string. Returns 0.0
    when the underlying scorer raises (per upstream's defensive try/except).
    """
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return 0.0
    if not reference.strip() or not candidate.strip():
        return 0.0

    tokenizer = _get_longformer_tokenizer()
    scorer = _get_bertscorer()

    ref_tokens = tokenizer(
        reference, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS
    )
    cand_tokens = tokenizer(
        candidate, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS
    )
    try:
        ref_text = tokenizer.batch_decode(ref_tokens["input_ids"], skip_special_tokens=True)[0]
        cand_text = tokenizer.batch_decode(cand_tokens["input_ids"], skip_special_tokens=True)[0]
        _, _, f1 = scorer.score([ref_text], [cand_text])
        return float(f1[0].item())
    except Exception:
        return 0.0


# =============================================================================
# DTW alignment — pure numpy DP plus a small score-from-matrix helper.
# =============================================================================


def _numeric_or_string_agree(a: Any, b: Any) -> float:
    """Pairwise value-agreement score used by the DTW ``bonus`` matrix.

    Returns 1.0 when either side is missing (so text similarity can drive
    alignment), 1.0 / 0.0 from the bag-of-words cosine when either side
    is a string, and a numeric within-tolerance check otherwise. Verbatim
    port of upstream.
    """
    if a is None or b is None:
        return 1.0
    if isinstance(a, str) or isinstance(b, str):
        return 1.0 if bag_of_words_cosine(str(a), str(b), min_val=0.3) else 0.0
    return 1.0 if abs(a - b) / (abs(a) + 1e-4) < VALUE_REL_TOL else 0.0


def align_with_stats(
    cost: np.ndarray,
    gap_cost: float,
) -> tuple[list[tuple[int, int]], int, float]:
    """Dynamic-programming alignment with insert/delete gap penalties.

    Returns the matched (i, j) pairs along the optimal path, the path
    length (including gap moves), and the total DP cost. Verbatim port
    of upstream's ``align_with_stats``.
    """
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    backtrack = np.zeros((n + 1, m + 1), dtype=int)
    dp[0, 0] = 0.0

    # Border initialization: cumulative gap cost along the top/left.
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_cost
        backtrack[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_cost
        backtrack[0, j] = 3

    # Standard Needleman-Wunsch-style fill: best of (diagonal, up, left).
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                dp[i - 1, j - 1] + cost[i - 1, j - 1],
                dp[i - 1, j] + gap_cost,
                dp[i, j - 1] + gap_cost,
            ]
            choice = int(np.argmin(candidates))
            dp[i, j] = candidates[choice]
            backtrack[i, j] = [1, 2, 3][choice]

    # Walk the backtrack matrix to extract matched pairs. Gap moves do not
    # contribute pairs but do count toward path_len.
    i, j = n, m
    path: list[tuple[int, int]] = []
    path_len = 0
    while i > 0 or j > 0:
        move = backtrack[i, j]
        path_len += 1
        if move == 1:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path, path_len, float(dp[n, m])


def _build_similarity_and_bonus(
    gold_steps: list[str],
    pred_steps: list[str],
    gold_values: list[Any],
    pred_values: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the (similarity, bonus) matrices used by ``compute_dtw_metrics``.

    Similarity is the sentence-transformer cosine, clipped to [0, 1].
    Bonus is the value-agreement matrix from ``_numeric_or_string_agree``.
    """
    if not gold_steps or not pred_steps:
        empty = np.zeros((len(gold_steps), len(pred_steps)))
        return empty, empty

    embedder = _get_sbert()
    gold_emb = embedder.encode(gold_steps)
    pred_emb = embedder.encode(pred_steps)
    # Convert to ndarray defensively in case the embedder returns a list of arrays.
    gold_emb = np.asarray(gold_emb, dtype=float)
    pred_emb = np.asarray(pred_emb, dtype=float)
    similarity = np.clip(_cosine_similarity_matrix(gold_emb, pred_emb), 0.0, 1.0)

    bonus = np.zeros_like(similarity)
    for i in range(len(gold_steps)):
        for j in range(len(pred_steps)):
            bonus[i, j] = _numeric_or_string_agree(gold_values[i], pred_values[j])
    return similarity, bonus


def _dtw_metrics_from_score(
    score: np.ndarray,
    pairs: list[tuple[int, int]],
    n_gold: int,
    n_pred: int,
) -> dict[str, float]:
    """Reduce a path through the score matrix into precision/recall/F1/path stats."""
    matched_gold = set()
    matched_pred = set()
    path_scores: list[float] = []

    for i, j in pairs:
        value = score[i, j]
        path_scores.append(float(value))
        if value >= DTW_SIM_ACCEPT:
            matched_gold.add(i)
            matched_pred.add(j)

    precision = len(matched_pred) / (n_pred + 1e-9)
    recall = len(matched_gold) / (n_gold + 1e-9)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    avg_path_score = float(np.mean(path_scores)) if path_scores else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_path_score": avg_path_score,
    }


def compute_dtw_metrics(
    gold_steps: list[str],
    pred_steps: list[str],
    gold_values: list[Any],
    pred_values: list[Any],
) -> dict[str, dict[str, float]]:
    """Return DTW alignment metrics in the ``{"bonus": ..., "gate": ...}`` shape.

    Two variants are produced from the same similarity / bonus matrices:

    - **bonus**: ``score = clip(alpha * similarity + beta * bonus, 0, 1)``.
      Numeric agreement adds to similarity rather than gating it.
    - **gate**: ``score = clip(similarity * bonus, 0, 1)``.
      Numeric disagreement zeros the cell, gating the alignment strictly.

    Each returns precision, recall, f1, avg_path_score, and norm_score.

    Deviation from spec note: the spec said this function takes
    ``(gold_steps, pred_steps, mask)``. Upstream actually rebuilds its
    own ``bonus`` matrix from values, and the mask used by ``score_trace``
    is mathematically different from this ``bonus``. We follow the
    upstream signature (taking step strings plus value lists) so the
    numerics are bit-equivalent to upstream rather than approximating.
    """
    n, m = len(gold_steps), len(pred_steps)
    if n == 0 or m == 0:
        zero = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_path_score": 0.0,
            "norm_score": 0.0,
        }
        return {"bonus": dict(zero), "gate": dict(zero)}

    similarity, bonus = _build_similarity_and_bonus(
        gold_steps, pred_steps, gold_values, pred_values
    )

    # Variant 1: additive bonus.
    score_bonus = np.clip(DTW_ALPHA_SIM * similarity + DTW_BETA_NUM * bonus, 0.0, 1.0)
    cost_bonus = 1.0 - score_bonus
    pairs_bonus, len_bonus, total_cost_bonus = align_with_stats(
        cost_bonus, gap_cost=DTW_GAP_PENALTY
    )
    metrics_bonus = _dtw_metrics_from_score(score_bonus, pairs_bonus, n, m)
    metrics_bonus["norm_score"] = float(1.0 - (total_cost_bonus / max(1, len_bonus)))

    # Variant 2: multiplicative gate.
    score_gate = np.clip(np.multiply(similarity, bonus), 0.0, 1.0)
    cost_gate = 1.0 - score_gate
    pairs_gate, len_gate, total_cost_gate = align_with_stats(
        cost_gate, gap_cost=DTW_GAP_PENALTY
    )
    metrics_gate = _dtw_metrics_from_score(score_gate, pairs_gate, n, m)
    metrics_gate["norm_score"] = float(1.0 - (total_cost_gate / max(1, len_gate)))

    return {"bonus": metrics_bonus, "gate": metrics_gate}


# =============================================================================
# Top-level score_trace — flattens ChainEval + DTW into one dict per row.
# =============================================================================


def score_trace(gold: str | None, pred: str | None) -> dict[str, float]:
    """Score one (gold, pred) trace pair on the full ChainEval metric set.

    Returns a flat dict with these keys:

    - ``step_recall`` (renamed from upstream ``recall``)
    - ``step_precision`` (renamed from upstream ``precision``)
    - ``chaineval_final_match`` (renamed from upstream ``final_answer_match``)
    - ``rouge2``, ``rougeL``, ``rougeLsum``
    - ``bertscore``
    - ``dtw_{precision,recall,f1,avg_path_score,norm_score}_{bonus,gate}``
      (ten DTW columns)

    The renames avoid name clashes with future per-source ``recall`` /
    ``precision`` columns (e.g., a FinChain retrieval-style metric) in
    downstream eval CSVs.

    Raises
    ------
    ChainEvalNotAvailable
        When the ``[chaineval]`` extras are not installed. The light
        helpers in this module (``split_trace_into_steps`` etc.) remain
        importable and callable; only ``score_trace`` requires the heavy
        stack.
    """
    if not CHAINEVAL_AVAILABLE:
        raise ChainEvalNotAvailable()

    # On non-string inputs, return a zero-filled record. Matches upstream's
    # defensive contract — eval rows occasionally carry None in either side.
    if not isinstance(gold, str) or not isinstance(pred, str):
        return _zero_metrics()

    rouge = _get_rouge()
    rouge_scores = rouge.score(gold, pred)
    rouge2 = float(rouge_scores["rouge2"].fmeasure)
    rouge_l = float(rouge_scores["rougeL"].fmeasure)
    rouge_lsum = float(rouge_scores["rougeLsum"].fmeasure)
    bert = float(_compute_bert_score(gold, pred))

    gold_steps, gold_values, gold_final = split_trace_into_steps(gold)
    pred_steps, pred_values, pred_final = split_trace_into_steps(pred)

    if not gold_steps or not pred_steps:
        result = _zero_metrics()
        result.update(
            {
                "rouge2": rouge2,
                "rougeL": rouge_l,
                "rougeLsum": rouge_lsum,
                "bertscore": bert,
            }
        )
        return result

    # Step-level recall/precision: cosine similarity gated by value-match mask.
    embedder = _get_sbert()
    gold_emb = np.asarray(embedder.encode(gold_steps), dtype=float)
    pred_emb = np.asarray(embedder.encode(pred_steps), dtype=float)
    sentence_sim = _cosine_similarity_matrix(gold_emb, pred_emb)

    mask = build_value_match_mask(gold_values, pred_values)
    masked_sim = np.multiply(sentence_sim, mask)

    step_recall = float(
        np.sum(np.max(masked_sim, axis=1) > ALIGN_THRESHOLD) / len(gold_steps)
    )
    step_precision = float(
        np.sum(np.max(masked_sim, axis=0) > ALIGN_THRESHOLD) / len(pred_steps)
    )

    # Final-answer match — binary; same three-branch contract as upstream.
    if gold_final is None or pred_final is None:
        final_match = 0
    elif isinstance(gold_final, str) or isinstance(pred_final, str):
        final_match = (
            1 if bag_of_words_cosine(str(gold_final), str(pred_final), min_val=0.1) else 0
        )
    else:
        final_match = int(
            abs(gold_final - pred_final) / (abs(gold_final) + 1e-4) < FINAL_REL_TOL
        )

    # DTW — runs on the already-split steps / values so we don't re-tokenize.
    dtw = compute_dtw_metrics(gold_steps, pred_steps, gold_values, pred_values)

    return {
        "step_recall": step_recall,
        "step_precision": step_precision,
        "chaineval_final_match": float(final_match),
        "rouge2": rouge2,
        "rougeL": rouge_l,
        "rougeLsum": rouge_lsum,
        "bertscore": bert,
        "dtw_precision_bonus": dtw["bonus"]["precision"],
        "dtw_recall_bonus": dtw["bonus"]["recall"],
        "dtw_f1_bonus": dtw["bonus"]["f1"],
        "dtw_avg_path_score_bonus": dtw["bonus"]["avg_path_score"],
        "dtw_norm_score_bonus": dtw["bonus"]["norm_score"],
        "dtw_precision_gate": dtw["gate"]["precision"],
        "dtw_recall_gate": dtw["gate"]["recall"],
        "dtw_f1_gate": dtw["gate"]["f1"],
        "dtw_avg_path_score_gate": dtw["gate"]["avg_path_score"],
        "dtw_norm_score_gate": dtw["gate"]["norm_score"],
    }


def _zero_metrics() -> dict[str, float]:
    """Return a fully-zero ChainEval metric dict.

    Used when score_trace receives non-string input or when one side
    has no extractable steps. Keeps every output row schema-stable.
    """
    return {
        "step_recall": 0.0,
        "step_precision": 0.0,
        "chaineval_final_match": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "rougeLsum": 0.0,
        "bertscore": 0.0,
        "dtw_precision_bonus": 0.0,
        "dtw_recall_bonus": 0.0,
        "dtw_f1_bonus": 0.0,
        "dtw_avg_path_score_bonus": 0.0,
        "dtw_norm_score_bonus": 0.0,
        "dtw_precision_gate": 0.0,
        "dtw_recall_gate": 0.0,
        "dtw_f1_gate": 0.0,
        "dtw_avg_path_score_gate": 0.0,
        "dtw_norm_score_gate": 0.0,
    }


# Column names exposed to the eval CLI so the CSV writer can append exactly
# the ChainEval fields without re-listing them.
CHAINEVAL_DETAIL_COLUMNS: tuple[str, ...] = (
    "step_recall",
    "step_precision",
    "step_f1",
    "chaineval_final_match",
    "rouge2",
    "rougeL",
    "rougeLsum",
    "bertscore",
    "dtw_precision_bonus",
    "dtw_recall_bonus",
    "dtw_f1_bonus",
    "dtw_avg_path_score_bonus",
    "dtw_norm_score_bonus",
    "dtw_precision_gate",
    "dtw_recall_gate",
    "dtw_f1_gate",
    "dtw_avg_path_score_gate",
    "dtw_norm_score_gate",
)


def compute_step_f1(step_recall: float, step_precision: float) -> float:
    """F1 of the step-level recall and precision, with upstream's +1e-4 smoothing.

    Matches the formula specified in the task: ``2*r*p / (r + p + 1e-4)``.
    The smoothing avoids a division by zero when both recall and precision
    are zero.
    """
    return 2.0 * step_recall * step_precision / (step_recall + step_precision + 1e-4)
