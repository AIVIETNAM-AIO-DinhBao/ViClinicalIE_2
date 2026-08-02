"""Local scorer for Task 2 — a reading of the host formula (NOT the official grader).

    final_score = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score

Aggregation is GLOBAL over all concepts across records (not a per-record mean):

  text  = Σ_matched (1 − WER) / (n_ref + 2·n_spurious)
  assn  = Σ_matched J_assert / (n_ref_assertable + 2·n_spurious_assertable)
  cand  = Σ_matched J_cand·w / (Σ_ref w + 2·Σ_spurious w),   w = n_codes + 1

Implementation choices the host left unspecified, all localized here:

  * Concept matching: a prediction matches a reference concept iff **same type**
    and **overlapping char span** (greedy by max overlap). Right-text/wrong-type
    is a brand-new concept that scores 0 everywhere (it cannot match its reference
    twin).
  * A spurious (unmatched) prediction is double-counted in every applicable
    denominator — the host penalty for spurious emission. Assertion labels on an
    already-spurious span add no further penalty; candidate codes do, because the
    candidate weight is ``n_codes + 1``.
  * WER is word-level edit distance over each matched concept's text.

Sanity guarantee: scoring the reference set against itself returns final_score == 1.0.
Reconcile with the organizers' official script when released.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import jiwer

    _HAS_JIWER = True
except Exception:  # pragma: no cover - jiwer optional at import
    _HAS_JIWER = False

from ..schema import ASSERTABLE_TYPES, CANDIDATE_TYPES

WEIGHTS = {"text": 0.3, "assertions": 0.3, "candidates": 0.4}


# ---- helpers ----------------------------------------------------------------
def _jaccard(pred: set, gold: set) -> float:
    """Jaccard with host edge cases: ∅,∅→1; ∅,X or X,∅→0; else |∩|/|∪|."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return len(pred & gold) / len(pred | gold)


def _span(c: dict) -> Tuple[int, int]:
    s, e = c["position"]
    return int(s), int(e)


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_concepts(
    pred: Sequence[dict], gold: Sequence[dict]
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Greedily match pred→reference by (same type, max char overlap).

    Returns ``(pred_idx -> ref_idx, unmatched_pred_idx, unmatched_ref_idx)``.
    """
    pairs = []  # (overlap, pred_i, ref_j)
    for pi, p in enumerate(pred):
        ps, pt = _span(p), p.get("type")
        for gj, g in enumerate(gold):
            if g.get("type") != pt:
                continue
            ov = _overlap(ps, _span(g))
            if ov > 0:
                pairs.append((ov, pi, gj))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    matched: Dict[int, int] = {}
    used_pred, used_gold = set(), set()
    for _, pi, gj in pairs:
        if pi in used_pred or gj in used_gold:
            continue
        matched[pi] = gj
        used_pred.add(pi)
        used_gold.add(gj)
    unmatched_pred = [i for i in range(len(pred)) if i not in used_pred]
    unmatched_gold = [j for j in range(len(gold)) if j not in used_gold]
    return matched, unmatched_pred, unmatched_gold


@dataclass
class RecordScore:
    stem: str
    text: float
    assertions: float
    n_pred: int
    n_gold: int


# ---- corpus aggregation -----------------------------------------------------
@dataclass
class Score:
    text_score: float
    assertions_score: float
    candidates_score: float
    final_score: float
    per_record: List[RecordScore] = field(default_factory=list)


def _wer_one(gold_text: str, pred_text: str) -> float:
    if not _HAS_JIWER:
        raise RuntimeError("jiwer required; pip install jiwer")
    ref = gold_text.strip()
    if not ref.split():
        return 0.0 if not pred_text.strip().split() else 1.0
    return jiwer.wer(ref, pred_text.strip())


def score_corpus_host(
    preds: Dict[str, list], golds: Dict[str, list], stems: Optional[Sequence[str]] = None
) -> Score:
    """Host-aligned scorer: per-concept WER/Jaccard with the spurious
    double-count penalty, aggregated globally over concepts.

    This is a local read of the official formula: a spurious (unmatched)
    prediction is penalised twice in each applicable denominator. Assertion
    labels emitted on that already-spurious span do not add another penalty;
    candidate codes do because the candidate weight is ``n_codes + 1``.
    Absolute numbers won't match the grader exactly, but the ranking of two
    runs does.

      text  = Σ_matched (1−WER) / (n_ref + 2·n_spurious)
      assn  = Σ_matched J_assert / (n_ref_assertable + 2·n_spurious_assertable)
      cand  = Σ_matched J_cand·w / (Σ_ref w + 2·Σ_spurious w),  w = n_codes+1
    """
    if stems is None:
        stems = sorted(golds, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))

    t_num = t_den = a_num = a_den = c_num = c_den = 0.0
    per_record: List[RecordScore] = []
    for s in stems:
        gold, pred = golds[s], preds.get(s, [])
        matched, up, ug = match_concepts(pred, gold)

        rt_num = 0.0
        for pi, gj in matched.items():
            rt_num += max(0.0, 1.0 - _wer_one(gold[gj]["text"], pred[pi]["text"]))
        rt_den = len(gold) + 2 * len(up)
        t_num += rt_num
        t_den += rt_den

        # assertions (assertable types)
        ga = [j for j in range(len(gold)) if gold[j].get("type") in ASSERTABLE_TYPES]
        spa = [i for i in up if pred[i].get("type") in ASSERTABLE_TYPES]
        ra_num = 0.0
        for pi, gj in matched.items():
            if gold[gj].get("type") in ASSERTABLE_TYPES:
                ra_num += _jaccard(set(pred[pi].get("assertions", []) or []),
                                   set(gold[gj].get("assertions", []) or []))
        a_num += ra_num
        a_den += len(ga) + 2 * len(spa)

        # candidates (candidate types), weighted by n_ref_codes+1
        for pi, gj in matched.items():
            if gold[gj].get("type") in CANDIDATE_TYPES:
                w = len(gold[gj].get("candidates", []) or []) + 1
                c_num += _jaccard(set(pred[pi].get("candidates", []) or []),
                                  set(gold[gj].get("candidates", []) or [])) * w
                c_den += w
        for gj in ug:
            if gold[gj].get("type") in CANDIDATE_TYPES:
                c_den += len(gold[gj].get("candidates", []) or []) + 1
        for pi in up:
            if pred[pi].get("type") in CANDIDATE_TYPES:
                c_den += 2 * (len(pred[pi].get("candidates", []) or []) + 1)

        per_record.append(RecordScore(
            stem=s, text=(rt_num / rt_den if rt_den else 1.0),
            assertions=(ra_num / (len(ga) + 2 * len(spa)) if (len(ga) + 2 * len(spa)) else 1.0),
            n_pred=len(pred), n_gold=len(gold)))

    text = t_num / t_den if t_den else 1.0
    assn = a_num / a_den if a_den else 1.0
    cand = c_num / c_den if c_den else 1.0
    final = WEIGHTS["text"] * text + WEIGHTS["assertions"] * assn + WEIGHTS["candidates"] * cand
    return Score(text, assn, cand, final, per_record)
