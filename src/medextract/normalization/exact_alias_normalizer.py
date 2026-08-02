"""Exact-alias candidate linking for ``improved_v2`` (precision-first).

    mention → clean → exact-alias lookup
            → emit a code ONLY on a unique exact alias
            → ICD bare-category remapped to its .9 leaf when that leaf exists
            → at most one candidate

The host scorer (``score_corpus_host``) double-penalises spurious concepts and
wrong codes, and only a minority of diag/drug concepts carry a target code, so
linking is precision-first: a code is emitted only when the mention uniquely
matches one official alias. The Qwen3-8B listwise reranker was ablated and
dropped — it emitted a code for far too many concepts, collapsing candidate
Jaccard (see docs/04_findings.md).

Implements the ``Normalizer.predict(text, span) -> List[str]`` contract.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ..schema import CANDIDATE_TYPES, Span
from .base import Normalizer

log = logging.getLogger("medextract.normalization.exact_alias")

_ICD_CATEGORY = re.compile(r"^[A-Z]\d\d$")   # bare 3-char ICD category, e.g. E11


class ExactAliasNormalizer(Normalizer):
    """Precision-first lexical linker: one code only on a unique exact alias."""

    def __init__(self, retriever, max_candidates: Optional[Dict[str, int]] = None,
                 icd_codes: Optional[set] = None, leaf_remap_enabled: bool = True):
        self.retriever = retriever
        self.max_candidates = max_candidates or {"ICD10": 1, "RXNORM": 1}
        self.icd_codes = icd_codes or set()
        # leaf_remap_enabled: toggle the bare-category -> .9 leaf remap for ICD10.
        self.leaf_remap_enabled = leaf_remap_enabled

    def _leaf_ify(self, codes: List[str]) -> List[str]:
        """Remap a bare 3-char ICD category to its ``.9`` (unspecified) leaf when
        that leaf exists. Mã đích gần như luôn là mã lá, và một mention
        bare-category không mang complication/severity, nên .9 là quy ước chung —
        phép này chỉ đổi 0 thành 1, không bao giờ ngược lại."""
        return [f"{c}.9" if (_ICD_CATEGORY.match(c) and f"{c}.9" in self.icd_codes) else c
                for c in codes]

    def predict(self, text: str, span: Span) -> List[str]:
        if span[2] not in CANDIDATE_TYPES:
            return []
        kb, mention, codes = self.retriever.retrieve(text, span)
        if kb is None or not mention or len(codes) != 1:
            return []                       # precision-first: no unique exact alias -> no code
        code = codes[0]
        if kb == "ICD10" and self.leaf_remap_enabled:
            code = self._leaf_ify([code])[0]
        return [code][: self.max_candidates.get(kb, 1)]


def from_config(cfg: dict) -> ExactAliasNormalizer:
    from .lexical_lookup import from_config as build_retriever
    n = (cfg or {}).get("normalization", {}) or {}
    retriever = build_retriever(cfg)
    # ICD code set for the bare-category -> .9 leaf remap: reuse the codes already
    # loaded by the ICD10 term table (one source of truth, no second parquet read,
    # no silent fallback if the KB is missing — load_terms raises upstream).
    icd_codes = set(retriever.term_tables["ICD10"].codes)
    return ExactAliasNormalizer(
        retriever=retriever,
        max_candidates=n.get("max_candidates", {"ICD10": 1, "RXNORM": 1}),
        icd_codes=icd_codes,
        leaf_remap_enabled=bool(n.get("leaf_remap_enabled", True)),
    )
