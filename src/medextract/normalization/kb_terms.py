"""Lexical KB term table for ``improved_v2`` exact-alias linking.

Loads a multi-alias parquet (one row per ``(code, alias)``) and serves the
exact-alias lookup used by :class:`~medextract.normalization.lexical_lookup.ExactAliasLookup`:

* **exact_alias** — normalized (and diacritic-stripped) mention → exact codes.

Normalization here is for retrieval only; it never touches document text, so
offsets are unaffected (only a retrieval copy of the mention is normalised).

The v2 parquet is written by the enriched builders
(:mod:`medextract.kb.build_icd` / :mod:`medextract.kb.build_rxnorm`); baseline and
improved keep using their single-name parquets unchanged.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

log = logging.getLogger("medextract.kb_terms")

V2_DIR = Path("data/kb/processed")
V2_FILES = {"ICD10": "icd_terms_v2.parquet", "RXNORM": "rxnorm_terms_v2.parquet"}


def _norm(s: str) -> str:
    return " ".join((unicodedata.normalize("NFC", str(s)).lower()).split())


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


class KBTermTable:
    """Multi-alias term table backing the exact-alias lookup for one KB."""

    def __init__(self, kb: str, df: pd.DataFrame):
        self.kb = kb
        self.codes = df["code"].astype(str).tolist()
        # normalized + diacritic-stripped aliases for retrieval
        self.norm_names = [_norm(n) for n in df["name"].astype(str)]
        self.plain_names = [_strip_diacritics(n) for n in self.norm_names]
        # exact-alias index: normalized-name -> row idx, and code -> row idx
        self._exact: Dict[str, List[int]] = {}
        self._exact_plain: Dict[str, List[int]] = {}
        for i, (nm, pl) in enumerate(zip(self.norm_names, self.plain_names)):
            self._exact.setdefault(nm, []).append(i)
            self._exact_plain.setdefault(pl, []).append(i)
        self._code_rows: Dict[str, List[int]] = {}
        for i, c in enumerate(self.codes):
            self._code_rows.setdefault(c, []).append(i)
        log.info("[%s] %d aliases / %d codes loaded", kb, len(self.norm_names),
                 len(self._code_rows))

    def exact_alias(self, mention: str) -> List[Tuple[str, float]]:
        """Exact (normalized or diacritic-stripped) alias match → codes (score 1.0)."""
        m = _norm(mention)
        mp = _strip_diacritics(m)
        rows = self._exact.get(m) or self._exact_plain.get(mp) or []
        seen: List[str] = []
        for i in rows:
            if self.codes[i] not in seen:
                seen.append(self.codes[i])
        return [(c, 1.0) for c in seen]


def load(kb: str, path: Path = None) -> KBTermTable:
    path = path or (V2_DIR / V2_FILES[kb])
    df = pd.read_parquet(path)
    return KBTermTable(kb, df)
