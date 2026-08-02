"""Exact-alias lookup for ``improved_v2`` candidate linking.

``improved_v2`` links a concept to an ontology code ONLY on a unique exact alias
match (precision-first: the host scorer double-penalises spurious codes, and only
a minority of diag/drug concepts carry a target code). The fuzzy channels used in
development (character n-gram, BM25, dense) changed no emitted candidate once
emission required a unique exact alias, so this module is a direct exact-alias
lookup over the v2 term tables — no fusion, no encoders, no FAISS.

Normalization here is for retrieval only; it never touches document text, so
offsets are unaffected (only a retrieval copy of the mention is normalised).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from ..schema import CANDIDATE_TYPES, Span

log = logging.getLogger("medextract.lexical_lookup")

# RXNORM mentions carry route/frequency/dose noise that is not part of the
# canonical drug name; strip it before the exact-alias lookup.
_DRUG_ROUTE_FREQ = re.compile(
    r"\b(po|iv|im|sc|sl|pr|tid|bid|qd|qid|qhs|qam|qpm|prn|q\d+h(:prn)?|daily|twice|once|"
    r"uống|tiêm|truyền|lần|ngày|viên|tab|caps?)\b",
    re.IGNORECASE,
)
_DECIMAL_COMMA = re.compile(r"(\d),(\d)")
_UNIT_SPACING = re.compile(r"\s*(mg|mcg|µg|ug|g|ml|iu|meq|mmol)\b", re.IGNORECASE)


class ExactAliasLookup:
    """Direct exact-alias lookup, one term table per candidate KB."""

    def __init__(self, term_tables: Dict[str, "object"]):
        self.term_tables = term_tables
        # deterministic per-(kb, mention) memo: clinical notes repeat the same
        # mention many times, so this turns an N-fold sweep into one lookup.
        self._cache: Dict[Tuple[str, str], Tuple] = {}

    @staticmethod
    def clean_mention(mention: str, kb: str) -> str:
        m = " ".join(mention.split())
        if kb == "RXNORM":
            m = _DRUG_ROUTE_FREQ.sub(" ", m)
            m = _DECIMAL_COMMA.sub(r"\1.\2", m)
            m = _UNIT_SPACING.sub(r" \1", m)
            m = " ".join(m.split())
        return m.strip()

    def retrieve(self, text: str, span: Span) -> Tuple[Optional[str], str, list]:
        """Return ``(kb, mention, exact_codes)``.

        ``exact_codes`` is the de-duplicated list of ontology codes whose
        normalized (or diacritic-stripped) alias equals the cleaned mention.
        The caller decides emission on ``len(exact_codes) == 1``.
        """
        start, end = span[0], span[1]
        kb = CANDIDATE_TYPES.get(span[2])
        if kb is None or kb not in self.term_tables:
            return None, "", []
        mention = self.clean_mention(text[start:end], kb)
        if not mention:
            return kb, "", []
        ckey = (kb, mention)
        hit = self._cache.get(ckey)
        if hit is not None:
            return hit
        codes = [c for c, _ in self.term_tables[kb].exact_alias(mention)]
        result = (kb, mention, codes)
        self._cache[ckey] = result
        return result


def from_config(cfg: dict) -> ExactAliasLookup:
    """Load the v2 term tables for both candidate KBs (fail fast if missing)."""
    from .kb_terms import load as load_terms

    term_tables = {kb: load_terms(kb) for kb in CANDIDATE_TYPES.values()}
    return ExactAliasLookup(term_tables=term_tables)
