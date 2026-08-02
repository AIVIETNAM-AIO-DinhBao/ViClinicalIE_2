"""Pipeline: compose NER + (assertions) + normalization into per-record concepts.

A single config-driven entry point. ``build_pipeline`` dispatches on the
``solution`` key in the config:

* ``baseline`` / ``improved`` — :class:`Pipeline`: GLiNER NER + ConText rule
  assertions + SapBERT retrieval (``improved`` adds an LLM rerank over retrieved
  candidates).
* ``improved_v2`` — :class:`PipelineV2`: GLiNER NER (per-type thresholds) + a
  two-teacher consensus selector (TRIỆU→CHẨN correction + non-candidate additions)
  + precision-first lexical linking + empty assertions.

The host scorer double-penalises spurious concepts, so ``improved_v2`` keeps
GLiNER's precision over higher recall and links candidates only on a unique exact
alias. Used by ``run.py`` (batch over an input dir, optional zip).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from . import io_utils
from .assertions.base import AssertionModel
from .ner.base import NERModel
from .normalization.base import Normalizer
from .schema import (
    ASSERTABLE_TYPES,
    CANDIDATE_TYPES,
    Concept,
    Span,
    validate_output,
)

log = logging.getLogger("medextract.pipeline")


def build_pipeline(config: dict, engine=None):
    """Build a pipeline from a config dict. Dispatches on ``solution``."""
    if (config or {}).get("solution") == "improved_v2":
        return _build_v2(config)
    return _build_legacy(config, engine=engine)


def _build_v2(config: dict):
    from .models.registry import get_registry
    from .ner.gliner_ner import from_config as build_ner
    from .normalization.exact_alias_normalizer import from_config as build_norm
    from .selector import from_config as build_selector

    cfg = config or {}
    models_cfg = cfg.get("models", {}) or {}
    get_registry(max_total_parameters=models_cfg.get("max_total_parameters", 9_000_000_000))

    bnd = cfg.get("boundary", {}) or {}
    return PipelineV2(
        ner=build_ner(cfg),            # GLiNER (per-type thresholds)
        normalizer=build_norm(cfg),    # precision-first lexical linking
        selector=build_selector(cfg),  # TRIỆU→CHẨN corrector + additions (None if disabled)
        name=cfg.get("solution", "improved_v2"),
        trim_generic=bool(bnd.get("generic_prefix", True)),
    )


def _build_legacy(config: dict, engine=None) -> "Pipeline":
    """Legacy baseline/improved build (GLiNER + ConText + SapBERT/LLM-rerank)."""
    from .ner.gliner_ner import from_config as build_ner
    from .assertions.context_rules import from_config as build_assertions

    norm_cfg = (config or {}).get("normalization", {}) or {}
    if norm_cfg.get("llm_rerank"):
        from .normalization.llm_reranker import from_config as build_norm
        if engine is None:
            from .llm.engine import LLMEngine
            lc = (config or {}).get("llm", {}) or {}
            engine = LLMEngine(
                model_name=lc.get("model", "Qwen/Qwen3-8B"),
                device=lc.get("device", "wait"),
                dtype=lc.get("dtype", "bfloat16"),
                min_free_gb=lc.get("min_free_gb", 18.0),
                enable_thinking=lc.get("enable_thinking", False),
                load_in_4bit=lc.get("load_in_4bit", False),
            )
        normalizer = build_norm(config, engine=engine)
    else:
        from .normalization.retriever import from_config as build_norm
        normalizer = build_norm(config)

    return Pipeline(
        ner=build_ner(config),
        assertion=build_assertions(config),
        normalizer=normalizer,
        name=(config or {}).get("solution", "medextract"),
        **pipeline_opts(config),
    )


def pipeline_opts(config: dict) -> dict:
    """Extract pipeline-level options (span cleanup / dedup) from config."""
    p = (config or {}).get("pipeline", {}) or {}
    return dict(
        clean_spans=p.get("clean_spans", True),
        dedup_repeats=p.get("dedup_repeats", False),
        max_repeats=p.get("max_repeats", 1),
    )


def _run_dir(pipe, in_dir, out_dir, zip_it: bool = True) -> None:
    """Run ``pipe.run_text`` over every ``*.txt`` in ``in_dir``; write per-stem
    JSON to ``out_dir``; optionally zip a flat ``submission.zip``.

    Shared by :class:`Pipeline` and :class:`PipelineV2` — identical loop, write,
    and logging, so both tiers produce the same on-disk layout.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = io_utils.list_inputs(in_dir)
    log.info("[%s] running over %d files -> %s", pipe.name, len(inputs), out_dir)
    for i, path in enumerate(inputs, 1):
        text = io_utils.read_text(path)
        io_utils.write_record(out_dir, path.stem, pipe.run_text(text), text)
        if i % 10 == 0 or i == len(inputs):
            log.info("[%s]  %d/%d", pipe.name, i, len(inputs))
    if zip_it:
        zp = io_utils.zip_submission(out_dir)
        log.info("[%s] wrote %s", pipe.name, zp)


class Pipeline:
    """baseline/improved: GLiNER + ConText assertions + retrieval/rerank."""

    def __init__(
        self,
        ner: Optional[NERModel] = None,
        assertion: Optional[AssertionModel] = None,
        normalizer: Optional[Normalizer] = None,
        name: str = "pipeline",
        clean_spans: bool = True,
        dedup_repeats: bool = False,
        max_repeats: int = 1,
    ):
        self.ner = ner
        self.assertion = assertion
        self.normalizer = normalizer
        self.name = name
        self.clean_spans = clean_spans
        self.dedup_repeats = dedup_repeats
        self.max_repeats = max_repeats

    def run_text(self, text: str) -> List[Concept]:
        if self.ner is None:
            return []
        spans: List[Span] = self.ner.predict(text)
        if self.clean_spans:
            from .ner.postprocess import clean_spans
            spans = clean_spans(text, spans, dedup_repeats=self.dedup_repeats,
                                max_repeats=self.max_repeats)

        if self.assertion is not None and spans:
            labels_per_span = self.assertion.predict(text, spans)
        else:
            labels_per_span = [[] for _ in spans]

        concepts: List[dict] = []
        for span, labels in zip(spans, labels_per_span):
            start, end, typ = span
            c: dict = {"text": text[start:end], "position": [start, end], "type": typ}
            c["assertions"] = list(labels) if typ in ASSERTABLE_TYPES else []
            if typ in CANDIDATE_TYPES and self.normalizer is not None:
                c["candidates"] = self.normalizer.predict(text, span)
            else:
                c["candidates"] = []
            concepts.append(c)

        return validate_output(concepts, text)

    def run_file(self, path) -> List[Concept]:
        return self.run_text(io_utils.read_text(path))

    def run_dir(self, in_dir, out_dir, zip_it: bool = True) -> None:
        _run_dir(self, in_dir, out_dir, zip_it)


class PipelineV2:
    """improved_v2: GLiNER (per-type thresholds) + consensus selector +
    precision-first lexical linking + empty assertions.

    Assertions are emitted empty: the dev-split assertions are sparse and the host scorer's
    spurious double-penalty makes any over-firing rule strictly worse.
    """

    def __init__(self, ner, normalizer: Optional[Normalizer], selector=None,
                 name: str = "improved_v2", trim_generic: bool = True):
        self.ner = ner
        self.normalizer = normalizer
        self.selector = selector
        self.name = name
        # generic-prefix trim is the one measured boundary lever kept (on by default).
        self.trim_generic = trim_generic

    def _build_concepts(self, text: str, spans) -> List[Concept]:
        from .ner.generic import is_header_span
        concepts: List[dict] = []
        for (s, e, typ) in spans:
            line_start = text.rfind("\n", 0, s) + 1
            if is_header_span(text, s, e, {"line_start": line_start}):
                continue
            c: dict = {"text": text[s:e], "position": [s, e], "type": typ,
                       "assertions": []}
            c["candidates"] = (self.normalizer.predict(text, (s, e, typ))
                                if typ in CANDIDATE_TYPES and self.normalizer else [])
            concepts.append(c)
        return validate_output(concepts, text)

    def run_text(self, text: str) -> List[Concept]:
        from .models.registry import get_registry
        from .ner.postprocess import clean_spans, trim_generic_prefix
        get_registry().check_budget()
        # capture raw GLiNER spans once (at raw_floor), then build the per-type-
        # threshold baseline; the selector reuses `raw` for additions.
        raw = self.ner.raw_scored_spans(text)
        thr = self.ner.thresholds or {}
        filt = [s for s in raw if s[2] not in thr or s[3] >= thr[s[2]]] if thr else list(raw)
        spans = clean_spans(text, self.ner.resolve_overlaps(filt))
        if self.selector is not None:               # TRIỆU→CHẨN corrector + additions
            spans = self.selector.select(text, spans, raw)
        if self.trim_generic:
            spans = [trim_generic_prefix(text, span) for span in spans]
        return self._build_concepts(text, spans)

    def run_file(self, path) -> List[Concept]:
        return self.run_text(io_utils.read_text(path))

    def run_dir(self, in_dir, out_dir, zip_it: bool = True) -> None:
        _run_dir(self, in_dir, out_dir, zip_it)
