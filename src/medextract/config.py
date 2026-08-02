"""YAML config loading/merging.

Configs are plain dicts.  A solution's ``config.yaml`` may set ``extends`` to a
path (relative to itself) to inherit a base config; leaf keys override.
No magic numbers in code — model names, top_k, thresholds, device all live here.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: "os.PathLike | str") -> Dict[str, Any]:
    """Load a YAML config, resolving a single ``extends: <relative path>``."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config((path.parent / parent).resolve())
        cfg = _deep_merge(base, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Known config keys — the self-check flags any shipped key not in this schema as
# an orphan (a key no code reads). Keep in sync with the code's ``.get(...)`` sites.
# LEAVES are scalar/list paths; CONTAINERS are dict paths whose children are
# dynamic (type names, ontology names, channel names) and not individually listed.
# ---------------------------------------------------------------------------
KNOWN_CONFIG_LEAVES: set[str] = {
    "solution", "seed",
    "kb.sapbert_model", "kb.device",
    "ner.model", "ner.threshold", "ner.max_chunk_chars", "ner.max_chunk_tokens",
    "ner.raw_floor",
    "assertions.neg_window_chars", "assertions.block_lookback_lines",
    "normalization.top_k", "normalization.strip_drug_noise",
    "normalization.leaf_remap_enabled",
    "normalization.llm_rerank.retrieve_k",
    "llm.model", "llm.device", "llm.dtype", "llm.load_in_4bit",
    "llm.min_free_gb", "llm.enable_thinking",
    "models.max_total_parameters",
    "consensus_selector.enabled", "consensus_selector.primary_model",
    "consensus_selector.secondary_model", "consensus_selector.primary_device",
    "consensus_selector.secondary_device", "consensus_selector.batch_size",
    "consensus_selector.max_length", "consensus_selector.addition_margin_none",
    "consensus_selector.addition_types",
    "quantization.mode", "quantization.dtype",
    "quantization.compute_dtype", "quantization.double_quant",
    "boundary.generic_prefix",
    "pipeline.clean_spans", "pipeline.dedup_repeats", "pipeline.max_repeats",
}

KNOWN_CONFIG_CONTAINERS: set[str] = {
    "ner.label_map", "ner.thresholds",
    "normalization.max_candidates", "normalization.cutoffs",
    "normalization.filter_tty",
    "normalization.llm_rerank.max_candidates",
}


def orphan_config_keys(cfg: Dict[str, Any]) -> list[str]:
    """Return dotted paths of keys in ``cfg`` absent from the known schema.

    Container paths (dynamic-keyed dicts) are accepted wholesale; intermediate
    dicts are recursed so each scalar leaf is checked. ``extends`` is consumed by
    :func:`load_config` and ignored here.
    """
    problems: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if prefix == "" and k == "extends":
                continue
            path = f"{prefix}.{k}" if prefix else k
            if path in KNOWN_CONFIG_CONTAINERS:
                continue
            if isinstance(v, dict):
                walk(v, path)
            elif path not in KNOWN_CONFIG_LEAVES:
                problems.append(path)

    walk(cfg, "")
    return problems
