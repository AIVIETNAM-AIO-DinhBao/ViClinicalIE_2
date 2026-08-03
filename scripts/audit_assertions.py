#!/usr/bin/env python3
"""Audit non-empty assertions in a submission/output directory.

This is intentionally model-free and fast. Use it after a Phase 4 run to ensure
the assertion experiment remains conservative before submitting:

    python scripts/audit_assertions.py --pred out/improved_v2_assertion_neg \
        --input data/input --samples 8

It prints overall non-empty assertion rate, counts by label/type, and a few
context snippets for manual inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


for stream in (sys.stdout, sys.stderr):
    try:  # Windows consoles may default to cp1252 and fail on Vietnamese labels.
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audit assertion usage in output JSON files.")
    p.add_argument("--pred", required=True, help="directory containing <stem>.json outputs")
    p.add_argument("--input", default=None, help="optional directory containing matching <stem>.txt inputs")
    p.add_argument("--samples", type=int, default=10, help="number of non-empty examples to print")
    p.add_argument("--window", type=int, default=90, help="context chars around each asserted span")
    return p.parse_args(argv)


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data.get("concepts", data) if isinstance(data, dict) else data


def context_snippet(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:start] + "[[" + text[start:end] + "]]" + text[end:right]
    return " ".join(snippet.split())


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_dir = Path(args.pred)
    input_dir = Path(args.input) if args.input else None

    json_paths = sorted(p for p in pred_dir.glob("*.json") if p.name != "submission.zip")
    total = 0
    non_empty = 0
    by_label = Counter()
    by_type = Counter()
    by_label_type = Counter()
    examples = []

    for jp in json_paths:
        concepts = read_json(jp)
        text = ""
        if input_dir is not None:
            txt_path = input_dir / f"{jp.stem}.txt"
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8-sig")
        for c in concepts:
            total += 1
            labels = c.get("assertions") or []
            if not labels:
                continue
            non_empty += 1
            typ = c.get("type", "?")
            by_type[typ] += 1
            for label in labels:
                by_label[label] += 1
                by_label_type[(label, typ)] += 1
            if len(examples) < args.samples:
                pos = c.get("position") or [0, 0]
                snippet = context_snippet(text, int(pos[0]), int(pos[1]), args.window) if text else c.get("text", "")
                examples.append((jp.name, typ, labels, c.get("text", ""), snippet))

    rate = (100.0 * non_empty / total) if total else 0.0
    print(f"files={len(json_paths)} concepts={total} non_empty_assertions={non_empty} rate={rate:.2f}%")
    print("by_label:", dict(by_label))
    print("by_type:", dict(by_type))
    print("by_label_type:", {f"{k[0]}|{k[1]}": v for k, v in by_label_type.items()})
    if examples:
        print("\nexamples:")
        for name, typ, labels, mention, snippet in examples:
            print(f"- {name} {typ} {labels} mention={mention!r}")
            print(f"  {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())