#!/usr/bin/env python3
"""Score predictions against a reference label set (local read of the host formula).

    python score.py --pred out/dev_baseline --gold <thư mục nhãn>
    python score.py --pred out/dev_improved --gold <thư mục nhãn> -v

``--gold`` là thư mục chứa nhãn tham chiếu (các file ``{stem}.json``, có thể nằm
trong một subdir riêng). ``--pred`` là thư mục chứa các file prediction
``{stem}.json``. In điểm text / assertion / candidate / FINAL (công thức host:
final = 0.3*text + 0.3*assertions + 0.4*candidates, x100).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medextract import io_utils
from medextract.scoring.scorer import score_corpus_host


def _load_dir(d: Path) -> dict:
    """Load every {stem}.json in a dir into {stem: [concepts]}."""
    return {f.stem: io_utils.read_concepts(f) for f in sorted(d.glob("*.json"))}


def main(argv=None):
    p = argparse.ArgumentParser(description="Score predictions against a reference label set.")
    p.add_argument("--pred", required=True, help="dir of prediction {stem}.json")
    p.add_argument("--gold", required=True, help="thư mục chứa nhãn tham chiếu (file {stem}.json)")
    p.add_argument("-v", "--verbose", action="store_true", help="print per-record scores")
    args = p.parse_args(argv)

    gold_dir = Path(args.gold)
    gold_dir = gold_dir / "gold" if (gold_dir / "gold").is_dir() else gold_dir
    golds = _load_dir(gold_dir)
    preds = _load_dir(Path(args.pred))

    missing = sorted(set(golds) - set(preds))
    extra = sorted(set(preds) - set(golds))
    if missing:
        print(f"WARNING: {len(missing)} gold record(s) with no prediction: "
              f"{missing[:10]}", file=sys.stderr)
    if extra:
        print(f"WARNING: {len(extra)} prediction record(s) with no matching gold: "
              f"{extra[:10]}", file=sys.stderr)

    score = score_corpus_host(preds, golds)
    if args.verbose:
        for r in sorted(score.per_record, key=lambda r: r.stem):
            print(f"  {r.stem:>4}  text={r.text:.3f}  assert={r.assertions:.3f}"
                  f"  (n_pred={r.n_pred} n_gold={r.n_gold})")
    print(f"text       {score.text_score:.3f}")
    print(f"assertions {score.assertions_score:.3f}")
    print(f"candidates {score.candidates_score:.3f}")
    print(f"FINAL      {score.final_score * 100:.2f}")


if __name__ == "__main__":
    main()
