#!/usr/bin/env python3
"""Run a medextract pipeline over an input dir -> per-file JSON (+ optional zip).

    python run.py --config configs/improved_v2.yaml --input <dir-of-txt> \\
                  --output out/submission --zip
"""
from __future__ import annotations

import argparse
import logging

from medextract import io_utils, set_seed
from medextract.config import load_config

CONFIGS = "configs/baseline.yaml | configs/improved.yaml | configs/improved_v2.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a medextract pipeline over an input dir.")
    p.add_argument("--config", required=True, help=f"config YAML ({CONFIGS})")
    p.add_argument("--input", required=True, help="dir of *.txt inputs")
    p.add_argument("--output", required=True, help="dir to write per-file JSON")
    p.add_argument("--zip", action="store_true", help="also build submission.zip")
    p.add_argument("--quantize", choices=["none", "8bit", "4bit"], default=None,
                   help="override the selector teacher quantization (improved_v2 only); "
                        "when omitted the config value is used as-is")
    p.add_argument("--offline", action="store_true",
                   help="force Hugging Face offline mode")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (default: config seed, else 42)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def resolve_seed(cli_seed, config):
    """Pick the random seed: CLI flag wins, else the config seed, else 42."""
    return cli_seed if cli_seed is not None else config.get("seed", 42)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    io_utils.configure_offline(args.offline)

    from medextract.pipeline import build_pipeline

    config = load_config(args.config)
    seed = resolve_seed(args.seed, config)
    config["seed"] = seed
    set_seed(seed)
    if args.quantize is not None:
        q = config.get("quantization") or {}
        q["mode"] = args.quantize
        config["quantization"] = q
    build_pipeline(config).run_dir(args.input, args.output, zip_it=args.zip)
    logging.getLogger("run").info("done: %s -> %s", args.config, args.output)


if __name__ == "__main__":
    main()
