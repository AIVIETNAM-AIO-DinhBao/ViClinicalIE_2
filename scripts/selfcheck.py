#!/usr/bin/env python3
"""Fast self-check (<5s, no GPU, no KB).

Runs four checks, prints one PASS/FAIL line each, and exits 1 if any FAIL.

  CONFIG   the shipped YAMLs load and have no orphan config keys
  IMPORTS  every src/medextract module imports cleanly
  SCHEMA   validate_output accepts a valid concept and rejects three bad ones
  PATHS    every repo path named in README/INSTALL/docs/notebooks exists

Run after clone as the post-install sanity check:
    python scripts/selfcheck.py
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIGS = [
    "configs/baseline.yaml",
    "configs/improved.yaml",
    "configs/improved_v2.yaml",
    "configs/improved_v2_assertion_neg.yaml",
]


def check_config():
    from medextract.config import load_config, orphan_config_keys

    yamls = sorted(p.name for p in (ROOT / "configs").glob("*.yaml"))
    expected = [Path(rel).name for rel in CONFIGS]
    if yamls != expected:
        return False, f"expected configs {expected}, found {yamls}"
    orphans = []
    for rel in CONFIGS:
        try:
            cfg = load_config(ROOT / rel)
        except Exception as e:  # noqa: BLE001
            return False, f"{rel} failed to load: {e}"
        for k in orphan_config_keys(cfg):
            orphans.append(f"{rel}:{k}")
    if orphans:
        return False, f"orphan keys: {orphans[:5]}"
    return True, f"{len(CONFIGS)} configs load, no orphan keys"


def check_imports():
    pkg = ROOT / "src" / "medextract"
    mods = []
    for py in sorted(pkg.rglob("*.py")):
        rel = py.relative_to(pkg)
        if rel.parent == Path("."):
            dotted = "medextract." + rel.with_suffix("").name if rel.name != "__init__.py" else "medextract"
        else:
            if rel.name == "__init__.py":
                dotted = "medextract." + ".".join(rel.parts[:-1])
            else:
                dotted = "medextract." + ".".join(rel.with_suffix("").parts)
        mods.append(dotted)
    fails = []
    for m in dict.fromkeys(mods):  # dedupe, keep order
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{m}: {type(e).__name__}: {e}")
    if fails:
        return False, f"{len(fails)} import failure(s): {fails[:3]}"
    return True, f"{len(dict.fromkeys(mods))} modules import"


def check_schema():
    from medextract.schema import SchemaError, validate_output

    text = "bệnh nhân đau đầu, dùng metformin 500mg"
    needle = "đau đầu"
    s, e = text.index(needle), text.index(needle) + len(needle)
    try:
        out = validate_output(
            [{"text": needle, "position": [s, e], "type": "TRIỆU_CHỨNG"}], text)
        assert isinstance(out, list) and out and out[0]["text"] == needle
    except Exception as ex:  # noqa: BLE001
        return False, f"valid payload rejected: {ex}"
    bad = {
        "offset-shift": [{"text": needle, "position": [s + 1, e + 1], "type": "TRIỆU_CHỨNG"}],
        "unknown-type": [{"text": needle, "position": [s, e], "type": "BỆNH_LÝ"}],
        "candidates-on-non-candidate": [
            {"text": needle, "position": [s, e], "type": "TRIỆU_CHỨNG", "candidates": ["X"]}],
    }
    for name, payload in bad.items():
        try:
            validate_output(payload, text)
        except SchemaError:
            continue
        return False, f"invalid payload accepted: {name}"
    return True, "valid accepted; offset/type/candidates violations rejected"


def check_paths():
    """Every repo path named in README/INSTALL/docs/notebooks must exist on disk,
    except runtime-created / build-output / upload dirs."""
    ref = re.compile(r"(?:docs|configs|scripts|src|data|examples|notebooks|prompts)/[A-Za-z0-9_./-]+")
    _exempt = ("out/", "demo_input", "data/kb/processed", "data/kb/raw/",
               "data/input", "data/synthetic", "data/dev", "colab_t4.yaml",
               "docs/03_improved.md", "data/sample_input")
    doc_files = ["README.md", "INSTALL.md"]
    doc_files += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))
    doc_files += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "notebooks").glob("*.ipynb"))
    missing = []
    for df in doc_files:
        try:
            txt = (ROOT / df).read_text(encoding="utf-8")
        except OSError:
            continue
        for m in ref.finditer(txt):
            path = m.group(0).rstrip(".,;:)`]")
            if path.endswith("/"):
                path = path.rstrip("/")
            if any(ex in path for ex in _exempt):
                continue
            if not (ROOT / path).exists():
                missing.append(f"{df} -> {path}")
    if missing:
        return False, f"dead path(s): {missing[:5]}"
    return True, "doc/notebook paths resolve"


def main():
    checks = [("CONFIG", check_config), ("IMPORTS", check_imports),
              ("SCHEMA", check_schema), ("PATHS", check_paths)]
    all_ok = True
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})", flush=True)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
