"""Build the RxNorm term table from RXNCONSO (CSV or RRF).

Input:  data/kb/raw/rxnorm_rxnconso.csv   (RXNCONSO columns, comma-delimited)
        or a classic pipe-delimited RXNCONSO.RRF.
Output: data/kb/processed/rxnorm_terms.parquet  with columns rxcui, name, tty.

We keep English RXNORM-source atoms of term types IN / SCDC / SCD / SBD — the
ingredient / clinical-drug-component / clinical-drug / branded-drug granularities
that cover the drug mentions in the notes (mostly international ingredient names,
sometimes with strength).  Các rxcui phổ biến (amlodipine 308135, …) đều có tên
SCD sạch dưới sab=RXNORM nên bộ lọc giữ lại chúng.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RRF_COLUMNS = [
    "rxcui", "lat", "ts", "lui", "stt", "sui", "ispref", "rxaui", "saui",
    "scui", "sdui", "sab", "tty", "code", "str", "srl", "suppress", "cvf",
]

KEEP_TTY = {"IN", "SCDC", "SCD", "SBD"}
# synonym term-types added as extra surface forms for the kept rxcuis (more names
# per code → better mention→code retrieval recall). Does not add new codes.
SYN_TTY = {"SY", "PSN", "TMSY"}
KEEP_SAB = {"RXNORM"}

RAW_RRF = Path("data/kb/raw/RXNCONSO.RRF")          # official NLM/UTS release
RAW_CSV = Path("data/kb/raw/rxnorm_rxnconso.csv")   # Kaggle CSV fallback
OUT = Path("data/kb/processed/rxnorm_terms.parquet")


def _default_raw() -> Path:
    return RAW_RRF if RAW_RRF.exists() else RAW_CSV


def _read_raw(path: Path) -> pd.DataFrame:
    import csv as _csv

    if path.suffix.lower() == ".rrf":
        # pipe-delimited, no header, trailing pipe -> extra empty col.
        # QUOTE_NONE: RRF strings can contain quote chars that aren't delimiters.
        df = pd.read_csv(
            path, sep="|", header=None, names=RRF_COLUMNS + ["_"],
            dtype=str, keep_default_na=False, encoding="utf-8",
            quoting=_csv.QUOTE_NONE, engine="c", on_bad_lines="skip",
        )
        return df[RRF_COLUMNS]
    # CSV/TSV form (has header, possibly a UTF-8 BOM on the first column name).
    # The RxNorm full monthly release ships RXNCONSO as a comma file; the Current
    # Prescribable Content subset ships a tab file. Pick the delimiter by counting
    # separators in the header line so both parse without extra flags.
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    sep = "\t" if first.count("\t") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build(raw: Path = None, out: Path = OUT, synonyms: bool = False) -> pd.DataFrame:
    raw = raw or _default_raw()
    df = _read_raw(raw)
    df.columns = [c.strip().lower() for c in df.columns]
    # Column aliases / defaults so the lighter RxNorm Current Prescribable Content
    # layout also parses. The full RXNCONSO has every column, so this is a no-op
    # for the release used to produce the submitted results.
    if "str" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "str"})
    if "rxcui" not in df.columns and "rxnorm_cui" in df.columns:
        df = df.rename(columns={"rxnorm_cui": "rxcui"})
    for col, default in (("lat", "ENG"), ("sab", "RXNORM"), ("suppress", "")):
        if col not in df.columns:
            df[col] = default

    base = (
        (df["lat"] == "ENG")
        & df["sab"].isin(KEEP_SAB)
        & (df["suppress"].str.upper() != "Y")
    )
    # Keep obsolete-string ('O') atoms: một số mã vẫn cần giữ (ví dụ chlorpheniramine
    # 360047 mang suppress='O' trong bản này nhưng vẫn là một mã cần giữ).
    mask = base & df["tty"].isin(KEEP_TTY)
    kept = df.loc[mask, ["rxcui", "str", "tty"]].rename(columns={"str": "name"})
    if synonyms:
        # Add synonym surface forms (SY/PSN/TMSY) only for rxcuis already in our
        # code set — more names per code → better mention→code recall, no new
        # codes. RELABEL each synonym with its parent code's kept tty so the
        # query-time tty filter (e.g. keep only SCD) still surfaces it; otherwise
        # the retriever would drop SY rows and the expansion would be a no-op.
        rxcui2tty = dict(zip(kept["rxcui"], kept["tty"]))  # base tty per kept code
        syn_mask = base & df["tty"].isin(SYN_TTY) & df["rxcui"].isin(rxcui2tty)
        syn = df.loc[syn_mask, ["rxcui", "str"]].rename(columns={"str": "name"})
        syn["tty"] = syn["rxcui"].map(rxcui2tty)
        kept = pd.concat([kept, syn], ignore_index=True)
    kept["name"] = kept["name"].str.strip()
    kept = kept[kept["name"].str.len() > 0]
    kept = kept.drop_duplicates(subset=["rxcui", "name"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out, index=False)
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(_default_raw()))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--synonyms", action="store_true",
                    help="add SY/PSN/TMSY surface forms for kept codes (recall)")
    ap.add_argument("--v2", action="store_true",
                    help="build the improved_v2 multi-TTY parquet (rxnorm_terms_v2)")
    args = ap.parse_args(argv)
    if args.v2:
        df = build_v2(Path(args.raw) if args.raw else None, synonyms=True)
        print(f"rxnorm_terms_v2: {len(df):,} rows, {df['code'].nunique():,} rxcui")
        return
    df = build(Path(args.raw), Path(args.out), synonyms=args.synonyms)
    print(f"rxnorm_terms: {len(df):,} rows, {df['rxcui'].nunique():,} rxcui -> {args.out}")
    print(df["tty"].value_counts().to_string())
    # một vài rxcui phổ biến để kiểm tra nhanh bảng vừa build
    for code in ["308135", "243670", "866436", "392085", "313782", "904475", "197527"]:
        names = df.loc[df["rxcui"] == code, "name"].tolist()
        print(f"  {code}: {names[:2]}")


# ---------------------------------------------------------------------------
# improved_v2 multi-TTY builder
#
# The legacy ``build`` keeps only IN/SCDC/SCD/SBD and (optionally) relabels
# synonyms with the parent TTY so the SCD-only filter still surfaces them. That
# loses the original granularity. ``improved_v2`` keeps every clinically useful
# term type (IN/PIN/MIN/SCDC/SCD/SBD) with synonym expansion ON by default,
# preserves the original TTY/SAB and a synonym flag, and never hard-filters to a
# single TTY (TTY becomes a rerank *preference*, not a filter). Writes a richer
# ``rxnorm_terms_v2.parquet``; the baseline parquet is untouched.
# ---------------------------------------------------------------------------
OUT_V2 = Path("data/kb/processed/rxnorm_terms_v2.parquet")
KEEP_TTY_V2 = {"IN", "PIN", "MIN", "SCDC", "SCD", "SBD"}


def build_v2(raw: Path = None, out: Path = OUT_V2, synonyms: bool = True) -> pd.DataFrame:
    raw = raw or _default_raw()
    df = _read_raw(raw)
    df.columns = [c.strip().lower() for c in df.columns]
    # Column aliases / defaults so the lighter RxNorm Current Prescribable Content
    # layout also parses. The full RXNCONSO has every column, so this is a no-op
    # for the release used to produce the submitted results.
    if "str" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "str"})
    if "rxcui" not in df.columns and "rxnorm_cui" in df.columns:
        df = df.rename(columns={"rxnorm_cui": "rxcui"})
    for col, default in (("lat", "ENG"), ("sab", "RXNORM"), ("suppress", "")):
        if col not in df.columns:
            df[col] = default

    base = (
        (df["lat"] == "ENG")
        & df["sab"].isin(KEEP_SAB)
        & (df["suppress"].str.upper() != "Y")
    )
    kept = df.loc[base & df["tty"].isin(KEEP_TTY_V2),
                  ["rxcui", "str", "tty", "sab"]].rename(columns={"str": "name"})
    kept["alias_source"] = "RXNORM:" + kept["tty"]
    kept["is_synonym"] = False
    kept["original_name"] = kept["name"]

    if synonyms:
        rxcui2tty = dict(zip(kept["rxcui"], kept["tty"]))
        syn_mask = base & df["tty"].isin(SYN_TTY) & df["rxcui"].isin(rxcui2tty)
        syn = df.loc[syn_mask, ["rxcui", "str", "sab"]].rename(columns={"str": "name"})
        syn["tty"] = syn["rxcui"].map(rxcui2tty)
        syn["alias_source"] = "RXNORM:SYN"
        syn["is_synonym"] = True
        syn["original_name"] = syn["name"]
        kept = pd.concat([kept, syn], ignore_index=True)

    kept["name"] = kept["name"].str.strip()
    kept = kept[kept["name"].str.len() > 0]
    kept = kept.drop_duplicates(subset=["rxcui", "name", "tty"]).reset_index(drop=True)
    kept = kept.rename(columns={"rxcui": "code"})[["code", "name", "tty", "alias_source",
                                                   "is_synonym", "original_name"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out, index=False)
    print(f"rxnorm_terms_v2: {len(kept):,} aliases / {kept['code'].nunique():,} rxcui -> {out}")
    print(kept["tty"].value_counts().to_string())
    return kept


if __name__ == "__main__":
    main()
