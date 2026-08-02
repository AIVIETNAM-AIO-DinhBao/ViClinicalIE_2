"""Build the ICD-10 term table.

Source: the official Vietnamese ICD-10 catalog, TT06/2026/TT-BYT (Phụ lục Ban
danh mục mã ICD-10 tiếng Việt), shipped at ``data/kb/raw/`` as an ``*.xlsx`` with
columns Mã / Mô tả tiếng Việt / Mô tả tiếng Anh. Its Vietnamese names become
``name_vi`` and dominate retrieval of Vietnamese diagnosis mentions. TT06
supersedes the older QĐ 4469/QĐ-BYT catalog; both share this column layout.

Output: ``data/kb/processed/icd_terms.parquet`` with columns
``code, name_vi, name_en, source`` (one row per (code, name); synonyms exploded).
Always adds COVID codes U07.1 / U07.2 (QĐ 98).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/kb/raw")
OUT = Path("data/kb/processed/icd_terms.parquet")

COVID_CODES = [
    ("U07.1", "COVID-19, vi rút được xác định", "COVID-19, virus identified"),
    ("U07.2", "COVID-19, vi rút không được xác định", "COVID-19, virus not identified"),
]


def dot_code(code: str) -> str:
    """Normalize a dotless ICD-10 code to dotted form (A001 -> A00.1)."""
    code = str(code).strip().upper().replace(".", "")
    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"
    return code


def _find_vn_xlsx() -> Path | None:
    for pat in ("*.xlsx", "*.xls"):
        for p in sorted(RAW_DIR.glob(pat)):
            return p
    return None


def _load_vn(path: Path) -> pd.DataFrame:
    """Parse the TT06 Excel (.xls/.xlsx) into (code, name_vi, name_en, source).

    The sheet has a multi-column layout (chapter / group / disease). We target the
    **disease-level** columns specifically: MÃ BỆNH (dotted code), TÊN BỆNH
    (Vietnamese), DISEASE NAME (English) — not the chapter/group MÃ/TÊN columns.
    The header row isn't row 0, so we locate it by finding "MÃ BỆNH".
    """
    engine = "xlrd" if path.suffix.lower() == ".xls" else None
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str, engine=engine).fillna("")

    # locate the header row (the one containing a 'MÃ BỆNH' cell)
    header_row = None
    for i in range(min(15, len(raw))):
        cells = [str(x).strip().upper() for x in raw.iloc[i].tolist()]
        if any(c == "MÃ BỆNH" for c in cells):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find 'MÃ BỆNH' header in {path}")

    header = [str(x).strip() for x in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = header

    def col(name_upper: str):
        for c in header:
            if c.strip().upper() == name_upper:
                return c
        return None

    c_code = col("MÃ BỆNH")
    c_vi = col("TÊN BỆNH")
    c_en = col("DISEASE NAME")
    if c_code is None or c_vi is None:
        raise ValueError(f"Missing MÃ BỆNH / TÊN BỆNH columns in {path}: {header}")

    out = pd.DataFrame({
        "code": body[c_code].map(dot_code),
        "name_vi": body[c_vi].astype(str).str.strip(),
        "name_en": (body[c_en].astype(str).str.strip() if c_en else ""),
        "source": "TT06",
    })
    return out[out["code"].str.len() >= 3]


def build(out: Path = OUT) -> pd.DataFrame:
    vn = _find_vn_xlsx()
    if vn is None:
        raise FileNotFoundError(
            "No Vietnamese ICD-10 xlsx found under data/kb/raw/. The catalog "
            "(TT06/2026/TT-BYT) ships with the repo; see data/kb/raw/README.md.")
    print(f"[icd] using Vietnamese source: {vn}")
    df = _load_vn(vn)

    covid = pd.DataFrame(COVID_CODES, columns=["code", "name_vi", "name_en"])
    covid["source"] = "QD98-COVID"
    df = pd.concat([df, covid], ignore_index=True)

    # explode: keep a 'name' column preferring VI, and drop empty names
    df["name"] = df["name_vi"].where(df["name_vi"].str.len() > 0, df["name_en"])
    df = df[df["name"].str.len() > 0]
    df = df.drop_duplicates(subset=["code", "name"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tt06", action="store_true",
                    help="build improved_v2 parquet from the TT06-2026 xlsx (VN aliases only)")
    args = ap.parse_args(argv)
    if args.tt06:
        df = build_tt06(Path(args.out) if args.out != str(OUT) else OUT_V2)
        print(f"icd_terms_v2: {len(df):,} rows, {df['code'].nunique():,} codes")
        for code in ["I10", "E11.9", "J18.9", "K21.0", "A00", "U07.1"]:
            print(f"  {code}: {df.loc[df['code']==code, 'name'].tolist()[:3]}")
        return
    df = build(Path(args.out))
    print(f"icd_terms: {len(df):,} rows, {df['code'].nunique():,} codes -> {args.out}")
    print(df["source"].value_counts().to_string())
    for code in ["K21.0", "K21.9", "I10", "E11.9", "U07.1"]:
        names = df.loc[df["code"] == code, "name"].tolist()
        print(f"  {code}: {names[:1]}")


# ---------------------------------------------------------------------------
# improved_v2 multi-alias builder (TT06)
# ---------------------------------------------------------------------------

OUT_V2 = Path("data/kb/processed/icd_terms_v2.parquet")


def _clean_name(s: str) -> str:
    """Normalise punctuation/whitespace but preserve the original diacritics."""
    return " ".join(str(s).replace("・", " ").split()).strip(" ,;:")


# ---------------------------------------------------------------------------
# improved_v2 multi-alias builder (TT06).
#
# ``build_tt06`` keeps ONLY the Vietnamese surfaces named by the spec: TÊN BỆNH
# (leaf) → MÃ BỆNH, and TÊN NHÓM BỆNH 3 KÝ TỰ (3-char category) → MÃ NHÓM BỆNH
# 3 KÝ TỰ. English aliases are not kept (we link Vietnamese mentions), and
# explicit diacritic-stripped rows are redundant — KBTermTable builds an
# ``_exact_plain`` diacritic-stripped key for every alias at load time.
# Normalisation stays light (lowercase, collapse whitespace, strip end
# punctuation only); heavier alias normalisation (stripping internal punctuation,
# merging hyphens) measured worse, so it is not applied.
# ---------------------------------------------------------------------------
_TT06_CAT_CODE = "MÃ NHÓM BỆNH 3 KÝ TỰ"
_TT06_CAT_NAME = "TÊN NHÓM BỆNH 3 KÝ TỰ"
_TT06_LEAF_CODE = "MÃ BỆNH"
_TT06_LEAF_NAME = "TÊN BỆNH"


def _load_tt06(path: Path) -> pd.DataFrame:
    """Parse the TT06-2026 xlsx into long-form (code, name, alias_source) rows.

    Locates the header row by the ``MÃ BỆNH`` cell, then reads the four spec
    columns directly (robust to column reordering). The sheet carries both
    3-char category rows (MÃ BỆNH == MÃ NHÓM) and leaf rows; we emit a category
    alias from the category columns and a leaf alias from the leaf columns, so a
    short mention like ``bệnh tả`` hits category A00 while ``bệnh tả do ...``
    hits the leaf.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str).fillna("")
    header_row = None
    for i in range(min(15, len(raw))):
        if any(str(x).strip().upper() == "MÃ BỆNH" for x in raw.iloc[i]):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find 'MÃ BỆNH' header in {path}")
    header = [str(x).strip() for x in raw.iloc[header_row]]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = header

    def col(name_upper: str):
        for c in header:
            if c.strip().upper() == name_upper:
                return c
        return None

    c_cat_code, c_cat_name = col(_TT06_CAT_CODE), col(_TT06_CAT_NAME)
    c_leaf_code, c_leaf_name = col(_TT06_LEAF_CODE), col(_TT06_LEAF_NAME)
    if c_leaf_code is None or c_leaf_name is None:
        raise ValueError(f"Missing MÃ BỆNH / TÊN BỆNH in {path}")

    rows, seen = [], set()

    def add(code, name, src):
        if not name or str(name) == "nan":
            return
        code = dot_code(code)               # A001 -> A00.1; A00 stays A00
        if len(code.replace(".", "")) < 3:
            return
        name = _clean_name(name)
        if not name:
            return
        key = (code, name.lower())
        if key in seen:
            return
        seen.add(key)
        rows.append({"code": code, "name": name, "alias_source": src,
                     "tty": "", "is_synonym": False, "original_name": name})

    for _, r in body.iterrows():
        add(r[c_leaf_code], r[c_leaf_name], "TT06:LEAF")
        if c_cat_code is not None and c_cat_name is not None:
            add(r[c_cat_code], r[c_cat_name], "TT06:CAT3")

    # COVID (QĐ 98) — Vietnamese surface only
    for code, name_vi, _name_en in COVID_CODES:
        add(code, name_vi, "QD98-COVID:VI")
    return pd.DataFrame(rows)


def build_tt06(out: Path = OUT_V2) -> pd.DataFrame:
    """Build ``icd_terms_v2.parquet`` from the official TT06-2026 xlsx:
    Vietnamese leaf + 3-char-category aliases only (no English). See module note."""
    vn = _find_vn_xlsx()
    if vn is None:
        raise FileNotFoundError("TT06 xlsx not found under data/kb/raw/")
    df = _load_tt06(vn).drop_duplicates(subset=["code", "name"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"icd_terms_v2 (TT06): {len(df):,} aliases / {df['code'].nunique():,} codes -> {out}")
    return df


if __name__ == "__main__":
    main()
