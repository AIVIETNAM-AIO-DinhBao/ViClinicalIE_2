"""Generic / header-span rejection.

A span whose text *is* a bare field label or section heading ("thuốc",
"triệu chứng", "chẩn đoán", "kết quả xét nghiệm", "khám lâm sàng", "tiền sử" …)
is not a codeable concept and must be dropped. But the same words appearing
*inside* a longer valid mention ("kết quả xét nghiệm glucose cao" → the concept
is the whole phrase, or "glucose") must NOT be hard-dropped.

The rule is therefore context-aware, not a flat lowercase blacklist:

* reject only when the **normalized span text exactly equals** a known header
  label (a long mention can never equal a bare label), and
* reject when the span sits at a line start and is immediately followed by a
  label separator (``:`` / ``-``) and is itself a header word — the classic
  ``"Tiền sử:"`` heading case.

``document[start:end]`` is never mutated.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# bare field labels / section headings (normalized, no diacritics needed — we
# normalize both sides). Kept short on purpose: a *concept* almost always carries
# more than the label word.
HEADER_LABELS = {
    "thuốc", "thuoc", "ten thuoc", "tên thuốc",
    "trieu chung", "triệu chứng", "cac trieu chung", "các triệu chứng",
    "chan doan", "chẩn đoán", "chan doan so bo", "chẩn đoán sơ bộ", "icd",
    "xet nghiem", "xét nghiệm", "ten xet nghiem", "tên xét nghiệm",
    "ket qua", "kết quả", "ket qua xet nghiem", "kết quả xét nghiệm",
    "kham lam sang", "khám lâm sàng", "can lam sang", "cận lâm sàng",
    "tien su", "tiền sử", "tien su benh", "tiền sử bệnh", "benh su", "bệnh sử",
    "dieu tri", "điều trị", "tinh trang", "tình trạng", "dien bien", "diễn biến",
    "ly do", "lý do", "ly do vao vien", "lý do vào viện", "hoi benh", "hỏi bệnh",
    "benh nhan", "bệnh nhân", "chan doan ra vien", "chẩn đoán ra viện",
}

_LABEL_SEP = re.compile(r"^\s*[:\-–—]\s")


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s)).lower().split())


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def is_header_span(text: str, start: int, end: int,
                   section_metadata: Optional[dict] = None) -> bool:
    """True iff the span ``text[start:end]`` is a bare header/field label.

    ``section_metadata`` (optional) may carry ``line_start`` (offset of the
    containing line) and is used only to detect the ``"Label:"`` heading shape.
    Never inspects text outside ``[start, end]`` except the immediately following
    separator char.
    """
    span = text[start:end]
    norm = _norm(span)
    plain = _strip_diacritics(norm)
    # exact-equal to a known header label (diacritics on either side)
    if norm in HEADER_LABELS or plain in HEADER_LABELS:
        return True
    # heading shape: span at a line start, short, followed by ":" / "-" separator
    line_start = (section_metadata or {}).get("line_start")
    if line_start is not None and start == line_start and len(norm.split()) <= 4:
        if end < len(text) and _LABEL_SEP.match(text[end:end + 3]):
            # only flag if the span word itself is a header label (plain)
            return plain in HEADER_LABELS
    return False
