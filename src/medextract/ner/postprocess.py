"""Span post-processing shared by the NER stage: clean up boundary errors.

- theo quy ước đề bài, bỏ từ phủ định đầu span ("Không sốt" → "sốt"); isNegated
  vẫn được phát hiện từ pre-context kề ngay sau đó.
- trailing punctuation ("… po daily." → "… po daily").
- spans that start/end mid-word ("-quang", "ồi máu não"): snap to word boundaries.
- literal section-header words extracted as concepts ("thuốc", "triệu chứng", …).
- drug spans absorbing Vietnamese route words ("… uống", "… tiêm dưới da").
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

from ..schema import Span

_WORD = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]")

_LEADING_NEG = re.compile(
    r"^(không có|không thấy|không ghi nhận|không còn|chưa ghi nhận|chưa có|"
    r"không bị|không|chưa|ko)\b[\s:]*",
    re.IGNORECASE,
)
_GENERIC_PREFIX = re.compile(
    r"^(?:dấu hiệu|biểu hiện|tình trạng|hội chứng)\b[\s:;,.-]*",
    re.IGNORECASE,
)
_WORD_TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
# Vietnamese route/administration tails to drop from drug spans
_DRUG_TAIL = re.compile(
    r"(\s+(uống|tiêm|tiêm dưới da|tiêm bắp|tiêm tĩnh mạch|truyền|ngậm|đặt|bôi|xịt|"
    r"nhỏ|hít|dán))+\s*$",
    re.IGNORECASE,
)
# trailing drug quantity ("x 1v", "x 02", "x1") — theo quy ước đề bài bỏ số lượng; giữ name+strength
_DRUG_QTY = re.compile(r"\s*[xX]\s*\d+(\s*v)?\s*$")
# literal header / stop words that are not real concepts
_JUNK = {
    "thuốc", "triệu chứng", "xét nghiệm", "chẩn đoán", "kết quả", "điều trị",
    "bệnh nhân", "cận lâm sàng", "khám lâm sàng", "tiền sử", "tiền sử bệnh",
    "bệnh sử", "lý do nhập viện", "kết quả xét nghiệm", "các triệu chứng",
    "tình trạng", "diễn biến", "thuốc điều trị",
}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower().strip()


def _snap_start(text: str, s: int) -> int:
    # if we start mid-word, move left to the word start
    while s > 0 and _WORD.match(text[s - 1]) and _WORD.match(text[s]):
        s -= 1
    return s


def _snap_end(text: str, e: int) -> int:
    while e < len(text) and _WORD.match(text[e - 1]) and _WORD.match(text[e]):
        e += 1
    return e


def clean_span(text: str, s: int, e: int, typ: str, drug_strip_qty: bool = True):
    """Return a cleaned (s, e) or None to drop the span.

    ``drug_strip_qty`` (default True) drops a trailing route word and a trailing
    "x N" quantity count from THUỐC spans. Quan sát trên dữ liệu mẫu cho thấy các
    nhãn tham chiếu thường GIỮ cụm dose/route, nên baseline an toàn đặt tham số
    này bằng False.
    """
    # snap to word boundaries first (fix broken subword spans)
    s = _snap_start(text, s)
    e = _snap_end(text, e)
    span = text[s:e]

    # strip a leading negation word (theo quy ước đề bài, bỏ từ phủ định)
    m = _LEADING_NEG.match(span)
    if m and m.end() < len(span):
        s += m.end()
        span = text[s:e]

    # drug: drop trailing Vietnamese route words
    if drug_strip_qty and typ == "THUỐC":
        m = _DRUG_TAIL.search(span)
        if m:
            e = s + m.start()
            span = text[s:e]

    # trim trailing punctuation / whitespace
    while e > s and (text[e - 1] in " .,;:\n\t-•*" or text[e - 1].isspace()):
        e -= 1
    # trim leading whitespace / punctuation
    while s < e and (text[s] in " .,;:\n\t-•*" or text[s].isspace()):
        s += 1

    # drug: drop a trailing quantity count ("x 1v", "x 02", "x1") AFTER punct trim;
    # theo quy ước đề bài chỉ giữ name+strength. Re-trim any leftover whitespace.
    if drug_strip_qty and typ == "THUỐC":
        q = _DRUG_QTY.search(text[s:e])
        if q and q.start() > 0:
            e = s + q.start()
            while e > s and text[e - 1].isspace():
                e -= 1

    span = text[s:e].strip()
    if len(span) < 2:
        return None
    if _norm(span) in _JUNK:
        return None
    # offset safety: the returned (s, e) must always slice a valid, non-empty
    # substring of text (text[s:e] == the cleaned span the writer emits).
    if not (0 <= s < e <= len(text)):
        return None
    return (s, e)


def clean_spans(text: str, spans: List[Span], dedup_repeats: bool = False,
                max_repeats: int = 1, drug_strip_qty: bool = True) -> List[Span]:
    """Clean spans; optionally collapse repeated identical (text, type) mentions.

    ``dedup_repeats`` keeps at most ``max_repeats`` occurrences of the same
    (normalized text, type) — clinical notes repeat the same symptom across
    sections, and the metric double-counts each extra copy as a spurious 0.

    .. note::
        This is the **baseline/improved** cleaner: it destructively strips the
        leading negation cue, snaps mid-word boundaries and hard-drops header
        words.
    """
    out = []
    seen = set()
    repeat_count: dict = {}
    for s, e, typ in spans:
        r = clean_span(text, s, e, typ, drug_strip_qty=drug_strip_qty)
        if r is None:
            continue
        key = (r[0], r[1], typ)
        if key in seen:
            continue
        seen.add(key)
        if dedup_repeats:
            rk = (_norm(text[r[0]:r[1]]), typ)
            repeat_count[rk] = repeat_count.get(rk, 0) + 1
            if repeat_count[rk] > max_repeats:
                continue
        out.append((r[0], r[1], typ))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def trim_generic_prefix(text: str, span: Span) -> Span:
    """Trim a generic lead only when a complete two-word concept remains.

    Chỉ cắt khi phần dư còn lại là một khái niệm đầy đủ (≥2 từ); phần dư một từ
    không ổn định nên giữ nguyên.
    """
    start, end, concept_type = span
    match = _GENERIC_PREFIX.match(text[start:end])
    if match is None:
        return span
    remainder = text[start + match.end() : end]
    if len(_WORD_TOKEN.findall(remainder)) < 2:
        return span
    return start + match.end(), end, concept_type
