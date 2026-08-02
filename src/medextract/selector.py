"""Two-teacher consensus selector: TRIỆU→CHẨN corrector + non-candidate additions.

Uses primary Qwen3-4B-Instruct-2507 for TRIỆU_CHỨNG→CHẨN_ĐOÁN correction and a
primary + Qwen3.5-4B committee for precision-first non-candidate additions.

Two roles:
  1. CORRECTOR — for each baseline TRIỆU_CHỨNG span, run the primary teacher's
     five-way single-token logits and re-type it to CHẨN_ĐOÁN when it agrees.
  2. ADDITIONS — for each low-threshold raw GLiNER span NOT overlapping a baseline
     span, run six-way logits (0=NONE, 1-5) in BOTH teachers; ADD the span iff
     both predict the SAME type AND that type is non-candidate-bearing
     (TRIỆU_CHỨNG / TÊN_XÉT_NGHIỆM / KẾT_QUẢ_XÉT_NGHIỆM) AND both NONE-margins
     >= threshold. Candidate-bearing additions (CHẨN_ĐOÁN/THUỐC) are rejected:
     they inflate the candidate denominator and lower candidate-Jaccard.

Uses logits_to_keep=1 (one forward per batch, no generation). Total parameter
count including GLiNER is 8.517B (< 9B).
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from .llm.next_token import (
    batch_group_scores,
    chat_prompt,
    digit_ids,
    line_context,
    token_ids,
)

log = logging.getLogger("medextract.consensus_selector")

_TYPES = ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]
_SYS5 = (
    "Bạn là chuyên gia y lâm sàng Việt Nam. Phân loại ý niệm y khoa vào ĐÚNG MỘT loại:\n"
    "1 = TRIỆU_CHỨNG (triệu chứng: đau đầu, sốt, phù, mệt mỏi)\n"
    "2 = CHẨN_ĐOÁN (chẩn đoán bệnh: đái tháo đường, tăng huyết áp, viêm phổi, ung thư)\n"
    "3 = THUỐC (paracetamol, ceftriaxone, amlodipine)\n"
    "4 = TÊN_XÉT_NGHIỆM (công thức máu, MRI, nội soi, X-quang)\n"
    "5 = KẾT_QUẢ_XÉT_NGHIỆM (giá trị số: 120 mg/dL, HbA1c 7.2%, GCS 15)\n"
    "Quy tắc: 'tăng huyết áp/đái tháo đường/xơ gan' = 2; 'đau/phù/sốt/mệt' = 1."
)
_SYS6 = (
    "Bạn là chuyên gia y lâm sàng Việt Nam. Phân loại ý niệm y khoa:\n"
    "0 = KHÔNG PHẢI ý niệm y khoa hợp lệ (tiêu đề, từ chung chung, số/liều rời, hành chính)\n"
    "1 = TRIỆU_CHỨNG (triệu chứng: đau đầu, sốt, phù)\n"
    "2 = CHẨN_ĐOÁN (bệnh: đái tháo đường, tăng huyết áp, viêm phổi)\n"
    "3 = THUỐC (paracetamol, ceftriaxone)\n"
    "4 = TÊN_XÉT_NGHIỆM (công thức máu, MRI, nội soi)\n"
    "5 = KẾT_QUẢ_XÉT_NGHIỆM (giá trị số: 120 mg/dL, GCS 15)\n"
    "Quy tắc: 'tăng huyết áp/đái tháo đường'=2; 'đau/phù/sốt'=1; tiêu đề=0."
)


def _resolve_dtype(name: str):
    """Map a config dtype string to a torch dtype (default bfloat16)."""
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


class _Teacher:
    """One Qwen teacher: tokenizer + five-way and six-way single-token label ids."""

    def __init__(self, model_path: str, device: str, dtype: str = "bfloat16",
                 quantization: Optional[dict] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        torch_dtype = _resolve_dtype(dtype)
        qmode = (quantization or {}).get("mode", "none")
        if qmode == "none":
            self.model = (AutoModelForCausalLM
                          .from_pretrained(model_path, dtype=torch_dtype)
                          .to(device).eval())
        else:
            from transformers import BitsAndBytesConfig
            compute_dtype = _resolve_dtype(
                (quantization or {}).get("compute_dtype", dtype))
            if qmode == "4bit":
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=bool(
                        (quantization or {}).get("double_quant", True)),
                )
            elif qmode == "8bit":
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"unknown quantization mode: {qmode!r}")
            self.model = (AutoModelForCausalLM
                          .from_pretrained(model_path, quantization_config=bnb,
                                           device_map={"": device}).eval())
        self.d5 = digit_ids(self.tok, range(1, 6))   # five-way 1-5
        self.d6ent = digit_ids(self.tok, range(1, 6))  # six-way entity 1-5
        self.d6none = token_ids(self.tok, ("0", " 0"))
        self.nparams = sum(p.numel() for p in self.model.parameters())

    def _chat(self, sys, user):
        return chat_prompt(self, sys, user)


class ConsensusSelector:
    """Two-teacher consensus: TRIỆU→CHẨN corrector + non-candidate additions."""

    def __init__(self, primary_path: str, secondary_path: str, device: str = "auto",
                 primary_device: Optional[str] = None, secondary_device: Optional[str] = None,
                 batch_size: int = 48, max_length: int = 384,
                 addition_margin_none: float = 1.0,
                 addition_types: Optional[List[str]] = None,
                 dtype: str = "bfloat16", quantization: Optional[dict] = None):
        from .models.registry import get_registry
        from .utils.gpu import resolve_device
        # place the two teachers on DISTINCT GPUs to avoid intra-GPU contention
        # (both on one GPU made each forward ~80x slower in testing). The primary
        # device is resolved first; the secondary is resolved AFTER the primary
        # is placed so "auto" naturally picks a different GPU.
        self.device = resolve_device(primary_device or device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.addition_margin_none = addition_margin_none
        # non-candidate-bearing types only (CHẨN_ĐOÁN/THUỐC additions hurt cand-J)
        self.addition_types = set(addition_types or ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"])
        self.primary = _Teacher(primary_path, self.device, dtype=dtype, quantization=quantization)
        sec_dev = resolve_device(secondary_device or device)
        self.sec_device = sec_dev
        self.secondary = _Teacher(secondary_path, sec_dev, dtype=dtype, quantization=quantization)
        reg = get_registry()
        reg.register(os.path.basename(primary_path.rstrip("/")), "consensus_primary",
                     self.primary.nparams, device=self.device)
        reg.register(os.path.basename(secondary_path.rstrip("/")), "consensus_secondary",
                     self.secondary.nparams, device=sec_dev)
        log.info("[consensus] primary %s (%.3fB) on %s + secondary %s (%.3fB) on %s",
                 primary_path, self.primary.nparams / 1e9, self.device,
                 secondary_path, self.secondary.nparams / 1e9, sec_dev)

    # -- shared context -------------------------------------------------------
    @staticmethod
    def _context(text: str, s: int, e: int) -> str:
        return line_context(text, s, e)

    # -- batch scoring helpers ------------------------------------------------
    def _five_argmax(self, teacher, prompts):
        """Return list of argmax type strings (five-way)."""
        scores = batch_group_scores(
            teacher, prompts, teacher.d5, self.batch_size, self.max_length
        )
        return [_TYPES[max(range(5), key=lambda index: row[index])] for row in scores]

    def _six(self, teacher, prompts):
        """Return list of (pred_type, m_none) for six-way."""
        scores = batch_group_scores(
            teacher,
            prompts,
            list(teacher.d6ent) + [teacher.d6none],
            self.batch_size,
            self.max_length,
        )
        result = []
        for row in scores:
            entity, none = row[:5], row[5]
            top = max(range(5), key=lambda index: entity[index])
            result.append((_TYPES[top], entity[top] - none))
        return result

    # -- API ------------------------------------------------------------------
    def select(self, text: str, baseline: List[Tuple[int, int, str]],
               raw: List[Tuple[int, int, str, float]]) -> List[Tuple[int, int, str]]:
        """Apply consensus TRIỆU→CHẨN corrector + non-candidate additions.
        baseline: resolved per-type-threshold spans. raw: low-floor GLiNER spans."""
        out = list(baseline)
        # --- 1. CORRECTOR: re-type TRIỆU spans the primary teacher reads as CHẨN ---
        sym = [i for i, sp in enumerate(baseline) if sp[2] == "TRIỆU_CHỨNG"]
        if sym:
            prompts = [self.primary._chat(
                _SYS5, f"Ngữ cảnh: {self._context(text, baseline[i][0], baseline[i][1])}\n"
                f"Ý niệm: \"{text[baseline[i][0]:baseline[i][1]]}\"\nTrả lời CHỈ MỘT chữ số 1-5.")
                for i in sym]
            order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
            prompts = [prompts[k] for k in order]
            p_pred = self._five_argmax(self.primary, prompts)
            for k, orig_i in enumerate(order):
                if p_pred[k] == "CHẨN_ĐOÁN":
                    bi = sym[orig_i]            # baseline index (orig_i is a sym-list position)
                    s, e, _ = baseline[bi]
                    out[bi] = (s, e, "CHẨN_ĐOÁN")
        # --- 2. ADDITIONS: raw spans not overlapping baseline, both teachers agree ---
        bsp = [(s[0], s[1]) for s in baseline]

        def overlaps(a, b):
            return any(not (b <= x[0] or a >= x[1]) for x in bsp)
        cands = []
        seen = set()
        for s, e, _t, _sc in raw:
            if (s, e) in seen or overlaps(s, e):
                continue
            frag = text[s:e].strip()
            if len(frag) < 2:
                continue
            seen.add((s, e))
            cands.append((s, e))
        if cands:
            prompts = [self.primary._chat(
                _SYS6, f"Ngữ cảnh: {self._context(text, s, e)}\nÝ niệm: \"{text[s:e]}\"\n"
                f"Trả lời CHỈ MỘT chữ số 0-5.") for (s, e) in cands]
            # teachers on the SAME GPU run sequentially (concurrent threads contend
            # for one device); on DISTINCT GPUs they run concurrently (GIL releases
            # during CUDA kernels), roughly halving selector time. Output is identical.
            if self.device == self.sec_device:
                p6 = self._six(self.primary, prompts)
                s6 = self._six(self.secondary, prompts)
            else:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(2) as ex:
                    fp = ex.submit(self._six, self.primary, prompts)
                    fs = ex.submit(self._six, self.secondary, prompts)
                    p6, s6 = fp.result(), fs.result()
            for k, (s, e) in enumerate(cands):
                pt, pn = p6[k]
                if (pt == s6[k][0] and pt in self.addition_types
                        and pn >= self.addition_margin_none and s6[k][1] >= self.addition_margin_none):
                    out.append((s, e, pt))
        out.sort(key=lambda sp: (sp[0], sp[1]))
        return out


def from_config(cfg: dict) -> Optional[ConsensusSelector]:
    """Build the two-teacher selector from `consensus_selector:` config, or None."""
    c = (cfg or {}).get("consensus_selector", {}) or {}
    if not c.get("enabled", False):
        return None
    q = (cfg or {}).get("quantization") or {}
    return ConsensusSelector(
        primary_path=c["primary_model"],
        secondary_path=c["secondary_model"],
        device=(cfg or {}).get("kb", {}).get("device", "auto"),
        primary_device=c.get("primary_device"),
        secondary_device=c.get("secondary_device"),
        batch_size=int(c.get("batch_size", 48)),
        max_length=int(c.get("max_length", 384)),
        addition_margin_none=float(c.get("addition_margin_none", 1.0)),
        addition_types=c.get("addition_types"),
        dtype=q.get("dtype", "bfloat16"),
        quantization=q,
    )
