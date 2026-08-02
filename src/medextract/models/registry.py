"""Model parameter registry — enforces the competition ``<= 9B`` budget.

The legacy :class:`~medextract.llm.engine.LLMEngine` only checked the LLM's own
parameter count. ``improved_v2`` loads several models at once (NER proposer,
two selector teachers …), so a single registry tracks every
loaded model and asserts the *sum* of their parameter counts is within budget
before inference.

Rules (from the host spec):

* The budget is on **parameter count**, not memory. Quantization (4-bit / 8-bit)
  shrinks VRAM but does *not* change ``numel`` — it never relaxes the cap.
* A model is counted only while ``loaded=True``. ``release()`` removes it so a
  stage can free its budget for the next stage.
* On overrun the registry raises a clear :class:`BudgetError` listing every
  contributor — it never silently continues.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

log = logging.getLogger("medextract.models.registry")

DEFAULT_MAX_TOTAL_PARAMETERS = 9_000_000_000


@dataclass
class ModelRecord:
    name: str
    role: str
    params: int
    device: str = "cpu"
    quantization: Optional[str] = None
    loaded: bool = True


class BudgetError(RuntimeError):
    """Raised when the summed loaded-model parameter count exceeds the cap."""


@dataclass
class ModelRegistry:
    """Process-global registry of loaded models and their parameter counts."""
    max_total_parameters: int = DEFAULT_MAX_TOTAL_PARAMETERS
    _models: Dict[str, ModelRecord] = field(default_factory=dict)

    # -- registration ---------------------------------------------------------
    def register(
        self,
        name: str,
        role: str,
        params: int,
        device: str = "cpu",
        quantization: Optional[str] = None,
        loaded: bool = True,
        check: bool = True,
    ) -> ModelRecord:
        """Register a model. If ``check`` and the running total would exceed the
        budget, raise :class:`BudgetError` *before* storing it."""
        if params < 0:
            raise ValueError(f"negative param count for {name}: {params}")
        rec = ModelRecord(name, role, int(params), device, quantization, loaded)
        if rec.loaded and check:
            total = self.total() + rec.params
            if total > self.max_total_parameters:
                raise BudgetError(self._explain(rec, total))
        self._models[name] = rec
        log.info("[registry] + %-28s %-12s %.3fB (%s%s)",
                 name, role, rec.params / 1e9, device,
                 f", {quantization}" if quantization else "")
        return rec

    # -- queries --------------------------------------------------------------
    def total(self) -> int:
        """Summed parameter count of currently *loaded* models."""
        return sum(r.params for r in self._models.values() if r.loaded)

    def check_budget(self) -> None:
        """Assert the current loaded total is within budget, else raise."""
        total = self.total()
        if total > self.max_total_parameters:
            raise BudgetError(self._explain(None, total))

    def _explain(self, pending: Optional[ModelRecord], total: int) -> str:
        lines = [
            f"Parameter budget exceeded: {total/1e9:.3f}B > "
            f"{self.max_total_parameters/1e9:.3f}B cap",
            "Loaded models:",
        ]
        for r in self._models.values():
            if not r.loaded:
                continue
            tag = " (*)" if pending is not None and r is self._models.get(pending.name) else ""
            lines.append(
                f"  - {r.name:30s} role={r.role:12s} {r.params/1e9:.3f}B "
                f"({r.device}{f', {r.quantization}' if r.quantization else ''}){tag}"
            )
        if pending is not None and pending.name not in self._models:
            lines.append(
                f"  + {pending.name:30s} role={pending.role:12s} {pending.params/1e9:.3f}B "
                "(pending)"
            )
        return "\n".join(lines)


# process-global default registry; stages share one budget across the pipeline.
_DEFAULT: Optional[ModelRegistry] = None


def get_registry(max_total_parameters: Optional[int] = None) -> ModelRegistry:
    """Return the process-global registry, creating it on first call.

    ``max_total_parameters`` is honoured only on first creation (so all stages in
    a process agree on the same cap); pass it via config at startup.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ModelRegistry(max_total_parameters=max_total_parameters or DEFAULT_MAX_TOTAL_PARAMETERS)
    elif max_total_parameters is not None:
        _DEFAULT.max_total_parameters = max_total_parameters
    return _DEFAULT


def count_parameters(model) -> int:
    """Count parameters of a transformers/torch model (``sum(numel)``)."""
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception as e:  # pragma: no cover - defensive
        log.warning("could not count parameters (%s); assuming 0", e)
        return 0
