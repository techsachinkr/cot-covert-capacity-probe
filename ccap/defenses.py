"""Paraphrase defense (§7.5).

An auditor's mitigation: pass the CoT through a paraphraser and re-measure capacity.
We quantify the defense as the **fraction of Ĉ_ctrl removed** plus the residual the
defense cannot touch. Lexical/format channels (high ``paraphrase_fragility``) are
largely flattened; conceptual channels survive — telling the auditor which mechanisms
a paraphraser must specifically target.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mechanisms.base import Mechanism
from .models.base import ModelBackend
from .prompts import PARAPHRASER_SYSTEM, build_paraphrase_prompt
from .types import Bits, GenRequest, Mode, SimContext


def paraphrase_cot(
    cot: str,
    backend: ModelBackend,
    mechanism: Mechanism,
    original_intended: Bits,
    strength: float = 1.0,
    seed: int = 0,
    max_tokens: int = 1024,
) -> str:
    """Return a paraphrased CoT. Real backends use the prompt; the mock uses ``sim``."""
    req = GenRequest(
        prompt=build_paraphrase_prompt(cot),
        system=PARAPHRASER_SYSTEM,
        max_tokens=max_tokens,
        temperature=0.7,
        sim=SimContext(
            mechanism=mechanism.name,
            intended_bits=original_intended,
            mode=Mode.PARAPHRASE,
            source_cot=cot,
            paraphrase_strength=strength,
            seed=seed,
        ),
    )
    return backend.generate(req).text


@dataclass
class DefenseReport:
    capacity_before: float
    capacity_after: float
    fraction_removed: float
    residual_bits: float


def summarize_defense(capacity_before: float, capacity_after: float) -> DefenseReport:
    frac = 0.0 if capacity_before <= 1e-9 else max(0.0, 1.0 - capacity_after / capacity_before)
    return DefenseReport(
        capacity_before=capacity_before,
        capacity_after=capacity_after,
        fraction_removed=frac,
        residual_bits=max(0.0, capacity_after),
    )
