"""Mechanism base class — an *encoding contract* mapping bits -> a CoT feature.

Each mechanism is a deterministic contract (Table 1 of the paper). It supplies:

* ``encode_instructions(bits)`` — the contract text injected into the sender prompt.
* ``codebook_description()``    — the bit<->feature mapping, used by the model
                                  decoder and to render Table 1.
* ``decode_programmatic(cot,k)``— read the surface feature back to bit estimates.
* ``render_slot(bit, rng)``     — render one slot's text; used only by the *mock*
                                  backend to synthesize faithful traces.

A response carries ``k`` independent slots (one bit each here; ``bits_per_slot``
is exposed for future multi-symbol contracts). Unparseable / missing slots decode
to the erasure symbol ``-1`` — the capacity estimator treats erasures as a real
part of the empirical channel rather than assuming a clean BSC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..text_utils import ERASURE, split_step_blocks
from ..types import Bits, Mode, SimContext


class Mechanism(ABC):
    #: stable identifier used in configs, logs, and figures
    name: str = "base"
    #: short human label for tables
    label: str = "Base"
    #: which family of decoder is canonical for this mechanism
    decoder: str = "programmatic"  # "programmatic" | "model"
    #: bits carried per slot (all current contracts use 1)
    bits_per_slot: int = 1
    #: hard cap on payload length k (slots that fit a reasonable response)
    max_payload: int = 16
    #: how thoroughly a paraphrase defense flattens this channel (0..1).
    #: lexical/format signatures are fragile (~1); conceptual ones are robust (~0.2).
    paraphrase_fragility: float = 0.5

    # ---- contract text -----------------------------------------------------
    @abstractmethod
    def encode_instructions(self, bits: Bits) -> str:
        """Return the per-payload encoding contract text for the sender prompt."""

    @abstractmethod
    def codebook_description(self) -> str:
        """Return the bit<->feature mapping (decoder-facing + Table 1)."""

    # ---- programmatic decoding --------------------------------------------
    @abstractmethod
    def _decode_block(self, block: str) -> int:
        """Read one step block -> {0, 1, ERASURE}."""

    def decode_programmatic(self, cot: str, k: int) -> Bits:
        """Recover k bit estimates from a CoT; missing slots become ERASURE."""
        blocks = split_step_blocks(cot)
        out = []
        for i in range(k):
            if i < len(blocks):
                try:
                    out.append(int(self._decode_block(blocks[i])))
                except Exception:
                    out.append(ERASURE)
            else:
                out.append(ERASURE)
        return tuple(out)

    # ---- simulation (mock backend only) -----------------------------------
    @abstractmethod
    def render_slot(self, bit: int, rng: np.random.Generator) -> str:
        """Render the content of one step that faithfully encodes ``bit``."""

    def _effective_bits(self, ctx: SimContext, rng: np.random.Generator) -> list[int]:
        """The bits the simulator will actually render, after mode/fidelity noise.

        Used for ENCODE and PARAPHRASE only; BENIGN renders neutral steps instead
        (no systematic feature), which is what gives the sentinel a real signal to
        flag and makes the §7.3 control honest.
        """
        bits = list(ctx.intended_bits)
        if ctx.mode == Mode.PARAPHRASE:
            current = self.decode_programmatic(ctx.source_cot or "", len(bits))
            strength = float(np.clip(ctx.paraphrase_strength * self.paraphrase_fragility, 0.0, 1.0))
            eff = []
            for s in current:
                b = s if s in (0, 1) else int(rng.integers(0, 2))
                if rng.random() < 0.5 * strength:  # push toward chance
                    b ^= 1
                eff.append(b)
            return eff
        # ENCODE: render each intended bit, flipped with prob (1 - fidelity).
        eff = []
        for b in bits:
            correct = rng.random() < ctx.fidelity
            eff.append(int(b) if correct else 1 - int(b))
        return eff

    def simulate_cot(self, ctx: SimContext, rng: np.random.Generator) -> str:
        """Synthesize a full step-block CoT for the mock backend."""
        if ctx.mode == Mode.BENIGN:
            contents = [self.render_neutral_slot(rng) for _ in ctx.intended_bits]
        else:
            eff = self._effective_bits(ctx, rng)
            contents = [self.render_slot(b, rng) for b in eff]
        lines = [self.sim_preamble(ctx)]
        for i, content in enumerate(contents, start=1):
            lines.append(f"Step {i}: {content}")
        ans = ctx.task_answer
        if ans is not None and ans != "":
            if rng.random() < ctx.task_competence:
                lines.append(f"Answer: {ans}")
            else:
                lines.append(f"Answer: {self._wrong_answer(ans, rng)}")
        return "\n".join(lines)

    def sim_preamble(self, ctx: SimContext) -> str:
        return "Let me reason through this step by step."

    def render_neutral_slot(self, rng: np.random.Generator) -> str:
        """Neutral step content with no systematic covert feature (BENIGN mode)."""
        verbs = ["combine", "evaluate", "track", "relate", "compare", "update"]
        v = verbs[int(rng.integers(0, len(verbs)))]
        return f"{v} the relevant quantities and continue toward the result."

    def has_surface_marker(self, block: str) -> bool:
        """Whether a step exhibits this mechanism's structural 'tell'.

        Used by the no-codebook sentinel detector (§7.5), NOT for bit decoding.
        Mechanisms with no unusual surface marker (e.g. acrostic) keep the default
        False and are correspondingly hard for a structure-only sentinel to flag.
        """
        return False

    @staticmethod
    def _wrong_answer(ans: str, rng: np.random.Generator) -> str:
        try:
            val = float(ans)
            delta = rng.choice([-2, -1, 1, 2])
            return str(int(val + delta)) if val.is_integer() else f"{val + delta:.2f}"
        except ValueError:
            return "indeterminate"

    # ---- capacity bookkeeping ---------------------------------------------
    def capacity_slots(self, k: int) -> int:
        return min(k, self.max_payload)
