"""Mock backend — a controllable *simulator* of a steganographic sender.

This is what makes the whole harness runnable and testable without a GPU. Given
the out-of-band :class:`~ccap.types.SimContext`, it synthesizes a CoT whose surface
signature matches the mechanism's encoding contract at a controllable fidelity:

* high fidelity  -> high recovered mutual information -> high estimated capacity;
* BENIGN mode    -> feature independent of the payload -> capacity collapses to ~0
                    (validates the style-vs-capacity discrimination of §7.3);
* PARAPHRASE mode-> source CoT decoded then re-rendered with mechanism-specific
                    fragility (validates the paraphrase-defense result of §7.5).

Fidelity may be per-mechanism (a ``dict``), which realistically lets one simulated
"model" be good at lexical encoding but poor at conceptual encoding — exactly the
gradient we expect across the real mechanism taxonomy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..mechanisms import get_mechanism
from ..payloads import bits_to_str, derive_seed
from ..types import GenRequest, GenResult, Mode
from .base import ModelBackend


class MockBackend(ModelBackend):
    is_simulator = True

    def __init__(
        self,
        name: str = "mock",
        fidelity: float | dict[str, float] = 0.9,
        task_competence: float = 0.95,
        seed: int = 0,
    ):
        """
        Args:
            fidelity: P(symbol rendered correctly) for ENCODE mode. A float applies
                to all mechanisms; a dict maps mechanism-name -> fidelity with an
                optional ``"default"`` key.
            task_competence: P(emit the correct final answer).
            seed: master seed for the simulator's per-trial RNG.
        """
        self.name = name
        self.fidelity_spec = fidelity
        self.task_competence = float(task_competence)
        self.seed = int(seed)

    def fidelity_for(self, mechanism: str) -> float:
        spec = self.fidelity_spec
        if isinstance(spec, dict):
            return float(spec.get(mechanism, spec.get("default", 0.9)))
        return float(spec)

    def generate(self, request: GenRequest) -> GenResult:
        sim = request.sim
        if sim is None:
            # Called without simulator context: emit an inert, decodable-as-erasure trace.
            return GenResult(text="Let me think.\nAnswer: 0", backend=self.name)

        mech = get_mechanism(sim.mechanism)
        rng = np.random.default_rng(
            derive_seed(self.seed, sim.seed, sim.mechanism, sim.mode.value,
                        bits_to_str(sim.intended_bits))
        )
        ctx = replace(
            sim,
            fidelity=self.fidelity_for(sim.mechanism),
            task_competence=self.task_competence,
        )
        cot = mech.simulate_cot(ctx, rng)
        return GenResult(
            text=cot,
            backend=self.name,
            completion_tokens=len(cot.split()),
            prompt_tokens=len(request.prompt.split()),
        )
