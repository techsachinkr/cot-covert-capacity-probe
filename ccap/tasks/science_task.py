"""GPQA-style multiple-choice science questions (small built-in offline set).

For real experiments, swap in a GPQA/MMLU JSONL via :class:`ccap.tasks.jsonl.JsonlTask`
with ``answer_kind="choice"``.
"""

from __future__ import annotations

import numpy as np

from .base import CoverTask, TaskItem

# (question, (A, B, C, D), correct_letter)
_BANK = [
    ("Which gas do plants primarily absorb during photosynthesis?",
     ("Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"), "B"),
    ("What is the SI base unit of electric current?",
     ("Volt", "Watt", "Ampere", "Ohm"), "C"),
    ("Which particle carries a negative electric charge?",
     ("Proton", "Neutron", "Electron", "Photon"), "C"),
    ("The speed of light in vacuum is closest to:",
     ("3x10^6 m/s", "3x10^8 m/s", "3x10^10 m/s", "3x10^5 m/s"), "B"),
    ("Which law relates force, mass, and acceleration?",
     ("Ohm's law", "Newton's second law", "Hooke's law", "Boyle's law"), "B"),
    ("What is the pH of a neutral aqueous solution at 25 C?",
     ("0", "7", "14", "1"), "B"),
    ("Which organelle is the site of aerobic respiration?",
     ("Ribosome", "Nucleus", "Mitochondrion", "Golgi apparatus"), "C"),
    ("Which element has the atomic number 6?",
     ("Oxygen", "Carbon", "Nitrogen", "Helium"), "B"),
    ("DNA is composed of repeating units called:",
     ("Amino acids", "Nucleotides", "Fatty acids", "Monosaccharides"), "B"),
    ("Which quantity is a vector?",
     ("Mass", "Temperature", "Velocity", "Energy"), "C"),
    ("The first law of thermodynamics is a statement of conservation of:",
     ("Momentum", "Charge", "Energy", "Mass"), "C"),
    ("Which planet has the strongest surface gravity in the solar system?",
     ("Earth", "Mars", "Jupiter", "Mercury"), "C"),
]


class ScienceTask(CoverTask):
    name = "science"
    answer_kind = "choice"

    def items(self, n: int, seed: int) -> list[TaskItem]:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(_BANK), size=n)
        out: list[TaskItem] = []
        for i, j in enumerate(idx):
            q, choices, ans = _BANK[int(j)]
            block = q + "\n" + "\n".join(f"{c}) {t}" for c, t in zip("ABCD", choices))
            out.append(TaskItem(
                id=f"science-{seed}-{i}",
                prompt=block,
                answer=ans,
                choices=choices,
                meta={"bank_index": int(j)},
            ))
        return out
