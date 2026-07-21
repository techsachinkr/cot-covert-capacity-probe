"""The pre-registered experiment grid (§2-§3).

Confirmatory grid: Model-Condition (MC) x Mechanism x Task. The Mechanism factor
includes the benign control ``M0`` as a level, so the cell count is
``len(MC) x len(mechanisms incl. benign) x len(tasks)`` (273 in the frozen design).

Each non-benign cell-item is realized by ONE generation embedding ``slots`` (S=16)
candidate symbol-slots; payload size ``k`` in ``ladder`` is evaluated at analysis
time by truncating to the first k slots of the SAME generation (§2.4) — no extra
generations. Items (n=60) and payloads are drawn once from the master seed and
reused across every MC and mechanism (paired design, §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .payloads import derive_seed

BENIGN = "benign"  # the M0 control level of the Mechanism factor


@dataclass
class GridSpec:
    models: list[str]                  # Model-Condition (MC) names; mapped to backends
    mechanisms: list[str]              # encoding mechanisms; include BENIGN for the M0 control
    tasks: list[str]
    slots: int = 16                    # S: symbol-slots embedded per generation
    ladder: tuple[int, ...] = (1, 2, 4, 8, 16)   # payload sizes k (analysis-time truncation)
    n_items: int = 60                  # distinct task items per cell (paired pool)
    master_seed: int = 0
    run_sentinel: bool = True          # exploratory S6 (no extra LLM calls)
    run_paraphrase: bool = False       # exploratory; OFF in the confirmatory grid

    # ---- factory presets ---------------------------------------------------
    @classmethod
    def demo(cls) -> "GridSpec":
        """Tiny offline grid (mock backend) exercising the full pre-reg pipeline."""
        return cls(
            models=["MC-think", "MC-nothink", "MC-small", "MC-mid", "MC-large"],
            mechanisms=[BENIGN, "acrostic", "ordering", "framing", "abstraction",
                        "enumeration", "hedging"],
            tasks=["math"],
            slots=16,
            ladder=(1, 2, 4, 8, 16),
            n_items=150,   # demo uses a larger pool so the strict lower-CI knee separates
        )

    # ---- enumeration -------------------------------------------------------
    @property
    def encoding_mechanisms(self) -> list[str]:
        return [m for m in self.mechanisms if m != BENIGN]

    def n_cells(self) -> int:
        return len(self.models) * len(self.mechanisms) * len(self.tasks)

    def n_generations(self) -> int:
        """One generation per (MC, mechanism, task, item) — payload axis is truncation."""
        return self.n_cells() * self.n_items

    def base_cells(self):
        yield from product(self.models, self.mechanisms, self.tasks)

    def item_seed(self, task: str) -> int:
        return derive_seed(self.master_seed, "task-items", task)

    def payload_seed(self, task: str, item_index: int) -> int:
        # keyed by (task, item) only -> identical payload across every MC and mechanism
        return derive_seed(self.master_seed, "payload", task, item_index)

    def trial_seed(self, model: str, mechanism: str, task: str, item_index: int) -> int:
        return derive_seed(self.master_seed, model, mechanism, task, "i", item_index)
