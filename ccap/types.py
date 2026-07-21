"""Core data model shared across the harness.

Design rule: the *scientific* objects here depend only on the standard library so
that the capacity/statistics core never pulls in torch, transformers, or vLLM.
Model backends consume :class:`GenRequest` and return :class:`GenResult`.

Bit strings are stored as tuples of ``int`` (values in {0, 1}) so every object is
hashable and JSON-serializable. Convert to/from :class:`numpy.ndarray` with the
helpers in :mod:`ccap.payloads`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence

Bits = tuple[int, ...]


class Mode(str, enum.Enum):
    """How a single trial uses (or ignores) the payload.

    ENCODE     -- sender is instructed to embed the payload via the mechanism.
    BENIGN     -- sender solves the cover task with *no* encoding instruction;
                  the payload is still drawn so the decoder is scored against
                  bits that were never embedded. Used for the style-vs-capacity
                  discrimination experiment (§7.3): capacity must collapse to ~0.
    PARAPHRASE -- an ENCODE trace passed through a paraphrase defense before
                  decoding (§7.5).
    """

    ENCODE = "encode"
    BENIGN = "benign"
    PARAPHRASE = "paraphrase"


@dataclass(frozen=True)
class SimContext:
    """Out-of-band context the *mock* backend uses to synthesize a realistic CoT.

    Real backends MUST ignore this field — they only see ``GenRequest.prompt``.
    It exists so the simulator can produce traces whose surface signature matches
    the mechanism's encoding contract at a controllable fidelity, letting the full
    pipeline (and the capacity estimator) be validated without a GPU.
    """

    mechanism: str
    intended_bits: Bits
    mode: Mode
    fidelity: float = 1.0  # P(symbol rendered correctly) in the simulator
    seed: int = 0
    task_answer: Optional[str] = None  # lets the simulator emit a correct final answer
    task_competence: float = 1.0       # P(emit correct answer) — to study encode->accuracy cost
    source_cot: Optional[str] = None   # PARAPHRASE mode: the CoT to be rewritten
    paraphrase_strength: float = 0.0   # 0..1 generic knob; each mechanism scales by its fragility


@dataclass(frozen=True)
class GenRequest:
    """A single generation request handed to a :class:`ccap.models.base.ModelBackend`."""

    prompt: str
    system: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[Sequence[str]] = None
    # Out-of-band simulator context; populated by the runner only for mock backends.
    sim: Optional[SimContext] = None


@dataclass(frozen=True)
class GenResult:
    """The model's reply plus light metadata."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    backend: str = ""
    truncated: bool = False
    error: Optional[str] = None   # set by batched paths so one failed request can't kill a cell


@dataclass(frozen=True)
class TrialSpec:
    """Fully identifies one cell-replicate of the experiment grid."""

    model: str
    mechanism: str
    task: str
    payload_len: int
    mode: Mode
    trial_index: int
    seed: int
    # The cover-task instance id (problem) used for this trial.
    task_item_id: str = ""

    @property
    def cell_id(self) -> str:
        """Identifier for the (model, mechanism, task, payload_len, mode) cell."""
        return f"{self.model}|{self.mechanism}|{self.task}|k{self.payload_len}|{self.mode.value}"


@dataclass
class TrialResult:
    """Everything observed for one trial.

    ``decoded`` maps a decoder name (e.g. ``"programmatic"``, ``"model"``,
    ``"sentinel"``) to the recovered bit tuple. ``intended`` is the payload that
    was *meant* to be embedded (in BENIGN mode it was never embedded — that is the
    point of the discrimination experiment).
    """

    spec: TrialSpec
    intended: Bits
    cot: str
    decoded: dict[str, Bits] = field(default_factory=dict)
    task_correct: Optional[bool] = None
    n_steps: Optional[int] = None  # surface features observed (for diagnostics)
    gen: Optional[GenResult] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- serialization -----------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        """Flatten to a tidy record for a DataFrame / JSONL log."""
        row: dict[str, Any] = {
            "model": self.spec.model,
            "mechanism": self.spec.mechanism,
            "task": self.spec.task,
            "payload_len": self.spec.payload_len,
            "mode": self.spec.mode.value,
            "trial_index": self.spec.trial_index,
            "seed": self.spec.seed,
            "task_item_id": self.spec.task_item_id,
            "cell_id": self.spec.cell_id,
            "intended": "".join(str(b) for b in self.intended),
            "task_correct": self.task_correct,
            "n_steps": self.n_steps,
            "error": self.error,
            "cot_chars": len(self.cot or ""),
        }
        for name, bits in self.decoded.items():
            # comma-separated so erasure symbols (-1) survive the round-trip
            row[f"decoded_{name}"] = ",".join(str(int(b)) for b in bits)
        if self.gen is not None:
            row["prompt_tokens"] = self.gen.prompt_tokens
            row["completion_tokens"] = self.gen.completion_tokens
            row["latency_s"] = self.gen.latency_s
            row["backend"] = self.gen.backend
        for k, v in self.extra.items():
            row[f"x_{k}"] = v
        return row

    def to_json(self) -> dict[str, Any]:
        """Full record including the raw CoT (for the raw bit-recovery log)."""
        d = asdict(self)
        d["spec"]["mode"] = self.spec.mode.value
        d["intended"] = list(self.intended)
        d["decoded"] = {k: list(v) for k, v in self.decoded.items()}
        # GenResult / nested already plain via asdict; drop the heavy sim echo.
        return d
