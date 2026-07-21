"""Mechanism registry — the candidate covert channels CCAP probes (Table 1).

Each mechanism is a deterministic encoding contract mapping payload bits to an
observable CoT feature. Add a new channel by subclassing
:class:`ccap.mechanisms.base.Mechanism` and registering it here.
"""

from __future__ import annotations

from .base import Mechanism
from .abstraction import AbstractionMechanism
from .acrostic import AcrosticMechanism
from .enumeration import EnumerationParityMechanism
from .framing import FramingMechanism
from .hedging import HedgingMechanism
from .ordering import StepOrderingMechanism
from .ordering_semantic import OrderingSemanticMechanism

_REGISTRY: dict[str, type[Mechanism]] = {
    m.name: m
    for m in (
        AcrosticMechanism,          # lexical baseline / upper anchor
        StepOrderingMechanism,      # M2: tagged ordering ([a]/[b])
        OrderingSemanticMechanism,  # tagless ordering alternative (model decoder, stealthy)
        FramingMechanism,           # conceptual
        AbstractionMechanism,       # conceptual
        EnumerationParityMechanism,
        HedgingMechanism,
    )
}

_INSTANCES: dict[str, Mechanism] = {name: cls() for name, cls in _REGISTRY.items()}


def get_mechanism(name: str) -> Mechanism:
    if name not in _INSTANCES:
        raise KeyError(f"unknown mechanism {name!r}; available: {list_mechanisms()}")
    return _INSTANCES[name]


def list_mechanisms() -> list[str]:
    return list(_REGISTRY.keys())


def all_mechanisms() -> list[Mechanism]:
    return list(_INSTANCES.values())


__all__ = [
    "Mechanism",
    "AcrosticMechanism",
    "StepOrderingMechanism",
    "OrderingSemanticMechanism",
    "FramingMechanism",
    "AbstractionMechanism",
    "EnumerationParityMechanism",
    "HedgingMechanism",
    "get_mechanism",
    "list_mechanisms",
    "all_mechanisms",
]
