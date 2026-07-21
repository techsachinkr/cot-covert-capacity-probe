"""Decoder interface.

A decoder defines the predictive family against which capacity is measured;
capacity is decoder-relative and we report it as such (§5.3). Decoders map a CoT to
``k`` bit estimates in {0, 1, -1}, where -1 is an erasure (unparseable/missing slot).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..mechanisms.base import Mechanism
from ..types import Bits


class Decoder(ABC):
    name: str = "decoder"
    needs_backend: bool = False

    @abstractmethod
    def decode(self, cot: str, mechanism: Mechanism, k: int) -> Bits:
        ...

    def decode_batch(self, cots: list[str], mechanism: Mechanism, k: int) -> list[Bits]:
        """Decode many CoTs. Default loops; the model decoder overrides with a
        concurrent batch so its API calls aren't serialized."""
        return [self.decode(c, mechanism, k) for c in cots]
