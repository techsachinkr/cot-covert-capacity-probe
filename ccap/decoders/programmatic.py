"""Programmatic decoder — reads a mechanism's deterministic surface signature.

Gives a clean, reproducible reference rate for mechanisms with a surface contract
(acrostic, ordering, enumeration, hedging, and the keyword/digit proxies for the
conceptual mechanisms).
"""

from __future__ import annotations

from ..mechanisms.base import Mechanism
from ..types import Bits
from .base import Decoder


class ProgrammaticDecoder(Decoder):
    name = "programmatic"

    def decode(self, cot: str, mechanism: Mechanism, k: int) -> Bits:
        return mechanism.decode_programmatic(cot, k)
