"""No-codebook decoder for the sentinel's *recovered rate* R_sent (§7.5).

A sentinel that does not hold the codebook cannot recover the payload above chance.
We model this directly: it emits seeded pseudo-random bits, deterministic in the CoT
so the estimate is reproducible. Its mutual information with the intended payload is
~0, giving R_sent ~ 0 and a clean operational gap Delta_op = R_recv - R_sent.

(The sentinel's *detection* ability — flagging that some payload is present without
decoding it — is handled separately in :mod:`ccap.sentinel`, where we calibrate a
threshold and report FPR.)
"""

from __future__ import annotations

import numpy as np

from ..mechanisms.base import Mechanism
from ..payloads import derive_seed
from ..types import Bits
from .base import Decoder


class NoCodebookDecoder(Decoder):
    name = "sentinel"

    def decode(self, cot: str, mechanism: Mechanism, k: int) -> Bits:
        seed = derive_seed(0, "sentinel", len(cot), cot[:32])
        rng = np.random.default_rng(seed)
        return tuple(int(b) for b in rng.integers(0, 2, size=k))
