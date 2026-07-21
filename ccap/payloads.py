"""Payload generation and bit-tuple helpers.

Payloads are random uniform k-bit strings. We sweep ``k`` per cell to find the
reliable-rate knee (§6, §7.2). Generation is deterministically seeded so a run is
fully reproducible from ``(cell, trial_index, master_seed)``.
"""

from __future__ import annotations

import numpy as np

from .types import Bits


def bits_to_array(bits: Bits) -> np.ndarray:
    """Tuple of {0,1} -> int8 ndarray."""
    return np.asarray(bits, dtype=np.int8)


def array_to_bits(arr: np.ndarray) -> Bits:
    """ndarray (any int/bool dtype) -> tuple of python ints in {0,1}."""
    return tuple(int(x) & 1 for x in np.asarray(arr).ravel())


def bits_to_str(bits: Bits) -> str:
    return "".join(str(int(b) & 1) for b in bits)


def str_to_bits(s: str) -> Bits:
    """Parse a compact 0/1 string, e.g. ``"0110"`` (no erasures)."""
    return tuple(int(c) for c in s.strip() if c in "01")


def csv_to_bits(s: object) -> Bits:
    """Parse a comma-separated int string, e.g. ``"0,-1,1"`` (erasures preserved)."""
    if not isinstance(s, str) or s.strip() == "":
        return ()
    return tuple(int(x) for x in s.split(",") if x.strip() not in ("", "nan"))


def hamming(a: Bits, b: Bits) -> int:
    """Hamming distance; positions present in only one operand count as errors."""
    n = max(len(a), len(b))
    dist = 0
    for i in range(n):
        ai = a[i] if i < len(a) else None
        bi = b[i] if i < len(b) else None
        if ai != bi:
            dist += 1
    return dist


def derive_seed(master_seed: int, *parts: object) -> int:
    """Deterministically derive a 63-bit child seed from a master seed and labels.

    Uses a stable hash so two processes / resumed runs reproduce identical payloads.
    """
    h = np.uint64(1469598103934665603)  # FNV-1a 64-bit offset basis
    prime = np.uint64(1099511628211)
    blob = "|".join([str(master_seed)] + [str(p) for p in parts]).encode("utf-8")
    with np.errstate(over="ignore"):
        for byte in blob:
            h = (h ^ np.uint64(byte)) * prime
    return int(h & np.uint64(0x7FFFFFFFFFFFFFFF))


class PayloadGenerator:
    """Deterministic uniform random k-bit payloads.

    Every call to :meth:`generate` is a pure function of the seed, so the same
    ``(master_seed, *labels, k, trial_index)`` always yields the same bits.
    """

    def __init__(self, master_seed: int = 0):
        self.master_seed = int(master_seed)

    def generate(self, k: int, *labels: object) -> Bits:
        """Draw a uniform random k-bit payload keyed by ``labels``."""
        if k < 0:
            raise ValueError("payload length k must be >= 0")
        seed = derive_seed(self.master_seed, *labels, "k", k)
        rng = np.random.default_rng(seed)
        return tuple(int(b) for b in rng.integers(0, 2, size=k, dtype=np.int8))
