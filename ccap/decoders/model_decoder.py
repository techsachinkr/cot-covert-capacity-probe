"""Model decoder — an LLM that holds the codebook and recovers bits.

The canonical decoder for conceptual mechanisms (framing, abstraction) whose signal
is semantic rather than a fixed surface token. Capacity measured this way is
relative to the decoder model; we report the strongest decoder we built (§9).
"""

from __future__ import annotations

import re

from ..mechanisms.base import Mechanism
from ..models.base import ModelBackend
from ..prompts import DECODER_SYSTEM, build_decoder_prompt
from ..text_utils import ERASURE, strip_reasoning
from ..types import Bits, GenRequest
from .base import Decoder

_BITS_RE = re.compile(r"[01]")


def parse_bits(text: str, k: int) -> Bits:
    """Pull up to ``k`` bits from model output; pad missing positions with erasure.

    Prefers the longest contiguous run of 0/1 (the model's bit answer) but falls back
    to all 0/1 characters if no clean run of length k exists. The decoder may itself be
    a reasoning model that emits a ``<think>`` span before its answer; we strip that
    scratchpad first so its internal numerals (e.g. "step 0 ... 1") cannot pollute the
    bit read. (An unclosed ``<think>`` strips to "" -> all erasures, the correct outcome
    for a decode that ran out of budget mid-thought.)
    """
    text = strip_reasoning(text)
    if not text:
        return (ERASURE,) * k
    runs = re.findall(r"[01]{2,}", text)
    chosen = None
    for r in runs:
        if len(r) >= k:
            chosen = r[:k]
            break
    if chosen is None:
        allbits = _BITS_RE.findall(text)
        chosen = "".join(allbits[:k])
    out = [int(c) for c in chosen]
    out += [ERASURE] * (k - len(out))
    return tuple(out[:k])


class ModelDecoder(Decoder):
    name = "model"
    needs_backend = True

    # A reasoning decoder (e.g. Qwen3-32B) emits a <think> span BEFORE the bit answer,
    # so a tiny budget is exhausted mid-thought and the response carries no parseable
    # bits -> all erasures, which the reliability estimator scores as a transmission
    # FAILURE (deflating Ĉ_ctrl for the model-decoded mechanisms M2/M3/M4). The budget
    # must cover the decoder's own reasoning plus the bit string; 64 was far too small.
    def __init__(self, backend: ModelBackend, name: str = "model",
                 max_tokens: int = 4096, temperature: float = 0.0):
        self.backend = backend
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _request(self, cot: str, mechanism: Mechanism, k: int) -> GenRequest:
        return GenRequest(
            prompt=build_decoder_prompt(cot, mechanism, k),
            system=DECODER_SYSTEM,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def decode(self, cot: str, mechanism: Mechanism, k: int) -> Bits:
        out = self.backend.generate(self._request(cot, mechanism, k))
        return parse_bits(out.text, k)

    def decode_batch(self, cots: list[str], mechanism: Mechanism, k: int) -> list[Bits]:
        outs = self.backend.generate_batch([self._request(c, mechanism, k) for c in cots])
        return [parse_bits(o.text if o and o.error is None else "", k) for o in outs]
