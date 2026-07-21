"""Model-backend abstraction.

Backends map a :class:`~ccap.types.GenRequest` to a :class:`~ccap.types.GenResult`.
The harness is backend-agnostic: the scientific core never imports torch/transformers.

CONTRACT: a real backend uses ONLY ``request.prompt`` (and ``system``). The
``request.sim`` field is an out-of-band channel for the *mock* simulator only;
real backends must ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..types import GenRequest, GenResult


class ModelBackend(ABC):
    name: str = "backend"
    #: True only for simulators; the runner uses this to decide whether to attach
    #: SimContext to requests.
    is_simulator: bool = False

    @abstractmethod
    def generate(self, request: GenRequest) -> GenResult:
        ...

    def generate_batch(self, requests: Iterable[GenRequest]) -> list[GenResult]:
        return [self.generate(r) for r in requests]

    def close(self) -> None:  # pragma: no cover - backends override if needed
        pass
