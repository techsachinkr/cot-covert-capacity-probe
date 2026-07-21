"""Model backends + a config-driven factory.

The mock backend is always importable; real backends (hf/vllm/openai) are imported
lazily by :func:`build_backend` so the core harness never requires torch/vllm.
"""

from __future__ import annotations

from typing import Any

from .base import ModelBackend
from .mock import MockBackend


def build_backend(spec: dict[str, Any]) -> ModelBackend:
    """Construct a backend from a config dict.

    ``spec`` must include ``type`` in {mock, hf, vllm, openai}; remaining keys are
    passed to the backend constructor.
    """
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "mock":
        return MockBackend(**spec)
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend(**spec)
    if kind == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(**spec)
    if kind == "openai":
        from .openai_api import OpenAIBackend
        return OpenAIBackend(**spec)
    raise ValueError(f"unknown backend type {kind!r}")


__all__ = ["ModelBackend", "MockBackend", "build_backend"]
