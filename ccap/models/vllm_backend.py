"""vLLM backend for fast local batched inference.

Lazy-imports vllm. Install with ``pip install ccap[vllm]``. Implements true batched
generation, which the runner uses to amortize the inference-only grid efficiently.
"""

from __future__ import annotations

import time
from typing import Iterable

from ..types import GenRequest, GenResult
from .base import ModelBackend


class VLLMBackend(ModelBackend):
    is_simulator = False

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        dtype: str = "auto",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        chat_template_kwargs: dict | None = None,
        quantization: str | None = None,
        max_tokens: int | None = None,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:  # pragma: no cover
            raise ImportError("VLLMBackend requires vllm: pip install 'ccap[vllm]'") from e

        self.name = name or model_id
        self.model_id = model_id
        self.max_tokens = max_tokens   # per-model cap that overrides the global sender budget
        self._SamplingParams = SamplingParams
        # e.g. {"enable_thinking": False} to toggle Qwen3 reasoning mode at inference
        self.chat_template_kwargs = chat_template_kwargs or {}
        llm_kwargs = dict(
            model=model_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
        )
        if quantization:  # e.g. "awq_marlin", "gptq_marlin", "fp8" (else auto-detected from repo)
            llm_kwargs["quantization"] = quantization
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _format(self, request: GenRequest) -> str:
        msgs = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        msgs.append({"role": "user", "content": request.prompt})
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                      **self.chat_template_kwargs)
        sys = (request.system + "\n\n") if request.system else ""
        return sys + request.prompt

    def _params(self, request: GenRequest):
        return self._SamplingParams(
            max_tokens=self.max_tokens or request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=list(request.stop) if request.stop else None,
        )

    def generate(self, request: GenRequest) -> GenResult:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Iterable[GenRequest]) -> list[GenResult]:
        reqs = list(requests)
        prompts = [self._format(r) for r in reqs]
        params = [self._params(r) for r in reqs]
        t0 = time.time()
        outputs = self.llm.generate(prompts, params)
        dt = (time.time() - t0) / max(len(reqs), 1)
        results = []
        for out, req in zip(outputs, reqs):
            comp = out.outputs[0]
            n_comp = len(comp.token_ids)
            effective_max = self.max_tokens or req.max_tokens
            # vLLM sets finish_reason "length" on a budget hit; keep a token-count
            # fallback for parity with the other backends (the runner excludes truncated).
            truncated = (getattr(comp, "finish_reason", None) == "length"
                         or (bool(effective_max) and n_comp >= effective_max))
            results.append(GenResult(
                text=comp.text,
                prompt_tokens=len(out.prompt_token_ids),
                completion_tokens=n_comp,
                latency_s=dt,
                backend=self.name,
                truncated=bool(truncated),
            ))
        return results
