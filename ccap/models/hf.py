"""HuggingFace transformers backend (local open-weight inference).

Lazy-imports torch/transformers so the core harness stays importable without them.
Install with ``pip install ccap[hf]``. Uses the tokenizer's chat template when
available so reasoning-tuned and instruct models are prompted correctly.
"""

from __future__ import annotations

import time

from ..types import GenRequest, GenResult
from .base import ModelBackend


class HFBackend(ModelBackend):
    is_simulator = False

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        device: str | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        max_model_len: int | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HFBackend requires torch + transformers: pip install 'ccap[hf]'"
            ) from e

        import torch
        self.name = name or model_id
        self.model_id = model_id
        self.max_model_len = max_model_len
        # e.g. {"enable_thinking": False} to toggle Qwen3 reasoning mode at inference
        self.chat_template_kwargs = chat_template_kwargs or {}
        resolved_dtype = {"auto": "auto", "float16": torch.float16,
                          "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, "auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
            device_map=device or "auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self._torch = torch

    def _format(self, request: GenRequest) -> str:
        msgs = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        msgs.append({"role": "user", "content": request.prompt})
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           **self.chat_template_kwargs)
        sys = (request.system + "\n\n") if request.system else ""
        return sys + request.prompt

    def generate(self, request: GenRequest) -> GenResult:
        torch = self._torch
        text = self._format(request)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=request.temperature > 0,
                temperature=max(request.temperature, 1e-5),
                top_p=request.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_new = int(gen_ids.shape[0])
        return GenResult(
            text=completion,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=n_new,
            latency_s=time.time() - t0,
            backend=self.name,
            truncated=n_new >= request.max_tokens,  # consumed the full budget w/o an EOS stop
        )
