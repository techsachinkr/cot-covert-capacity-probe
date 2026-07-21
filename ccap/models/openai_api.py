"""OpenAI-compatible HTTP backend (OpenRouter / Together / vLLM-serve / Ollama / ...).

Zero extra dependencies: uses ``urllib`` from the standard library. Point ``base_url``
at any server exposing ``/chat/completions``. The API key is read from ``api_key`` or
the environment variable named by ``api_key_env`` (default ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..types import GenRequest, GenResult
from .base import ModelBackend


class OpenAIBackend(ModelBackend):
    is_simulator = False

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        max_retries: int = 4,
        extra_body: dict | None = None,
        max_workers: int = 8,
        max_tokens: int | None = None,
    ):
        self.name = name or model_id
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max(1, int(max_workers))  # concurrency for generate_batch
        self.max_tokens = max_tokens   # per-model cap that overrides the request budget
        # merged into the request body; e.g. vLLM server thinking toggle:
        #   {"chat_template_kwargs": {"enable_thinking": False}}
        self.extra_body = extra_body or {}

    def generate(self, request: GenRequest) -> GenResult:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens or request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if self.extra_body:
            payload.update(self.extra_body)
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choice = body["choices"][0]
                usage = body.get("usage", {})
                completion_tokens = int(usage.get("completion_tokens", 0))
                effective_max = self.max_tokens or request.max_tokens
                # Some servers (esp. with a reasoning parser) report finish_reason
                # "stop" even when the budget was exhausted. Fall back to the token
                # count so a generation that consumed the whole budget is still flagged
                # truncated (the runner then excludes it from the denominator).
                truncated = (choice.get("finish_reason") == "length"
                             or (bool(effective_max) and completion_tokens >= effective_max))
                return GenResult(
                    text=choice.get("message", {}).get("content") or "",
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=completion_tokens,
                    latency_s=time.time() - t0,
                    backend=self.name,
                    truncated=bool(truncated),
                )
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as e:
                detail = e
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
                    except Exception:
                        detail = f"HTTP {e.code}"
                last_err = detail
                backoff = 2.0 ** attempt  # 1s, 2s, 4s, ... (deterministic)
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
        # Surface the failure so the runner records a real error instead of a silent
        # empty trial (e.g. a wrong/unavailable model slug -> every trial excluded).
        raise RuntimeError(f"{self.name}: request failed after {self.max_retries} attempts: {last_err}")

    def generate_batch(self, requests):
        """Concurrent batch: fire up to ``max_workers`` requests at once. A failed
        request is captured as a GenResult with ``error`` set (so one bad call doesn't
        abort the whole cell)."""
        reqs = list(requests)
        if not reqs:
            return []

        def safe(r):
            try:
                return self.generate(r)
            except Exception as e:  # capture per-request, keep the batch alive
                return GenResult(text="", backend=self.name, error=str(e))

        if self.max_workers <= 1 or len(reqs) == 1:
            return [safe(r) for r in reqs]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(safe, reqs))  # map preserves order
