#!/usr/bin/env python
"""Probe a single OpenRouter model with one raw chat-completions call and print the
full response/error — the fast way to see WHY a listed model is failing.

    $env:OPENROUTER_API_KEY = "sk-or-..."          # PowerShell
    python scripts/probe_openrouter.py deepseek/deepseek-r1-distill-qwen-32b
    python scripts/probe_openrouter.py deepseek/deepseek-r1-distill-qwen-32b --max-tokens 512

Unlike the harness backend (which retries then raises), this prints the raw HTTP
status and body so OpenRouter's error message (no endpoints / data policy / credits /
bad param) is visible verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="OpenRouter model slug, e.g. deepseek/deepseek-r1-distill-qwen-32b")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--endpoints", action="store_true",
                    help="list the model's provider endpoints instead of calling it "
                         "(diagnoses '404 No endpoints found': if endpoints DO exist here "
                         "but the chat call says none, your account privacy/data-policy "
                         "setting is excluding them)")
    args = ap.parse_args()

    key = os.environ.get(args.api_key_env, "")
    if not key:
        sys.exit(f"Set {args.api_key_env} in this shell first.")

    if args.endpoints:
        url = f"{args.base_url.rstrip('/')}/models/{args.model}/endpoints"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode()).get("data", {})
            eps = data.get("endpoints", [])
            print(f"{args.model}: {len(eps)} endpoint(s)")
            for e in eps:
                print(f"  - provider={e.get('provider_name')!r} "
                      f"ctx={e.get('context_length')} "
                      f"data_policy={e.get('status')} {e.get('pricing', {})}")
            if not eps:
                print("  (no endpoints -> model effectively unavailable to anyone)")
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}")
        return

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    print(f"-> POST {args.model} (max_tokens={args.max_tokens})")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        print(f"HTTP {resp.status}")
        if "error" in body:
            print("ERROR IN BODY:", json.dumps(body["error"], indent=2))
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message", {})
        print("finish_reason:", ch.get("finish_reason"))
        print("content:", repr((msg.get("content") or "")[:200]))
        if msg.get("reasoning"):
            print("reasoning (first 200):", repr(msg["reasoning"][:200]))
        print("usage:", body.get("usage"))
        print("provider:", body.get("provider"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}")
        print("body:", e.read().decode("utf-8", "replace")[:1000])
    except Exception as e:  # noqa
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
