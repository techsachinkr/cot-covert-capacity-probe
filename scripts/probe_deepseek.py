#!/usr/bin/env python
"""Probe the DeepSeek *direct* API (api.deepseek.com) with one raw chat-completions
call per thinking mode -- the fast way to confirm, BEFORE the T0 lock, that:

  (1) the `deepseek-v4-flash` slug is live (not the legacy deepseek-chat/-reasoner
      ids, which retire 2026-07-24);
  (2) the top-level `thinking: {type: enabled|disabled}` toggle actually flips
      reasoning on/off (enabled -> `reasoning_content` present; disabled -> absent);
  (3) token accounting: whether `usage.completion_tokens` INCLUDES the reasoning
      tokens. The harness truncation guard (ccap/models/openai_api.py) flags a
      generation truncated when completion_tokens >= max_tokens, so reasoning MUST
      be counted or a budget-exhausted think generation slips through unflagged
      (prereg sec.8 residual-truncation exclusion).

    $env:DEEPSEEK_API_KEY = "sk-..."                 # PowerShell
    python scripts/probe_deepseek.py                 # runs BOTH modes, prints a diff
    python scripts/probe_deepseek.py --thinking enabled
    python scripts/probe_deepseek.py --max-tokens 16384   # mirror the think-arm budget

Unlike the harness backend (which retries then raises), this prints the raw HTTP
status and body so DeepSeek's error message is visible verbatim.

NOTE (prereg sec.4 deviation): DeepSeek thinking mode IGNORES temperature/top_p
(accepted, no effect). This probe still sends the frozen 0.7/0.95 -- mirroring the
exact harness request -- so the call is representative; it cannot *prove* they were
ignored from a single stochastic call (that rests on the DeepSeek docs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _call(base_url, key, model, thinking, max_tokens, temperature, top_p, timeout):
    """One chat-completions POST. Returns the parsed body dict; raises on HTTP error."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": ("A train travels 60 km in 1.5 hours, then 90 km in 2 hours. "
                        "What is its average speed over the whole trip? Show your reasoning."),
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,   # ignored in thinking mode (see module docstring)
        "top_p": top_p,
    }
    if thinking in ("enabled", "disabled"):
        payload["thinking"] = {"type": thinking}   # top-level field, per DeepSeek API
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _summarize(body, max_tokens):
    """Pull the fields we care about out of a successful response body."""
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message", {}) or {}
    # DeepSeek direct API returns the CoT in `reasoning_content`; OpenRouter-style
    # proxies use `reasoning`. Accept either so the probe also works through a proxy.
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    usage = body.get("usage", {}) or {}
    ctd = usage.get("completion_tokens_details") or {}
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "finish_reason": ch.get("finish_reason"),
        "reasoning_present": bool(reasoning),
        "reasoning_chars": len(reasoning),
        "content_preview": content[:200],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": ctd.get("reasoning_tokens"),
        "usage": usage,
        # mirrors the truncation guard in ccap/models/openai_api.py
        "harness_would_flag_truncated": (
            ch.get("finish_reason") == "length" or completion_tokens >= max_tokens
        ),
    }


def _print_one(label, s):
    print(f"\n=== thinking={label} ===")
    print(f"  finish_reason     : {s['finish_reason']}")
    print(f"  reasoning_present : {s['reasoning_present']}  ({s['reasoning_chars']} chars)")
    print(f"  prompt_tokens     : {s['prompt_tokens']}")
    print(f"  completion_tokens : {s['completion_tokens']}")
    print(f"  reasoning_tokens  : {s['reasoning_tokens']}  (usage.completion_tokens_details)")
    print(f"  harness->truncated: {s['harness_would_flag_truncated']}")
    print(f"  content[:200]     : {s['content_preview']!r}")
    print(f"  raw usage         : {json.dumps(s['usage'])}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--thinking", choices=["enabled", "disabled", "both"], default="both",
                    help="which mode(s) to probe; 'both' (default) runs each and prints a verdict")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)   # frozen §4 (ignored in think mode)
    ap.add_argument("--top-p", type=float, default=0.95)        # frozen §4 (ignored in think mode)
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    key = os.environ.get(args.api_key_env, "")
    if not key:
        sys.exit(f"Set {args.api_key_env} in this shell first.")

    modes = ["enabled", "disabled"] if args.thinking == "both" else [args.thinking]
    results = {}
    for mode in modes:
        print(f"-> POST {args.model}  thinking={mode}  max_tokens={args.max_tokens}")
        try:
            body = _call(args.base_url, key, args.model, mode,
                         args.max_tokens, args.temperature, args.top_p, args.timeout)
        except urllib.error.HTTPError as e:
            print(f"   HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:1000]}")
            return
        except Exception as e:  # noqa: BLE001 - probe prints any failure verbatim
            print(f"   {type(e).__name__}: {e}")
            return
        if "error" in body:
            print("   ERROR IN BODY:", json.dumps(body["error"], indent=2))
            return
        s = _summarize(body, args.max_tokens)
        results[mode] = s
        _print_one(mode, s)

    # Cross-mode checks -- the whole point of `both`.
    if "enabled" in results and "disabled" in results:
        en, dis = results["enabled"], results["disabled"]
        print("\n=== verdict ===")
        toggle_ok = en["reasoning_present"] and not dis["reasoning_present"]
        print(f"  toggle works (enabled has CoT, disabled does not): {toggle_ok}")
        # Token accounting: if reasoning_tokens is reported and completion_tokens >= it,
        # completion_tokens includes the reasoning -> the harness truncation guard is sound.
        rt = en["reasoning_tokens"]
        if rt is not None:
            includes = en["completion_tokens"] >= rt > 0
            print(f"  completion_tokens includes reasoning "
                  f"({en['completion_tokens']} >= reasoning {rt}): {includes}")
        else:
            print("  reasoning_tokens NOT in usage.completion_tokens_details -- inspect raw "
                  "usage above to confirm reasoning is counted in completion_tokens before T0.")
        if not toggle_ok:
            print("  WARNING: toggle did not behave as documented -- DO NOT lock until resolved.")


if __name__ == "__main__":
    main()
