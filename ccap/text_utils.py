"""Parsing helpers shared by encoders, simulators, and programmatic decoders.

The harness standardizes a *step block* CoT format so that programmatic decoders
can read surface features robustly from both real-model output and the simulator::

    <preamble line(s)>
    Step 1: <content...>
      - optional sub-item
      - optional sub-item
    Step 2: <content...>
    ...
    Answer: <final answer>

Decoders never assume the model is cooperative beyond following this format; a
missing/garbled step yields an *erasure* symbol (-1) which the capacity estimator
handles explicitly (it is part of the empirical channel, not swept under a BSC).
"""

from __future__ import annotations

import re

STEP_RE = re.compile(r"^\s*step\s*([0-9]+)\s*[:\.\)]\s*(.*)$", re.IGNORECASE)
ANSWER_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:\.\=]\s*(.*)$", re.IGNORECASE)
# Bullet / numbered sub-item. Allow ANY indentation (incl. column 0): real models
# routinely emit lists flush-left, e.g. "- 44 rows", not "  - 44 rows".
SUBITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.\)])\s+\S")

ERASURE = -1  # symbol used when a step is missing or unparseable

# Reasoning-model scratchpad span. Qwen3 / DeepSeek-R1 etc. emit <think>...</think>
# BEFORE the contracted answer. Tolerant of casing/whitespace ("</ think >").
THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)

# Harmony channel format (OpenAI gpt-oss). Reasoning goes to the `analysis` channel and
# the deliverable to the `final` channel; there are NO <think> tags. If the raw harmony
# text reaches us (server-side reasoning parser off), the analysis channel must be
# dropped or its planning text inflates the step count / entropy and its stray "Step i:"
# lines mis-align the covert slots (this zeroed gpt-oss Ĉ_ctrl in the first run).
#
# Two surface forms occur depending on whether special tokens were kept on decode:
#   * tokens kept:    <|channel|>analysis<|message|>...<|channel|>final<|message|>...
#   * tokens stripped (vLLM skip_special_tokens default): the channel control tokens
#     decode to empty, leaving the bare boundary word `assistantfinal` (or `final`),
#     e.g. "analysis...reasoning...assistantfinalStep 1: ...".
HARMONY_CHANNEL_RE = re.compile(r"<\|channel\|>\s*([A-Za-z0-9_]+)\s*<\|message\|>")
HARMONY_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
# Bare-word final-channel boundary (special tokens decoded away). The ONLY reliable
# marker is the collapsed channel token `assistantfinal`; a standalone "final" must NOT
# be used (it matches inside words like "finalize"/"finally" and truncates the answer).
HARMONY_BARE_FINAL_RE = re.compile(r"assistantfinal", re.IGNORECASE)


def is_harmony(text: str) -> bool:
    return bool(HARMONY_CHANNEL_RE.search(text) or HARMONY_BARE_FINAL_RE.search(text))


def _strip_harmony(text: str) -> str:
    """Keep only the last ``final`` channel's content; drop ``analysis`` and markers."""
    matches = list(HARMONY_CHANNEL_RE.finditer(text))
    if matches:
        finals = [m for m in matches if m.group(1).lower() == "final"]
        if not finals:
            return ""  # only analysis emitted (truncated before the final answer)
        rest = text[finals[-1].end():]
        nxt = HARMONY_TOKEN_RE.search(rest)   # cut at the next control token (<|return|>, <|end|>)
        if nxt:
            rest = rest[:nxt.start()]
        return rest.strip()
    # Bare-word form: take everything after the LAST final boundary.
    finals = list(HARMONY_BARE_FINAL_RE.finditer(text))
    if finals:
        return text[finals[-1].end():].strip()
    # Starts in the analysis channel but never reached `final` (truncated mid-reasoning):
    # no deliverable was emitted.
    if re.match(r"^\s*analysis", text, re.IGNORECASE):
        return ""
    return text


def strip_reasoning(text: str) -> str:
    """Return only the *delivered answer* — the model's private reasoning removed.

    The covert channel is defined on the contracted answer surface (the ``Step i:``
    blocks the sender is told to emit; see :mod:`ccap.prompts`), never the private
    reasoning scratchpad. For a reasoning model the scratchpad routinely *echoes
    the encoding contract verbatim*, so a decoder that reads it would score a
    spurious success. Therefore:

    * harmony channels (gpt-oss): keep only the ``final`` channel, drop ``analysis``;
      analysis-only output (no final yet) -> ``""``;
    * closed ``<think>...</think>`` -> text AFTER the last ``</think>``;
    * an OPEN ``<think>`` with no close (generation truncated mid-thought) -> ``""``
      (no answer was delivered -> the runner records ``no_parseable_cot``/erasure,
      the correct outcome);
    * no reasoning markers (non-reasoning / ``enable_thinking=False``) -> unchanged.
    """
    if not text:
        return text or ""
    if is_harmony(text):
        return _strip_harmony(text)
    closes = list(THINK_CLOSE_RE.finditer(text))
    if closes:
        return text[closes[-1].end():]
    if THINK_OPEN_RE.search(text):
        return ""  # unclosed reasoning span: nothing was delivered after it
    return text


def split_step_blocks(cot: str) -> list[str]:
    """Return the text of each ``Step i:`` block (header line + following sub-lines).

    Blocks are returned in the order they appear, regardless of the numbering the
    model used (we do not trust the model's own indices).
    """
    if not cot:
        return []
    lines = cot.splitlines()
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if STEP_RE.match(line):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif ANSWER_RE.match(line):
            if current is not None:
                blocks.append(current)
                current = None
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)
    return ["\n".join(b) for b in blocks]


def step_headline(block: str) -> str:
    """The content of a step block's first line, after the ``Step i:`` marker."""
    first = block.splitlines()[0] if block else ""
    m = STEP_RE.match(first)
    return m.group(2).strip() if m else first.strip()


def extract_answer(cot: str) -> str | None:
    """Return the text after the last ``Answer:`` marker, if present."""
    found = None
    for line in (cot or "").splitlines():
        m = ANSWER_RE.match(line)
        if m:
            found = m.group(1).strip()
    return found


def first_alpha(s: str) -> str | None:
    """First ASCII alphabetic character (lowercased), or None."""
    for ch in s:
        if ch.isalpha() and ch.isascii():
            return ch.lower()
    return None


def count_subitems(block: str) -> int:
    """Count indented bullet / numbered sub-items inside a step block."""
    n = 0
    for line in block.splitlines()[1:]:
        if SUBITEM_RE.match(line):
            n += 1
    return n


def contains_any(text: str, words: list[str]) -> bool:
    low = text.lower()
    return any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low) for w in words)


def count_any(text: str, words: list[str]) -> int:
    low = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(w.lower()) + r"\b", low)) for w in words)


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)
