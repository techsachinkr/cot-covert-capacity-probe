"""Reasoning-strip + truncation-exclusion regression tests.

These guard the two think-mode measurement bugs:
  * the decoder must read the DELIVERED answer, not the <think> scratchpad (which
    echoes the encoding contract verbatim -> spurious perfect decodes);
  * a generation that hit the token budget must be excluded, not scored as a failure.
"""

from ccap.decoders.model_decoder import parse_bits
from ccap.runner import ExperimentRunner
from ccap.text_utils import split_step_blocks, strip_reasoning


def test_strip_closed_block_returns_only_answer():
    raw = ("<think>\nStep 1: [b] then [a]\nStep 2: [b] then [a]\nlots of planning\n</think>\n\n"
           "Step 1: setup [a] then [b]\nStep 2: result\nAnswer: 71")
    out = strip_reasoning(raw)
    assert "planning" not in out and "<think>" not in out and "</think>" not in out
    # the scratchpad's "Step" lines must NOT leak into the parsed answer steps
    assert len(split_step_blocks(out)) == 2


def test_strip_unclosed_block_is_empty():
    # truncated mid-thought (like the real trial #27): no answer was ever delivered
    raw = "<think>\nStep 1: [a] then [b]\nStep 2: [a] then [b]\n...(cut off at the token cap"
    assert strip_reasoning(raw) == ""
    assert split_step_blocks(strip_reasoning(raw)) == []


def test_strip_no_think_is_unchanged():
    raw = "Step 1: foo\nStep 2: bar\nAnswer: 3"
    assert strip_reasoning(raw) == raw


def test_strip_is_case_insensitive_and_uses_last_close():
    raw = "pre<THINK>a</THINK>mid</think>Answer: x"
    assert strip_reasoning(raw) == "Answer: x"


def test_strip_handles_empty_and_none():
    assert strip_reasoning("") == ""
    assert strip_reasoning(None) == ""


def test_strip_harmony_keeps_only_final_channel():
    # gpt-oss harmony: the `analysis` channel is private reasoning and must be dropped,
    # else its planning text (incl. stray "Step i:" lines) inflates steps and mis-aligns slots.
    raw = ("<|channel|>analysis<|message|>Let me plan.\nStep 1: scratch\nStep 2: scratch"
           "<|end|><|channel|>final<|message|>Step 1: real\nStep 2: real\nAnswer: 42<|return|>")
    out = strip_reasoning(raw)
    assert "scratch" not in out and "analysis" not in out and "<|" not in out
    assert len(split_step_blocks(out)) == 2
    assert "Answer: 42" in out


def test_strip_harmony_analysis_only_is_empty():
    # only the analysis channel emitted (truncated before `final`) -> no delivered answer
    raw = "<|channel|>analysis<|message|>Step 1: thinking hard, ran out of budget"
    assert strip_reasoning(raw) == ""
    assert split_step_blocks(strip_reasoning(raw)) == []


def test_strip_harmony_bare_assistantfinal_boundary():
    # vLLM skip_special_tokens decodes the channel tokens to empty, leaving the bare
    # `assistantfinal` boundary (this is the form actually preserved in the run shards).
    raw = ("analysisWe need 16 steps each starting 'Step i:'.\nStep 1: scratch plan\n"
           "Step 2: scratch plan...accordingly.assistantfinalStep 1: real one\n"
           "Step 2: real two\nAnswer: 42")
    out = strip_reasoning(raw)
    assert "scratch" not in out and "analysis" not in out.lower()
    assert len(split_step_blocks(out)) == 2
    assert "Answer: 42" in out


def test_parse_bits_strips_decoder_think_span():
    # a reasoning DECODER (e.g. Qwen3-32B) emits <think> before its bit answer; the
    # scratchpad's numerals must not pollute the read, and the answer must still parse.
    assert parse_bits("<think>slot 0 is 1, slot 1 is 0...</think>10", 2) == (1, 0)
    # unclosed think (ran out of budget mid-thought) -> all erasures, not a garbage read
    assert parse_bits("<think>still reasoning 0101 and never finished", 4) == (-1, -1, -1, -1)


def test_exclusion_truncated_takes_precedence():
    assert ExperimentRunner._exclusion(None, 16, True) == (True, "truncated")
    assert ExperimentRunner._exclusion(None, 0, True) == (True, "truncated")
    assert ExperimentRunner._exclusion(None, 0, False) == (True, "no_parseable_cot")
    assert ExperimentRunner._exclusion("boom", 16, False) == (True, "generation_error")
    assert ExperimentRunner._exclusion(None, 16, False) == (False, "")
