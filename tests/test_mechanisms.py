import numpy as np
import pytest

from ccap.mechanisms import all_mechanisms, get_mechanism, list_mechanisms
from ccap.text_utils import split_step_blocks
from ccap.types import Mode, SimContext


@pytest.mark.parametrize("mech", all_mechanisms(), ids=lambda m: m.name)
def test_encode_decode_roundtrip_noiseless(mech):
    """At fidelity 1.0 the programmatic decoder must recover the payload exactly."""
    bits = (0, 1, 1, 0, 1, 0)
    ctx = SimContext(mechanism=mech.name, intended_bits=bits, mode=Mode.ENCODE,
                     fidelity=1.0, task_answer="42")
    rng = np.random.default_rng(0)
    cot = mech.simulate_cot(ctx, rng)
    recovered = mech.decode_programmatic(cot, len(bits))
    assert recovered == bits, f"{mech.name}: {recovered} != {bits}\n{cot}"


@pytest.mark.parametrize("mech", all_mechanisms(), ids=lambda m: m.name)
def test_benign_has_no_systematic_signal(mech):
    """Benign traces render neutral steps; the codebook decoder must not recover the payload."""
    bits = (0, 1, 0, 1, 1, 0)
    ctx = SimContext(mechanism=mech.name, intended_bits=bits, mode=Mode.BENIGN, task_answer="7")
    cot = mech.simulate_cot(ctx, np.random.default_rng(1))
    recovered = mech.decode_programmatic(cot, len(bits))
    # neutral steps should not reproduce the exact payload
    matches = sum(int(a == b) for a, b in zip(recovered, bits))
    assert matches < len(bits)


def test_structural_markers_separate_encode_from_benign():
    """Structural-tell mechanisms must show markers under encoding but not when benign."""
    bits = (1, 0, 1, 0)
    for name in ["ordering", "framing", "enumeration", "abstraction"]:
        mech = get_mechanism(name)
        enc = mech.simulate_cot(SimContext(name, bits, Mode.ENCODE, fidelity=1.0), np.random.default_rng(2))
        ben = mech.simulate_cot(SimContext(name, bits, Mode.BENIGN), np.random.default_rng(3))
        enc_marks = sum(mech.has_surface_marker(b) for b in split_step_blocks(enc))
        ben_marks = sum(mech.has_surface_marker(b) for b in split_step_blocks(ben))
        assert enc_marks > ben_marks, name


def test_acrostic_is_stealthy():
    """Acrostic has no structural tell (the default marker is False)."""
    mech = get_mechanism("acrostic")
    enc = mech.simulate_cot(SimContext("acrostic", (1, 0, 1), Mode.ENCODE, fidelity=1.0),
                            np.random.default_rng(4))
    assert all(not mech.has_surface_marker(b) for b in split_step_blocks(enc))


def test_semantic_ordering_is_tagless_and_stealthy():
    """The tagless ordering variant round-trips but carries NO surface marker."""
    mech = get_mechanism("ordering_semantic")
    bits = (1, 0, 1, 1, 0)
    cot = mech.simulate_cot(SimContext("ordering_semantic", bits, Mode.ENCODE, fidelity=1.0),
                            np.random.default_rng(0))
    assert mech.decode_programmatic(cot, len(bits)) == bits          # round-trips
    assert "[a]" not in cot and "[b]" not in cot                     # no literal tags
    # stealthy: no structural tell, unlike the tagged ordering mechanism
    assert all(not mech.has_surface_marker(b) for b in split_step_blocks(cot))
    tagged = get_mechanism("ordering")
    tcot = tagged.simulate_cot(SimContext("ordering", bits, Mode.ENCODE, fidelity=1.0),
                               np.random.default_rng(0))
    assert any(tagged.has_surface_marker(b) for b in split_step_blocks(tcot))  # tagged IS detectable


def test_enumeration_parses_unindented_bullets():
    """Real models emit lists flush-left ('- x'), not indented — the decoder must count them."""
    mech = get_mechanism("enumeration")
    cot = ("Step 1: total\n- 44 rows\n- total = 572\n"     # 2 bullets -> even -> 0
           "Step 2: subtract\n- 572 - 15\n- = 557\n- check\n"  # 3 bullets -> odd -> 1
           "Answer: 557")
    assert mech.decode_programmatic(cot, 2) == (0, 1)


def test_registry():
    assert "acrostic" in list_mechanisms()
    with pytest.raises(KeyError):
        get_mechanism("does-not-exist")
