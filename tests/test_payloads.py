import numpy as np

from ccap.payloads import (
    PayloadGenerator, array_to_bits, bits_to_array, bits_to_str,
    csv_to_bits, derive_seed, hamming, str_to_bits,
)


def test_payload_determinism():
    g1 = PayloadGenerator(123)
    g2 = PayloadGenerator(123)
    assert g1.generate(8, "m", "mech", 0) == g2.generate(8, "m", "mech", 0)
    # different labels -> (almost surely) different payloads
    assert g1.generate(16, "a") != g1.generate(16, "b")


def test_payload_length_and_alphabet():
    g = PayloadGenerator(0)
    bits = g.generate(12, "x")
    assert len(bits) == 12
    assert set(bits) <= {0, 1}


def test_bit_helpers_roundtrip():
    bits = (0, 1, 1, 0, 1)
    assert array_to_bits(bits_to_array(bits)) == bits
    assert str_to_bits(bits_to_str(bits)) == bits


def test_csv_bits_preserve_erasure():
    # comma form must survive erasure (-1) round-trips; compact 0/1 form must not be used
    bits = (0, -1, 1, -1)
    s = ",".join(str(b) for b in bits)
    assert csv_to_bits(s) == bits
    assert csv_to_bits("") == ()
    assert csv_to_bits(float("nan")) == ()


def test_hamming():
    assert hamming((0, 1, 0), (0, 1, 0)) == 0
    assert hamming((0, 1, 0), (1, 1, 1)) == 2
    assert hamming((0, 1), (0, 1, 1)) == 1  # length mismatch counts as error


def test_derive_seed_stable():
    assert derive_seed(7, "a", 1) == derive_seed(7, "a", 1)
    assert derive_seed(7, "a", 1) != derive_seed(7, "a", 2)
