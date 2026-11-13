"""The flat fading channel as one object (`comnum.Fading`) and the thematic namespaces."""

import numpy as np
import pytest

import comnum


@pytest.mark.parametrize(
    ("kind", "parameter", "fourth_moment"),
    [
        ("rayleigh", None, 2.0),
        ("rice", 4.0, 1.0 + 9.0 / 25.0),
        ("nakagami", 0.5, 3.0),
        ("nakagami", 3.0, 4.0 / 3.0),
        ("hoyt", 0.5, 2.0 + 0.36),
    ],
)
def test_the_gains_have_the_moments_of_their_distribution(kind, parameter, fourth_moment):
    h = comnum.Fading(kind, parameter).gains(400_000, comnum.Rng(3))
    power = np.abs(h) ** 2
    assert power.mean() == pytest.approx(1.0, abs=0.01)
    assert (power**2).mean() == pytest.approx(fourth_moment, rel=0.03)


def test_the_object_and_the_functions_give_the_same_closed_forms():
    eb = np.array([0.0, 10.0, 20.0])
    ch = comnum.Fading("nakagami", 2.0)
    by_name = comnum.theory_ber_binary_fading(eb, fading="nakagami", parameter=2.0)
    assert np.array_equal(ch.ber(eb), by_name)
    assert np.array_equal(comnum.theory_ber_binary_fading(eb, fading=ch), by_name)
    assert np.array_equal(ch.ser(16, eb), comnum.theory_ser_fading(16, eb, fading=ch))
    assert np.array_equal(
        ch.ber_noncoherent(eb), comnum.theory_ber_noncoherent_binary_fading(eb, fading=ch)
    )
    # the default channel of the functions is Rayleigh fading
    assert np.array_equal(comnum.theory_ber_binary_fading(eb), comnum.Fading().ber(eb))
    assert comnum.Fading().ber(10.0) == pytest.approx(
        comnum.theory_ber_rayleigh_mrc(10.0), rel=1e-12
    )
    # the MGF: (1 - s mean / m)^-m
    assert ch.mgf(-1.0, 10.0) == pytest.approx(1.0 / 36.0, rel=1e-12)


def test_the_coherence_the_absence_of_fading_and_the_errors():
    rng = comnum.Rng(1)
    h = comnum.Fading("rice", 3.0).gains(10, rng, coherence=4)
    assert h[0] == h[3] and h[4] == h[7] and h[3] != h[4]
    assert np.array_equal(comnum.Fading("awgn").gains(3, rng), np.ones(3))
    ch = comnum.Fading("Nakagami", 2.0)
    assert (ch.kind, ch.parameter, repr(ch)) == ("nakagami", 2.0, 'Fading("nakagami", 2.0)')
    assert comnum.Fading("rayleigh", 5.0).parameter is None
    for kind, parameter in [("weibull", None), ("rice", None), ("hoyt", 2.0), ("nakagami", 0.2)]:
        with pytest.raises(ValueError):
            comnum.Fading(kind, parameter)
    with pytest.raises(ValueError, match="coherence"):
        ch.gains(4, rng, coherence=0)
    with pytest.raises(ValueError, match="name or a Fading"):
        comnum.theory_ber_binary_fading(10.0, fading=3)


def test_a_simulated_link_follows_the_theory_of_its_channel():
    ch = comnum.Fading("rice", 3.0)
    rng = comnum.Rng(7)
    n = 400_000
    bits = comnum.bits(n, rng)
    h = ch.gains(n, rng)
    y = h * (1.0 - 2.0 * bits) + rng.complex_noise(n, comnum.db2lin(-8.0))
    ber = np.mean(((y / h).real < 0) != (bits == 1))
    assert ber == pytest.approx(ch.ber(8.0), rel=0.05)


def test_the_namespaces_hold_the_objects_of_the_top_level():
    assert comnum.theory.ber is comnum.theory_ber
    assert comnum.theory.ser_fading is comnum.theory_ser_fading
    assert comnum.optical.Ssfm is comnum.Ssfm
    assert comnum.channels.Fading is comnum.Fading
    names = [
        "theory", "coding", "waveforms", "mimo", "channels",
        "optical", "hardware", "dsp", "metrics", "systems",
    ]  # fmt: skip
    for name in names:
        module = getattr(comnum, name)
        assert module.__all__, name
        for member in module.__all__:
            top = f"theory_{member}" if name == "theory" else member
            assert getattr(module, member) is getattr(comnum, top), (name, member)
    # every closed form is in comnum.theory
    closed_forms = {n.removeprefix("theory_") for n in comnum.__all__ if n.startswith("theory_")}
    assert closed_forms == set(comnum.theory.__all__)
