"""The orthogonal FSK of comnum against liquid-dsp (MIT).

liquid-dsp (<https://liquidsdr.org>) is a C library of software radio; its ``fskmod``
and ``fskdem`` are an M-FSK modulator (a numerically controlled oscillator, float32)
and a noncoherent demodulator (the largest bin of an FFT of the symbol). Their outputs
are recorded fixtures (``tests/oracle/liquid_fsk.json`` and ``.npz``, see
``tests/oracle/__init__.py``), the library loaded with ``ctypes``
(``oracle.liquid_library``)::

    brew install liquid-dsp
    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_liquid_fsk.py

liquid centres its tones: symbol :math:`s` is the frequency :math:`(s - (M-1)/2)\\,
b/((M-1)/2)` cycles per sample, :math:`b` its ``bandwidth``. With :math:`b = \\frac{M-1}{2}
\\frac{\\Delta}{N}`, they are the tones :math:`s \\Delta / N` of :class:`comnum.Fsk`
(:math:`N` samples per symbol, spacing :math:`\\Delta`) shifted by :math:`-\\frac{M-1}{2}
\\Delta` cycles per symbol, a whole number for an even :math:`\\Delta`: the signals of
the two are the same up to the factor :math:`e^{-j 2\\pi \\frac{M-1}{2} \\Delta n / N}`.

Compared, for 2-, 4- and 8-FSK (:math:`N` = 8, 16, 32, :math:`\\Delta` = 2) and 4-FSK with
:math:`\\Delta` = 4:

- the waveform of 64 symbols, liquid's against comnum's times the factor;
- the decisions of ``fskdem`` on 250 symbols in white noise at :math:`E_s/N_0` = 3 dB
  (complex64 samples) against those of :meth:`comnum.Fsk.demodulate` on the same
  samples divided by the factor (recorded with them, in double precision).

The card ``liquid_fsk`` holds the comparisons.
"""

import numpy as np
import pytest

import comnum
from oracle import liquid_library, oracle

# (bits per symbol, samples per symbol, spacing)
CASES = [(1, 8, 2), (2, 16, 2), (3, 32, 2), (2, 16, 4)]


def key(bits, n, spacing):
    return f"{2**bits}_{n}_{spacing}"


def shift(m, n, spacing, length):
    """The factor from comnum's tones to liquid's."""
    return np.exp(-2j * np.pi * (m - 1) / 2 * spacing * np.arange(length) / n)


def _compute():
    lib = liquid_library()
    rng = np.random.default_rng(20261117)
    out = {}
    for bits, n, spacing in CASES:
        m = 2**bits
        bw = (m - 1) / 2 * spacing / n
        mod = lib.fskmod_create(bits, n, bw)
        sym = rng.integers(0, m, 64)
        buf = np.zeros(n, np.complex64)
        wave = []
        for s in sym:
            lib.fskmod_modulate(mod, int(s), buf.ctypes.data)
            wave.append(buf.copy())
        lib.fskmod_destroy(mod)
        tx = rng.integers(0, m, 250)
        x = np.asarray(comnum.Fsk(m, n, spacing=spacing).modulate(tx)) * shift(
            m, n, spacing, 250 * n
        )
        noise_var = n / 10 ** (3.0 / 10)  # Es/N0 = N / sigma^2
        w = np.sqrt(noise_var / 2) * (
            rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
        )
        y = (x + w).astype(np.complex64)
        dem = lib.fskdem_create(bits, n, bw)
        dec = [lib.fskdem_demodulate(dem, np.ascontiguousarray(y[i * n:(i + 1) * n]).ctypes.data)
               for i in range(tx.size)]  # fmt: skip
        lib.fskdem_destroy(dem)
        wave = np.concatenate(wave)
        out[key(bits, n, spacing)] = {
            "symbols": sym, "waveform": wave, "sent": tx, "decisions": np.array(dec, np.int64),
            # the signals of liquid brought to the tones of comnum (divided by the factor,
            # in double precision): the inputs and outputs of the card
            "waveform_comnum": wave / shift(m, n, spacing, wave.size),
            "received_comnum": y / shift(m, n, spacing, y.size),
        }  # fmt: skip
    return out


@pytest.fixture(scope="module")
def liquid_fsk():
    return oracle("liquid_fsk", _compute, tool="liquid-dsp")


@pytest.mark.parametrize(("bits", "n", "spacing"), CASES)
def test_the_waveform_is_that_of_liquid(liquid_fsk, bits, n, spacing):
    ref = liquid_fsk[key(bits, n, spacing)]
    m = 2**bits
    x = np.asarray(comnum.Fsk(m, n, spacing=spacing).modulate(ref["symbols"]))
    # liquid's oscillator runs in float32: 6e-5 after 64 symbols
    np.testing.assert_allclose(ref["waveform"], x * shift(m, n, spacing, x.size), atol=1e-4)
    np.testing.assert_allclose(ref["waveform_comnum"], x, atol=1e-4)


@pytest.mark.parametrize(("bits", "n", "spacing"), CASES)
def test_the_decisions_in_noise_are_those_of_liquid(liquid_fsk, bits, n, spacing):
    ref = liquid_fsk[key(bits, n, spacing)]
    m = 2**bits
    ours = comnum.Fsk(m, n, spacing=spacing).demodulate(ref["received_comnum"])
    np.testing.assert_array_equal(ours, ref["decisions"])
    assert 0.05 < np.mean(ours != ref["sent"]) < 0.7  # errors to agree on
