"""comnum against OptiCommPy on a random grid of parameters, edge cases included.

``test_oracle.py`` compares comnum with OptiCommPy (GPL-3.0) at a few hand-picked
points. This file draws the parameters at random (comnum's generator, seed ``SEED``,
so that the grid does not depend on NumPy's) and adds the edges: the roll-offs that
put a singularity of the root-raised cosine on a sample (:math:`t = \\pm T/(4\\beta)`),
a roll-off of 1, filters of one tap and longer than the signal, odd numbers of taps,
1024-QAM, 64-PSK and 16-PAM, signal-to-noise ratios from -2 to 30 dB. OptiCommPy's
outputs are one recorded fixture (``tests/oracle/opticommpy_grid.json`` and ``.npz``,
see ``tests/oracle/__init__.py``); a case where OptiCommPy raises is recorded with its
message::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_opticommpy_grid.py

The card ``opticommpy_grid`` states what agrees and where the two differ.
"""

import numpy as np
import pytest

import comnum
from oracle import oracle

TOOL = "OptiCommPy"
SEED = 20261117
FAMILIES = {"qam": [4, 16, 64, 256, 1024], "psk": [4, 8, 16, 32, 64], "pam": [2, 4, 8, 16]}


def _grid():
    """The random cases, drawn by comnum's generator (stable across NumPy versions)."""
    rng = comnum.Rng(SEED)

    def pick(seq):
        return seq[int(rng.integers(len(seq), 1)[0])]

    def uniform(lo, hi):
        return float(lo + (hi - lo) * rng.uniform(1)[0])

    kinds = list(FAMILIES)
    theory = [
        (k, pick(FAMILIES[k]), round(uniform(-2.0, 30.0), 3))
        for k in (pick(kinds) for _ in range(60))
    ]
    pulses = []
    for _ in range(30):
        sps = pick([2, 3, 4, 5, 8, 16])
        pulses.append(
            (
                pick(["rrc", "rc"]),
                sps,
                2 * int(rng.integers(512, 1)[0]) + 16,
                round(uniform(0.01, 1.0), 3),
            )
        )
    # the singularities t = ±T/(4 beta) on a sample: beta = sps / (4 j) for a whole j
    pulses += [("rrc", 4, 256, 0.25), ("rrc", 4, 256, 0.5), ("rrc", 8, 512, 0.25), ("rrc", 2, 128, 0.5),
               ("rrc", 4, 256, 1.0), ("rc", 4, 256, 1.0), ("rc", 4, 256, 0.25)]  # fmt: skip
    firs = [(int(rng.integers(64, 1)[0]) + 1, int(rng.integers(300, 1)[0]) + 1) for _ in range(20)]
    firs += [(1, 50), (80, 40), (101, 101)]
    lowpass = [(round(uniform(0.01, 0.49), 4), int(rng.integers(200, 1)[0]) + 2, pick(["rect", "gauss"]))
               for _ in range(20)]  # fmt: skip
    links = [
        (k, pick(FAMILIES[k]), round(uniform(0.0, 30.0), 2))
        for k in (pick(kinds) for _ in range(20))
    ]
    return {"theory": theory, "pulses": pulses, "firs": firs, "lowpass": lowpass, "links": links}


GRID = _grid()


def _signals(i, n_taps, n):
    rng = comnum.Rng(SEED + i)
    return rng.gaussian(n_taps), rng.complex_noise(n)


def _link(i, kind, m, snr):
    rng = comnum.Rng(SEED + 100 + i)
    c = comnum.Constellation(kind, m)
    b = comnum.bits(c.bits_per_symbol * 400, rng)
    x = c.map_bits(b)
    noise = rng.complex_noise(x.size)
    scale = np.sqrt(np.mean(np.abs(x) ** 2) / 10 ** (snr / 10))
    y = x + scale * (noise if kind != "pam" else noise.real)
    return c, b, x, y


def _compute():
    from optic.comm import metrics as om
    from optic.comm import modulation as omod
    from optic.dsp import core as ocore
    from optic.utils import parameters

    def attempt(f):
        try:
            return {"value": f()}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def pulse(kind, sps, n, beta):
        p = parameters()
        p.pulseType, p.SpS, p.nFilterTaps, p.rollOff = kind, sps, n, beta
        return np.asarray(ocore.pulseShape(p), float)

    out = {
        "theory": [
            attempt(lambda c=c: float(om.theoryBER(c[1], c[2], c[0]))) for c in GRID["theory"]
        ],
        "pulses": [attempt(lambda c=c: pulse(*c)) for c in GRID["pulses"]],
        "firs": [],
        "lowpass": [
            attempt(lambda c=c: np.asarray(ocore.lowPassFIR(c[0], 1.0, c[1], c[2])))
            for c in GRID["lowpass"]
        ],
        "links": [],
    }
    for i, (n_taps, n) in enumerate(GRID["firs"]):
        h, x = _signals(i, n_taps, n)
        out["firs"].append(attempt(lambda h=h, x=x: np.asarray(ocore.firFilter(h, x))))
    for i, (kind, m, snr) in enumerate(GRID["links"]):
        _c, b, x, y = _link(i, kind, m, snr)
        out["links"].append({
            "modulated": attempt(lambda b=b, m=m, kind=kind: np.asarray(omod.modulateGray(b, m, kind)).ravel()),
            "demodulated": attempt(lambda y=y, m=m, kind=kind: np.asarray(omod.demodulateGray(y, m, kind)).ravel()),
            "evm": attempt(lambda y=y, x=x, m=m, kind=kind: float(om.calcEVM(y, m, kind, x)[0])),
        })  # fmt: skip
    return out


@pytest.fixture(scope="module")
def grid():
    return oracle("opticommpy_grid", _compute, tool=TOOL)


@pytest.mark.parametrize("i", range(len(GRID["theory"])))
def test_theory_ber(grid, i):
    """Above 1e-9 the two agree to 1e-6 relative. Below, OptiCommPy's Qfunc, written
    0.5 - 0.5 erf(x / sqrt 2) where its docstring (eq. (1)) has erfc(x / sqrt 2) / 2,
    cancels: its error is that of erf near 1, about 1e-16, and it reaches 0 below
    1e-17, where comnum keeps every digit down to 1e-310 (the exact Gray BER of PAM and
    QAM, ``theory_ber_gray``, to 1e-12)."""
    kind, m, ebn0 = GRID["theory"][i]
    ref = grid["theory"][i]["value"]
    ours = comnum.theory_ber(m, ebn0, kind)
    if ref >= 1e-9:
        assert ours == pytest.approx(ref, rel=1e-6)
    else:
        assert abs(ours - ref) <= 1e-15
        if kind in ("pam", "qam"):
            assert ours == pytest.approx(comnum.theory_ber_gray(m, ebn0, kind), rel=1e-12)


@pytest.mark.parametrize("i", range(len(GRID["pulses"])))
def test_pulse_shape(grid, i):
    kind, sps, n, beta = GRID["pulses"][i]
    ref = grid["pulses"][i]
    assert "value" in ref, ref
    np.testing.assert_allclose(
        comnum.pulse_shape(kind, sps, n_taps=n, rolloff=beta), ref["value"], rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("i", range(len(GRID["firs"])))
def test_fir_filter(grid, i):
    h, x = _signals(i, *GRID["firs"][i])
    ref = grid["firs"][i]
    assert "value" in ref, ref
    np.testing.assert_allclose(comnum.fir_filter(h, x), ref["value"], atol=1e-12)


@pytest.mark.parametrize("i", range(len(GRID["lowpass"])))
def test_low_pass_fir(grid, i):
    fc, n, kind = GRID["lowpass"][i]
    ref = grid["lowpass"][i]
    assert "value" in ref, ref
    np.testing.assert_allclose(
        comnum.low_pass_fir(fc, 1.0, n, kind=kind), ref["value"], rtol=1e-10, atol=1e-15
    )


@pytest.mark.parametrize("i", range(len(GRID["links"])))
def test_modulation_and_evm(grid, i):
    kind, m, snr = GRID["links"][i]
    c, _b, x, y = _link(i, kind, m, snr)
    ref = grid["links"][i]
    np.testing.assert_allclose(x, ref["modulated"]["value"], atol=1e-12)
    np.testing.assert_array_equal(c.demap_bits(y), ref["demodulated"]["value"])
    assert comnum.evm(y, c, tx=x) == pytest.approx(ref["evm"]["value"], rel=1e-9)
