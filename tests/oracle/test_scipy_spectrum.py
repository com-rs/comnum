"""The spectral metrics of comnum against SciPy (BSD-3-Clause).

SciPy implements Welch's method in ``scipy.signal.welch`` and the empirical
distribution of a sample in ``scipy.stats.ecdf``, independently of comnum. The outputs
are recorded fixtures (``tests/oracle/scipy_spectrum.json`` and ``.npz``, see
``tests/oracle/__init__.py``), with the signals they are computed from::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_scipy_spectrum.py

Compared, on white noise, a band-limited QPSK signal and a tone, for segments of 64,
255 (odd), 256 and 1000 samples and two sampling rates:

- :func:`comnum.psd` with ``welch(x, fs, window="hann", nperseg=n, noverlap=n - n // 2,
  detrend=False, return_onesided=False, scaling="density")`` (SciPy's ``hann`` of
  ``welch`` is the periodic window), its frequencies and density shifted to increasing
  frequencies;
- :func:`comnum.periodogram`, the same with ``window="boxcar"`` and ``noverlap=0``;
- :func:`comnum.aclr_db`, the ratio of the powers of the bands of SciPy's density
  (the sum of the bins in each band times the bin width, the definition of the
  function), for the QPSK signal;
- :func:`comnum.ccdf` with ``ecdf(values).sf.evaluate(thresholds)``, :math:`\\Pr(X > t)`,
  at thresholds that include values of the sample (ties).

The card ``scipy_spectrum`` holds the tolerances.
"""

import numpy as np
import pytest

import comnum
from oracle import oracle

TOOL = "SciPy"
NFFT = [64, 255, 256, 1000]
RATES = [1.0, 30.72e6]
THRESHOLDS = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def _signals():
    rng = np.random.default_rng(20261117)
    n = 5000
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    sym = (
        rng.integers(0, 2, n // 8) * 2 - 1 + 1j * (rng.integers(0, 2, n // 8) * 2 - 1)
    ) / np.sqrt(2)
    up = np.zeros(n, complex)
    up[::8] = sym
    t = np.arange(-32, 33) / 8
    taps = np.sinc(t) * np.hanning(t.size)  # a low-pass pulse of a quarter of the band
    qpsk = np.convolve(up, taps, mode="same")
    tone = np.exp(2j * np.pi * 0.1234 * np.arange(n))
    values = np.round(rng.exponential(1.5, 2000), 1)  # ties at every tenth
    return {"noise": noise, "qpsk": qpsk, "tone": tone, "values": values}


def _compute():
    from scipy.signal import welch
    from scipy.stats import ecdf

    sig = _signals()
    out = {"signals": sig, "psd": {}, "periodogram": {}, "aclr": {}}
    for name in ("noise", "qpsk", "tone"):
        for n in NFFT:
            for fs in RATES:
                key = f"{name}_{n}_{fs:g}"
                for kind, window, overlap in (
                    ("psd", "hann", n - n // 2),
                    ("periodogram", "boxcar", 0),
                ):
                    f, s = welch(sig[name], fs=fs, window=window, nperseg=n, noverlap=overlap,
                                 detrend=False, return_onesided=False, scaling="density")  # fmt: skip
                    order = np.argsort(f, kind="stable")
                    out[kind][key] = {"freqs": f[order], "density": s[order]}
    for n in NFFT:
        f = out["psd"][f"qpsk_{n}_1"]["freqs"]
        s = out["psd"][f"qpsk_{n}_1"]["density"]
        df = f[1] - f[0]

        def band(lo, hi, f=f, s=s, df=df):
            return float(np.sum(s[(f >= lo) & (f < hi)]) * df)

        main = band(-0.125, 0.125)
        adj = max(band(-0.375, -0.125), band(0.125, 0.375))
        out["aclr"][str(n)] = 10 * np.log10(main / adj)
    out["ccdf"] = ecdf(sig["values"]).sf.evaluate(np.array(THRESHOLDS))
    return out


@pytest.fixture(scope="module")
def scipy_spectrum():
    return oracle("scipy_spectrum", _compute, tool=TOOL)


@pytest.mark.parametrize("kind", ["psd", "periodogram"])
@pytest.mark.parametrize("fs", RATES)
@pytest.mark.parametrize("n", NFFT)
@pytest.mark.parametrize("name", ["noise", "qpsk", "tone"])
def test_the_averaged_periodograms_are_those_of_scipy(scipy_spectrum, kind, name, n, fs):
    ref = scipy_spectrum[kind][f"{name}_{n}_{fs:g}"]
    f, s = getattr(comnum, kind)(scipy_spectrum["signals"][name], fs=fs, nfft=n)
    np.testing.assert_allclose(f, ref["freqs"], rtol=1e-12, atol=1e-12 * fs)
    np.testing.assert_allclose(s, ref["density"], rtol=1e-9, atol=1e-12 * np.max(ref["density"]))


@pytest.mark.parametrize("n", NFFT)
def test_the_aclr_is_that_of_the_density_of_scipy(scipy_spectrum, n):
    ref = scipy_spectrum["psd"][f"qpsk_{n}_1"]
    got = comnum.aclr_db(ref["freqs"], ref["density"], 0.25, 0.25)
    assert got == pytest.approx(scipy_spectrum["aclr"][str(n)], abs=1e-9)


def test_the_ccdf_is_the_survival_function_of_scipy(scipy_spectrum):
    got = comnum.ccdf(scipy_spectrum["signals"]["values"], THRESHOLDS)
    np.testing.assert_allclose(got, scipy_spectrum["ccdf"], rtol=0, atol=1e-15)
