"""Recordings (SigMF), the least-squares information rates, single-precision split step
and the verification of specifications from Python."""

import json

import numpy as np
import pytest
from conftest import ROOT, qpsk_spec

import comnum


def test_sigmf_round_trip_with_metadata(tmp_path):
    rng = comnum.Rng(1)
    x = np.round(rng.complex_noise(1000, 1e6))
    comnum.write_sigmf(
        tmp_path / "rec", x, 80e9, datatype="ci16_le", global_meta={"comnum:symbol_rate": 32e9}
    )
    y, fs, meta = comnum.read_sigmf(tmp_path / "rec.sigmf-meta")
    np.testing.assert_array_equal(
        y, np.clip(x.real, -32768, 32767) + 1j * np.clip(x.imag, -32768, 32767)
    )
    assert fs == 80e9 and meta["global"]["comnum:symbol_rate"] == 32e9
    assert meta["global"]["core:datatype"] == "ci16_le"
    raw = np.fromfile(tmp_path / "rec.sigmf-data", dtype="<i2")
    assert raw.size == 2 * x.size
    with pytest.raises(ValueError, match="data type"):
        comnum.write_sigmf(tmp_path / "bad", x, 1.0, datatype="cf16_be")


def test_the_synthetic_capture_is_readable():
    y, fs, meta = comnum.read_sigmf(ROOT / "examples" / "data" / "synthetic_16qam.sigmf-meta")
    ref = np.load(ROOT / "examples" / "data" / "synthetic_16qam_reference.npy")
    assert (y.size, fs, ref.size) == (32768, 80e9, 4096)
    assert meta["global"]["comnum:symbol_rate"] == 32e9 and ref.max() < 16


def test_least_squares_rates_are_consistent_where_the_opticommpy_estimators_are_biased():
    c = comnum.Constellation("qam", 16).normalized()
    rng = comnum.Rng(2)
    idx, x = comnum.symbols(c, 60_000, rng)
    y = comnum.awgn(x, 0.0, rng) * 0.3 * np.exp(0.7j)  # any complex gain
    r = comnum.information_rates(y, idx, c)
    assert r["mi"] == pytest.approx(comnum.theory_mi(16, 0.0), abs=0.01)
    assert r["gmi"] == pytest.approx(comnum.theory_gmi(16, 0.0), abs=0.01)
    assert r["snr_db"] == pytest.approx(0.0, abs=0.05)
    # the power normalisation of OptiCommPy's estimator loses about 0.06 bit at 0 dB
    assert comnum.monte_carlo_mi(y, x, c) < r["mi"] - 0.04


def test_single_precision_split_step():
    link = comnum.Ssfm(
        comnum.FiberParams(n_spans=3), step_km=0.5, amplification="ideal", check_sampling=False
    )
    x = comnum.Rng(3).complex_noise(4096, 2e-3)
    a = link.propagate(x, 64e9)
    b = link.propagate(x, 64e9, precision="f32")
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-4
    with pytest.raises(ValueError, match="precision"):
        link.propagate(x, 64e9, precision="f16")


def test_the_effective_length_step_runs_in_double_precision_only():
    fibre = comnum.FiberParams(n_spans=1)
    x = comnum.Rng(3).complex_noise(1024, 2e-3)
    link = comnum.Ssfm(fibre, nonlinear_step="effective_length", check_sampling=False)
    assert "nonlinear_step='effective_length'" in repr(link)
    y = link.propagate(x, 64e9)
    z = comnum.Ssfm(fibre, check_sampling=False).propagate(x, 64e9)
    # the two weightings of the attenuation within a 1 km step: close, not equal
    assert 1e-10 < np.linalg.norm(y - z) / np.linalg.norm(z) < 1e-3
    with pytest.raises(ValueError, match="midpoint"):
        link.propagate(x, 64e9, precision="f32")
    with pytest.raises(ValueError, match="effective_length"):
        comnum.Ssfm(fibre, nonlinear_step="euler")


def test_verify_spec_reports_every_check():
    out = comnum.verify_spec(ROOT / "specs" / "validation" / "proakis_awgn.yaml")
    assert out["passed"] and len(out["checks"]) > 20
    c = out["checks"][0]
    assert c["source"].startswith("Proakis") and c["lo"] <= c["value"] <= c["hi"]
    bad = qpsk_spec(name="wrong", expect=[{"metric": "ber", "value": 0.1, "rtol": 0.1}])
    out = comnum.verify_spec(bad)
    assert not out["passed"] and json.dumps(out["checks"])
    with pytest.raises(ValueError, match="expect"):
        comnum.verify_spec({**bad, "expect": []})
