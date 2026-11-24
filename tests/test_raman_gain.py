"""The Raman gain spectrum of the ``gain`` mapping, as the simulators use it."""

import numpy as np
import pytest

import comnum

F_REF_THZ = comnum.raman_gain([0.0])["reference_hz"] / 1e12


def test_the_default_is_the_table_of_standard_single_mode_fibre():
    g = comnum.raman_gain([0.0, 13.25e12, 50e12])
    assert g["gain_m_w"][0] == 0.0 and g["gain_m_w"][2] == 0.0
    assert abs(g["gain_m_w"][1] - 3.29e-14) < 0.01e-14
    assert abs(g["efficiency_w_km"][1] - g["gain_m_w"][1] / 80e-12 * 1e3) < 1e-9
    assert abs(g["peak_m_w"] - 3.33e-14) < 0.01e-14 and 12.5e12 <= g["peak_hz"] <= 13.5e12
    assert abs(g["reference_hz"] - 206.18e12) < 0.01e12


@pytest.mark.parametrize("model", ["blow_wood", "lin_agrawal", "hollenbeck_cantrell"])
def test_a_model_is_scaled_to_the_peak_of_the_table_or_to_peak_m_w(model):
    f = np.arange(0.0, 42e12, 0.1e12)
    table = comnum.raman_gain(f)
    g = comnum.raman_gain(f, {"model": model, "reference_thz": F_REF_THZ})
    assert abs(g["peak_m_w"] - table["peak_m_w"]) < 1e-28 and abs(g["peak_hz"] - 13.2e12) <= 0.3e12
    assert abs(g["reference_hz"] - F_REF_THZ * 1e12) < 1.0
    h = comnum.raman_gain(f, {"model": model, "reference_thz": F_REF_THZ, "peak_m_w": 1e-14})
    assert np.allclose(h["gain_m_w"], g["gain_m_w"] / g["peak_m_w"] * 1e-14)
    rms = np.sqrt(np.mean(((g["gain_m_w"] - table["gain_m_w"]) / table["peak_m_w"]) ** 2))
    assert rms < (0.05 if model == "hollenbeck_cantrell" else 0.15)


def test_a_table_and_the_errors_of_the_mapping():
    g = comnum.raman_gain(
        [0.0, 7e12, 14e12],
        {"offsets_thz": [0.0, 14.0], "gain_m_w": [0.0, 2e-14], "reference_thz": 206.0},
    )
    assert np.allclose(g["gain_m_w"], [0.0, 1e-14, 2e-14])
    for bad in (
        {"reference_thz": 206.0},
        {"model": "stolen", "reference_thz": 206.0},
        {
            "model": "blow_wood",
            "offsets_thz": [0.0, 1.0],
            "gain_m_w": [0.0, 1e-14],
            "reference_thz": 206.0,
        },
    ):
        with pytest.raises(ValueError):
            comnum.raman_gain([1e12], bad)
    with pytest.raises(ValueError, match="aeff_um2"):
        comnum.raman_gain([1e12], aeff_um2=0.0)
