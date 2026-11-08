"""comnum against GNPy, the oracle of the Raman amplification.

GNPy (Telecom Infra Project, BSD-3-Clause) solves the same coupled power equations of
stimulated Raman scattering (explicit Euler steps, forward-backward sweeps for the
counter-propagating pumps), the same spontaneous noise of the pumps and the generalised
GN model with power profiles (spectrally separated: SPM and XPM). Its Raman coefficient
matrix is replaced by that of comnum, written again here from its definition: GNPy scales
the effective area with the frequency, and its pumps lose the power that the signals
gain where comnum conserves the photons (the pump loses ``f_p/f_s`` times that power).
The documented differences are tested as such:

- GNPy integrates by Euler's method: its profiles converge to those of comnum (RK4) at
  first order in its step (the S+C+L span extrapolates two fine steps to a zero step);
- the NLI of GNPy converges to that of comnum as its frequency grid is refined (its
  ``dispersion_tolerance`` and ``phase_shift_tolerance``); at its default tolerances it
  underestimates the NLI of counter-pumped spans by about 1 dB;
- comnum, as GNPy, takes the pumps alone as the sources of spontaneous scattering.

The outputs of GNPy are recorded fixtures (``tests/oracle/<name>.json``, see
``tests/oracle/__init__.py``): every ``oracle(...)`` call below names the fixture and the
function that computes it, and GNPy is imported only inside that function, when
recording. The inputs (channel grids, pumps, fibre) are fixed here. The tests run from
the files alone and skip when a file is missing. To re-record, with GNPy installed
(``pip install --group oracle``)::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_gnpy_oracle.py

Re-recording is a deliberate act: the commit that changes the fixtures says why and
names the version of GNPy (the ``version`` field of the files).
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

import comnum
from oracle import oracle

TOOL = "GNPy"
C0 = 299_792_458.0
FC = 193.1e12


def coupling(freqs, aeff=80e-12):
    """The Raman coefficients of comnum (1/(W m)): the gain efficiency of the lower wave,
    the loss of the higher one scaled by the ratio of the frequencies."""
    from gnpy.core.parameters import DEFAULT_RAMAN_COEFFICIENT as d

    def eff(fs, fp):
        g = np.interp(fp - fs, d["frequency_offset"], d["gamma_raman"], right=0.0)
        return g * fp / d["reference_frequency"] / aeff

    n = len(freqs)
    cr = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            if freqs[j] > freqs[k]:
                cr[k, j] = eff(freqs[k], freqs[j])
            elif freqs[j] < freqs[k]:
                cr[k, j] = -(freqs[k] / freqs[j]) * eff(freqs[j], freqs[k])
    return cr


def pump_frequency(p):
    return p["frequency_thz"] * 1e12 if "frequency_thz" in p else C0 / (p["wavelength_nm"] * 1e-9)


def gnpy_pumps(pumps):
    return [
        {
            "power": p["power_mw"] * 1e-3,
            "frequency": pump_frequency(p),
            "propagation_direction": "coprop" if p["direction"] == "co" else "counterprop",
        }
        for p in pumps
    ]


class Span:
    """The fibre GNPy's Raman solver reads: the coupling of comnum for the waves of
    ``freqs``, a uniform attenuation, the pumps."""

    def __init__(self, length_km, freqs, pumps):
        self.params = NS(length=length_km * 1e3)
        self.lumped_losses = np.array([])
        self.z_lumped_losses = np.array([])
        self.temperature = 300.0
        self.raman_pumps = [
            NS(
                power=q["power"],
                frequency=q["frequency"],
                propagation_direction=q["propagation_direction"],
            )
            for q in gnpy_pumps(pumps)
        ]
        self._cr = coupling(freqs)
        self._index = {f: i for i, f in enumerate(freqs)}

    def cr(self, f):
        i = [self._index[x] for x in f]
        return self._cr[np.ix_(i, i)]

    def alpha(self, f):
        return np.full(len(f), 0.2 / (10 * np.log10(np.e)) / 1e3)


def test_power_profiles_and_spontaneous_noise_of_multi_pump_spans():
    """A co-pump and two counter-pumps on 40 channels over 80 km."""
    length, nch = 80.0, 40
    rel = (np.arange(nch) - nch / 2) * 100e9
    pumps = [
        {"frequency_thz": 205.0, "power_mw": 150, "direction": "co"},
        {"wavelength_nm": 1425, "power_mw": 300, "direction": "counter"},
        {"wavelength_nm": 1455, "power_mw": 250, "direction": "counter"},
    ]
    fiber = comnum.FiberParams(span_km=length, fc_hz=FC)
    for p_dbm, step_m in [(0.0, 20.0), (0.0, 5.0), (-30.0, 5.0)]:
        ours = comnum.raman_profile(fiber, list(rel), p_dbm, {"pumps": pumps, "step_km": 0.05})
        freqs = ours["freqs_hz"]

        def compute(p_dbm=p_dbm, step_m=step_m, freqs=freqs):
            from gnpy.core.parameters import SimParams
            from gnpy.core.science_utils import RamanSolver

            span = Span(length, freqs, pumps)
            SimParams.set_params(
                {
                    "raman_params": {
                        "flag": True,
                        "method": "numerical",
                        "result_spatial_resolution": 1e3,
                        "solver_spatial_resolution": step_m,
                    }
                }
            )
            si = NS(
                pch=np.full(nch, 1e-3 * 10 ** (p_dbm / 10)),
                frequency=freqs[:nch],
                baud_rate=np.ones(nch),
                number_of_channels=nch,
            )
            srs = RamanSolver.calculate_stimulated_raman_scattering(si, span)
            out = {
                "net_gain_db": 10 * np.log10(srs.loss_profile[:nch, -1]),
                "pumps_dbm": 10 * np.log10(srs.power_profile[nch:][:, [0, -1]]) + 30,
            }
            if p_dbm < -20:
                out["ase_psd"] = (
                    RamanSolver.calculate_spontaneous_raman_scattering(si, srs, span)
                    * srs.loss_profile[:nch, -1]
                )
            return out

        ref = oracle(f"raman_profile_{p_dbm}dbm_{step_m}m", compute, tool=TOOL)
        err = np.abs(ours["net_gain_db"] - ref["net_gain_db"]).max()
        # Euler converges at first order: 0.016 dB at 20 m, 0.004 dB at 5 m
        assert err < 1e-3 * step_m, (step_m, err)
        pumps_ours = ours["power_dbm"][nch:][:, [0, -1]]
        assert np.abs(pumps_ours - ref["pumps_dbm"]).max() < 1e-3 * step_m
        if p_dbm < -20:
            np.testing.assert_allclose(ours["ase_psd"], ref["ase_psd"], rtol=2e-3)


# The attenuation of a low-water-peak standard fibre against wavelength (nm, dB/km), a
# representative table and not a measured fibre: from the pumps of the E band to the
# end of the L band.
ATTENUATION_NM_DB_KM = [
    (1340, 0.33),
    (1383, 0.31),
    (1420, 0.27),
    (1460, 0.24),
    (1500, 0.21),
    (1550, 0.19),
    (1580, 0.19),
    (1625, 0.20),
]


def attenuation_spectrum():
    """The table above in increasing frequency (THz, dB/km)."""
    points = sorted((C0 / (w * 1e-9) / 1e12, a) for w, a in ATTENUATION_NM_DB_KM)
    return [f for f, _ in points], [a for _, a in points]


SCL_STEPS_M = {"coarse": 0.5, "fine": 0.25}
"""The two Euler steps of GNPy whose results are extrapolated to a zero step."""


def test_power_profiles_of_an_s_c_l_span_with_frequency_dependent_loss_and_area():
    """161 channels from 1490 to 1620 nm, a co-pump and four counter-pumps over 80 km,
    with the attenuation of the table above and the effective area of GNPy's
    step-index scaling (80 µm² at 1550 nm), against GNPy's own fibre: its attenuation
    interpolated in frequency, its effective areas and their overlap, its gain spectrum
    and solver. Only the depletion of the higher frequency is that of comnum (GNPy's
    negative coefficients multiplied by ``f_k/f_j``, the photon-number conservation)."""
    length, spacing = 80.0, 100e9
    abs_ch = np.arange(185.2e12, 201.2e12 + 1.0, spacing)
    nch = len(abs_ch)
    pumps = [
        {"wavelength_nm": 1350, "power_mw": 120, "direction": "co"},
        {"wavelength_nm": 1380, "power_mw": 250, "direction": "counter"},
        {"wavelength_nm": 1410, "power_mw": 200, "direction": "counter"},
        {"wavelength_nm": 1440, "power_mw": 150, "direction": "counter"},
        {"wavelength_nm": 1470, "power_mw": 120, "direction": "counter"},
    ]
    freqs = np.concatenate([abs_ch, [pump_frequency(p) for p in pumps]])
    f_thz, alpha_db_km = attenuation_spectrum()
    fiber = comnum.FiberParams(span_km=length, fc_hz=FC)
    for p_dbm in [0.0, -20.0]:

        def compute(p_dbm=p_dbm):
            from gnpy.core.elements import Fiber, RamanFiber
            from gnpy.core.parameters import SimParams
            from gnpy.core.science_utils import RamanSolver

            params = {
                "length": length,
                "length_units": "km",
                "loss_coef": {"value": alpha_db_km, "frequency": [f * 1e12 for f in f_thz]},
                "dispersion": 16.7e-6,
                "effective_area": 80e-12,
                "pmd_coef": 0.0,
                "con_in": 0,
                "con_out": 0,
                "att_in": 0,
                "ref_frequency": C0 / 1550e-9,
            }
            span = RamanFiber(
                uid="span",
                params=params,
                operational={"temperature": 300, "raman_pumps": gnpy_pumps(pumps)},
            )

            def photon_conserving(f):
                cr = Fiber.cr(span, f)
                return np.where(cr < 0, cr * np.outer(f, 1 / f), cr)

            span.cr = photon_conserving
            out = {
                "aeff_um2": span.params.effective_area_scaling(freqs) * 1e12,
                "alpha_db_km": span.alpha(freqs) * 1e3 * 10 * np.log10(np.e),
            }
            for name, step_m in SCL_STEPS_M.items():
                SimParams.set_params(
                    {
                        "raman_params": {
                            "flag": True,
                            "method": "numerical",
                            "result_spatial_resolution": 100.0,
                            "solver_spatial_resolution": step_m,
                        }
                    }
                )
                si = NS(
                    pch=np.full(nch, 1e-3 * 10 ** (p_dbm / 10)),
                    frequency=abs_ch,
                    baud_rate=np.ones(nch),
                    number_of_channels=nch,
                )
                srs = RamanSolver.calculate_stimulated_raman_scattering(si, span)
                ase = RamanSolver.calculate_spontaneous_raman_scattering(si, srs, span)
                gain = srs.loss_profile[:nch, -1]
                out[f"net_gain_db_{name}"] = 10 * np.log10(gain)
                out[f"pumps_dbm_{name}"] = 10 * np.log10(srs.power_profile[nch:][:, [0, -1]]) + 30
                out[f"ase_db_{name}"] = 10 * np.log10(ase * gain)
            return out

        ref = oracle(f"raman_profile_scl_{p_dbm}dbm", compute, tool=TOOL)
        raman = {
            "pumps": pumps,
            "step_km": 0.0125,
            "alpha_spectrum": {"frequency_thz": f_thz, "alpha_db_km": alpha_db_km},
            "aeff_spectrum": {
                "frequency_thz": list(np.sort(freqs) / 1e12),
                "aeff_um2": list(np.asarray(ref["aeff_um2"])[np.argsort(freqs)]),
            },
        }
        ours = comnum.raman_profile(fiber, list(abs_ch - FC), p_dbm, raman)
        np.testing.assert_allclose(ours["freqs_hz"], freqs, rtol=1e-12)
        # the attenuation that GNPy interpolates is that of comnum
        loss = -(ours["net_gain_db"] - ours["on_off_gain_db"]) / length
        np.testing.assert_allclose(loss, ref["alpha_db_km"][:nch], rtol=1e-9)

        # GNPy's Euler steps converge at first order: the Richardson extrapolation
        # 2 x(h/2) - x(h) of its runs at 0.5 and 0.25 m removes the term in h
        def extrapolated(key, ref=ref):
            return 2 * np.asarray(ref[f"{key}_fine"]) - np.asarray(ref[f"{key}_coarse"])

        err = np.abs(ours["net_gain_db"] - extrapolated("net_gain_db")).max()
        pumps_err = np.abs(ours["power_dbm"][nch:][:, [0, -1]] - extrapolated("pumps_dbm")).max()
        ase_err = np.abs(10 * np.log10(ours["ase_psd"]) - extrapolated("ase_db")).max()
        raw_err = np.abs(ours["net_gain_db"] - ref["net_gain_db_fine"]).max()
        print(
            f"{p_dbm} dBm: gains {err:.1e} dB (GNPy at 0.25 m: {raw_err:.1e}), "
            f"pumps {pumps_err:.1e} dB, ASE {ase_err:.1e} dB"
        )
        # measured (0 and -20 dBm): 4.9e-7 and 1.0e-6 dB on the gains, 5.0e-7 and 8.1e-7 dB
        # on the pumps, 1.0e-5 and 9.5e-6 dB on the spontaneous noise (GNPy integrates it on its 100 m output grid); GNPy alone at
        # 0.25 m is 2.8e-4 dB away on the gains and 8.1e-4 dB on the pumps
        assert err < 2e-6, (p_dbm, err)
        assert pumps_err < 2e-6, (p_dbm, pumps_err)
        assert ase_err < 3e-5, (p_dbm, ase_err)


@pytest.mark.parametrize("with_pumps", [False, True])
def test_generalized_gn_model_with_counter_pumps(with_pumps):
    """The NLI of the centre of three channels, GNPy with a fine frequency grid."""
    nch, rs, spacing, roll, length = 3, 32e9, 50e9, 0.02, 80.0
    rel = (np.arange(nch) - nch // 2) * spacing
    pumps = [
        {"wavelength_nm": 1425, "power_mw": 300, "direction": "counter"},
        {"wavelength_nm": 1455, "power_mw": 250, "direction": "counter"},
    ]
    raman = {"pumps": pumps if with_pumps else []}
    fiber = comnum.FiberParams(span_km=length, d_ps_nm_km=16.7, gamma_w_km=1.3, n_spans=1, fc_hz=FC)
    ours = (
        comnum.gn_model(fiber, list(rel), 0.0, rs, rolloff=roll, coherent=False, raman=raman)[
            "nli_psd"
        ]
        * rs
    )

    def compute():
        from gnpy.core.elements import Fiber, RamanFiber
        from gnpy.core.info import create_arbitrary_spectral_information
        from gnpy.core.parameters import SimParams
        from gnpy.core.science_utils import NliSolver, RamanSolver

        SimParams.set_params(
            {
                "raman_params": {
                    "flag": True,
                    "method": "numerical",
                    "result_spatial_resolution": 50,
                    "solver_spatial_resolution": 5,
                },
                "nli_params": {
                    "method": "ggn_spectrally_separated",
                    "computed_channels": [nch // 2 + 1],
                    "dispersion_tolerance": 0.5,
                    "phase_shift_tolerance": 0.02,
                },
            }
        )
        params = {
            "length": length,
            "length_units": "km",
            "loss_coef": 0.2,
            "dispersion": 16.7e-6,
            "gamma": 1.3e-3,
            "pmd_coef": 0.0,
            "con_in": 0,
            "con_out": 0,
            "att_in": 0,
            "ref_frequency": FC,
        }
        si = create_arbitrary_spectral_information(
            FC + rel,
            slot_width=spacing,
            pch=1e-3,
            baud_rate=rs,
            roll_off=roll,
            tx_osnr=40,
            tx_power=1e-3,
        )
        if with_pumps:
            f = RamanFiber(
                uid="span",
                params=params,
                operational={"temperature": 300, "raman_pumps": gnpy_pumps(pumps)},
            )
        else:
            f = Fiber(uid="span", params=params)
        f.cr = coupling
        srs = RamanSolver.calculate_stimulated_raman_scattering(si, f)
        return float(NliSolver.compute_nli(si, srs, f)[nch // 2])

    theirs = oracle(f"ggn_nli_{'counter_pumps' if with_pumps else 'no_pumps'}", compute, tool=TOOL)
    assert abs(10 * np.log10(ours / theirs)) < 0.03
