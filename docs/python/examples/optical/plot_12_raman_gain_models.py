r"""
The Raman gain of silica: the measurement and five ways to model it
===================================================================

The stimulated Raman scattering that pumps a distributed amplifier, and that tilts a wide
WDM comb even without pump, is set by one function of the fibre: the Raman gain
spectrum :math:`g_R(\Delta f)` of silica, the gain that a wave receives from a wave
:math:`\Delta f` above it. This page shows the measured spectrum that the simulators use
by default, the five ways the ``gain`` mapping of ``raman`` can model it, how far each is
from the measurement, and what the choice changes in two situations: a pumped span and
an unpumped C+L comb. :func:`comnum.raman_gain` returns the spectrum of a mapping;
:func:`comnum.raman_profile` solves the power equations with it.

**The measurement.** The default is the table of standard single-mode fibre of GNPy: 90
points from 0 to 42 THz for a pump at 1454 nm, a peak of :math:`0.42` 1/(W·km) at 80 µm²
near 13 THz, a shoulder from 2 to 8 THz (the boson peak of the glass), a steep fall at
15 THz, then the weaker bands at 18, 24 and 32 THz.

**The models.** The gain spectrum is the imaginary part of the Fourier transform of the
Raman response function :math:`h_R(t)` (Agrawal, *Nonlinear Fiber Optics*, Sec. 2.3.2);
a model gives :math:`h_R(t)` in closed form and the library computes the transform, scaled
to the peak of the measurement. Blow & Wood (1989) use one damped oscillator,
:math:`\tau_1 = 12.2` fs and :math:`\tau_2 = 32` fs: a Lorentzian with the right peak, a
quarter of the measured gain at 3 THz and a long tail above 15 THz. Lin & Agrawal (2006)
add the boson peak, :math:`f_b = 0.21` and :math:`\tau_b = 96` fs: the shoulder is
recovered, the tail stays. Hollenbeck & Cantrell (2002) sum thirteen vibrational modes
of the Si–O–Si structure, each with Gaussian and Lorentzian widths: the whole spectrum,
including the fall at 15 THz and the bands beyond. The closed forms of the literature
use simpler shapes still, here written as tables of the mapping: the constant slope of
the ISRS GN model (0.028 1/(W·km·THz), Semrau, Killey & Bayvel 2018), good to 10 THz,
and the triangle to the peak (Zirngibl 1998). Relative to the peak, the rms errors are
9.5, 2.7, 3.2, 4.1 and 5.3 % over 0–10 THz and 13.3, 13.1, 3.8, 18.3 and 15.5 % over the
whole spectrum.

**A pumped span.** A counter-pump of 500 mW at 1450 nm over 100 km gives 17.6 dB of
on-off gain to 40 channels of the C band, which sit 11 to 15 THz below the pump: on the
peak and on its steep edge. The three response models agree with the measurement within
0.5 dB over most of the band, but differ by up to 3.4 dB at 1565 nm, on the fall at
15 THz where their shapes diverge; the straight lines miss the long-wavelength half by
14 to 18 dB. The edge of the band of a Raman amplifier is where the model of the gain
matters.

**An unpumped C+L comb.** 100 channels of 0 dBm at 100 GHz, 10 THz wide, feed each other
over 100 km: a tilt of 2.56 dB with the measured spectrum. The offsets are 0 to 10 THz,
the band of the shoulder: the single oscillator gives 1.91 dB, the boson peak brings Lin
& Agrawal to 2.39 dB and Hollenbeck & Cantrell to 2.37 dB, the slope of the closed form
2.47 dB, the triangle 2.77 dB. What the inter-channel Raman scattering needs is the
shoulder, not the peak.

The models are checked against the Lorentzian of Blow & Wood in closed form and against
the table of GNPy in the unit tests of ``comnum_channel::optical::raman::gain``;
``tests/oracle/test_gnpy_oracle.py`` checks the profiles computed with the table against
GNPy.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BLUE, GREEN, GREY, INK, ORANGE

F_REF_THZ = 206.18  # the pump of the measurement of standard single-mode fibre, 1454 nm
AEFF_UM2 = 80.0

# %%
# The measured spectrum and the models, as gain efficiencies at 80 µm². The measured
# table is that of GNPy (standard single-mode fibre, 90 points to 42 THz); every model is
# scaled to its peak. The two straight lines are the assumptions of the closed forms,
# written as tables of the ``gain`` mapping: the constant slope of the ISRS GN model
# (0.028 1/(W·km·THz), Semrau, Killey & Bayvel 2018) to 14 THz, and the triangle that
# rises to the peak and falls to zero at 15 THz (Zirngibl 1998).
offsets = np.arange(0.0, 42.01, 0.1) * 1e12
measured = comnum.raman_gain(offsets)
peak = measured["peak_m_w"]
slope_m_w = 0.028 * AEFF_UM2 * 1e-12 / 1e3  # 1/(W·km·THz) at 80 µm² -> m/W per THz
models = {
    "blow_wood": (
        "Blow & Wood 1989: one oscillator",
        {"model": "blow_wood", "reference_thz": F_REF_THZ},
        ORANGE,
        "-",
    ),
    "lin_agrawal": (
        "Lin & Agrawal 2006: + boson peak",
        {"model": "lin_agrawal", "reference_thz": F_REF_THZ},
        GREEN,
        "-",
    ),
    "hollenbeck_cantrell": (
        "Hollenbeck & Cantrell 2002: 13 modes",
        {"model": "hollenbeck_cantrell", "reference_thz": F_REF_THZ},
        BLUE,
        "-",
    ),
    "slope": (
        "constant slope 0.028 /(W·km·THz)",
        {
            "offsets_thz": [0.0, 14.0],
            "gain_m_w": [0.0, 14.0 * slope_m_w],
            "reference_thz": F_REF_THZ,
        },
        GREY,
        "--",
    ),
    "triangle": (
        "triangle to the peak",
        {
            "offsets_thz": [0.0, 13.25, 15.0],
            "gain_m_w": [0.0, peak, 0.0],
            "reference_thz": F_REF_THZ,
        },
        GREY,
        ":",
    ),
}
spectra = {k: comnum.raman_gain(offsets, m[1]) for k, m in models.items()}

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(
    offsets / 1e12,
    measured["efficiency_w_km"],
    color=INK,
    linewidth=2.2,
    label="measured, standard single-mode fibre",
)
for k, (label, _, color, style) in models.items():
    ax.plot(offsets / 1e12, spectra[k]["efficiency_w_km"], style, color=color, label=label)
ax.set(xlabel="frequency offset pump − signal (THz)", ylabel="gain efficiency $C_R$ (1/(W·km)) at 80 µm²",
       xlim=(0, 42), ylim=(0, 0.5), title="The Raman gain of silica and its models")  # fmt: skip
ax.legend(fontsize="small")
fig.tight_layout()

# %%
# How far each model is from the measurement: the rms and the largest error relative
# to the peak, over the band of the inter-channel Raman scattering (0–10 THz) and over
# the whole spectrum.
rows = []
for k, (label, _, _, _) in models.items():
    e = (spectra[k]["gain_m_w"] - measured["gain_m_w"]) / peak
    low = offsets <= 10e12
    rows.append(
        (
            label,
            np.sqrt(np.mean(e[low] ** 2)),
            np.abs(e[low]).max(),
            np.sqrt(np.mean(e**2)),
            np.abs(e).max(),
        )
    )
print(
    f"{'model':36s} {'rms 0-10':>9s} {'max 0-10':>9s} {'rms 0-42':>9s} {'max 0-42':>9s}  (relative to the peak)"
)
for label, r10, m10, r42, m42 in rows:
    print(f"{label:36s} {r10:9.3f} {m10:9.3f} {r42:9.3f} {m42:9.3f}")

# %%
# What the choice changes in an amplifier: the on-off gain of a 100 km span counter-pumped
# at 1450 nm (500 mW) over 40 channels of the C band. The signals sit 12 to 16 THz below
# the pump, on the peak: every model that has the peak gives the amplifier within a few
# tenths of a decibel, the straight lines do not.
c_band = comnum.FiberParams(span_km=100, fc_hz=193.5e12)
rel = list(np.arange(-20, 20) * 100e9 + 50e9)
wavelengths = 299_792_458.0 / (193.5e12 + np.array(rel)) * 1e9
pump = [{"wavelength_nm": 1450, "power_mw": 500, "direction": "counter"}]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
errors = {}
ref = comnum.raman_profile(c_band, rel, -3.0, {"pumps": pump, "step_km": 0.5})["on_off_gain_db"]
ax.plot(wavelengths, ref, color=INK, linewidth=2.2, label=f"measured spectrum: {ref.mean():.1f} dB")
for k, (label, gain, color, style) in models.items():
    p = comnum.raman_profile(c_band, rel, -3.0, {"pumps": pump, "step_km": 0.5, "gain": gain})
    errors[k] = np.abs(p["on_off_gain_db"] - ref).max()
    ax.plot(
        wavelengths,
        p["on_off_gain_db"],
        style,
        color=color,
        label=f"{label}: {errors[k]:.2f} dB off",
    )
ax.set(xlabel="wavelength (nm)", ylabel="on-off gain (dB)",
       title="Counter-pump 1450 nm, 500 mW, 100 km, 40 × 100 GHz at −3 dBm")  # fmt: skip
ax.legend(fontsize="small", loc="lower left")
fig.tight_layout()

# %%
# What the choice changes without pump: the tilt that the channels of a C+L comb give
# each other (inter-channel stimulated Raman scattering). 100 channels at 100 GHz and
# 0 dBm, 10 THz wide: the offsets are 0 to 10 THz, where the models differ most. A single
# oscillator has a quarter of the measured gain at 3 THz and underestimates the tilt;
# the boson peak of Lin & Agrawal restores it.
cl_band = comnum.FiberParams(span_km=100, fc_hz=190.0e12)
rel_cl = list((np.arange(100) - 49.5) * 100e9)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ref = comnum.raman_profile(cl_band, rel_cl, 0.0, {"pumps": [], "step_km": 0.5})["net_gain_db"]
tilts = {"measured": np.ptp(ref)}
ax.plot(
    (190e12 + np.array(rel_cl)) / 1e12,
    ref,
    color=INK,
    linewidth=2.2,
    label=f"measured spectrum: tilt {tilts['measured']:.2f} dB",
)
for k, (label, gain, color, style) in models.items():
    p = comnum.raman_profile(cl_band, rel_cl, 0.0, {"pumps": [], "step_km": 0.5, "gain": gain})
    tilts[k] = np.ptp(p["net_gain_db"])
    ax.plot(
        (190e12 + np.array(rel_cl)) / 1e12,
        p["net_gain_db"],
        style,
        color=color,
        label=f"{label}: tilt {tilts[k]:.2f} dB",
    )
ax.set(xlabel="frequency (THz)", ylabel="net gain after 100 km (dB)",
       title="ISRS tilt of 100 × 0 dBm at 100 GHz, no pump")  # fmt: skip
ax.legend(fontsize="small")
fig.tight_layout()
print("on-off gain, largest error (dB):", {k: round(float(v), 2) for k, v in errors.items()})
print("ISRS tilt (dB):", {k: round(float(v), 2) for k, v in tilts.items()})
