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
use simpler shapes still: the constant slope of the ISRS GN model (0.028 1/(W·km·THz),
Semrau, Killey & Bayvel 2018, ``slope_w_km_thz`` of the mapping), good to 10 THz, and
the triangle to the peak (Zirngibl 1998), a three-point table. Relative to the peak, the rms errors are
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

AEFF_UM2 = 80.0

# %%
# The measured spectrum and the models, as gain efficiencies at 80 µm². The measured
# table is that of GNPy (standard single-mode fibre, 90 points to 42 THz), for a pump at
# ``reference_hz``; every model is scaled to its peak. The two straight lines are the
# assumptions of the closed forms: the constant slope of the ISRS GN model (0.028
# 1/(W·km·THz), Semrau, Killey & Bayvel 2018) to 14 THz, ``slope_w_km_thz`` of the
# mapping, and the triangle that rises to the peak and falls to zero at 15 THz (Zirngibl
# 1998), a three-point table.
offsets = np.arange(0.0, 42.01, 0.1) * 1e12
measured = comnum.raman_gain(offsets)
f_ref_thz, peak = measured["reference_hz"] / 1e12, measured["peak_m_w"]
models = {
    "Blow & Wood 1989: one oscillator": ({"model": "blow_wood"}, ORANGE, "-"),
    "Lin & Agrawal 2006: + boson peak": ({"model": "lin_agrawal"}, GREEN, "-"),
    "Hollenbeck & Cantrell 2002: 13 modes": ({"model": "hollenbeck_cantrell"}, BLUE, "-"),
    "constant slope 0.028 /(W·km·THz)": ({"slope_w_km_thz": 0.028}, GREY, "--"),
    "triangle to the peak": (
        {"offsets_thz": [0.0, 13.25, 15.0], "gain_m_w": [0.0, peak, 0.0]},
        GREY,
        ":",
    ),
}
for mapping, _, _ in models.values():
    mapping["reference_thz"] = f_ref_thz
spectra = {label: comnum.raman_gain(offsets, m) for label, (m, _, _) in models.items()}

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(
    offsets / 1e12,
    measured["efficiency_w_km"],
    color=INK,
    linewidth=2.2,
    label="measured, standard single-mode fibre",
)
for label, (_, color, style) in models.items():
    ax.plot(offsets / 1e12, spectra[label]["efficiency_w_km"], style, color=color, label=label)
ax.set(xlabel="frequency offset pump − signal (THz)", ylabel="gain efficiency $C_R$ (1/(W·km)) at 80 µm²",
       xlim=(0, 42), ylim=(0, 0.5), title="The Raman gain of silica and its models")  # fmt: skip
ax.legend(fontsize="small")
fig.tight_layout()

# %%
# How far each model is from the measurement: the rms and the largest error relative
# to the peak, over the band of the inter-channel Raman scattering (0–10 THz) and over
# the whole spectrum.
low = offsets <= 10e12
print(
    f"{'model':36s} {'rms 0-10':>9s} {'max 0-10':>9s} {'rms 0-42':>9s} {'max 0-42':>9s}  (relative to the peak)"
)
for label, s in spectra.items():
    e = (s["gain_m_w"] - measured["gain_m_w"]) / peak
    rms, biggest = (lambda x: (np.sqrt(np.mean(x**2)), np.abs(x).max())), None
    print(
        f"{label:36s} {rms(e[low])[0]:9.3f} {rms(e[low])[1]:9.3f} {rms(e)[0]:9.3f} {rms(e)[1]:9.3f}"
    )


def compare(fiber, freqs, power_dbm, pumps, key, reference_label, metric, title, xlabel, ylabel):
    """The quantity ``key`` of :func:`comnum.raman_profile` with the measured spectrum and
    with every model, drawn against ``freqs``; the reference is labelled by
    ``reference_label(curve)``, every model by ``metric(curve, reference)``."""
    raman = {"pumps": pumps, "step_km": 0.5}
    ref = comnum.raman_profile(fiber, list(freqs), power_dbm, raman)[key]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = (fiber.fc_hz + np.asarray(freqs)) / 1e12
    ax.plot(x, ref, color=INK, linewidth=2.2, label=f"measured spectrum: {reference_label(ref)}")
    for label, (mapping, color, style) in models.items():
        curve = comnum.raman_profile(fiber, list(freqs), power_dbm, {**raman, "gain": mapping})[key]
        ax.plot(x, curve, style, color=color, label=f"{label}: {metric(curve, ref)}")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize="small")
    fig.tight_layout()


# %%
# What the choice changes in an amplifier: the on-off gain of a 100 km span counter-pumped
# at 1450 nm (500 mW) over 40 channels of the C band. The signals sit 11 to 15 THz below
# the pump, on the peak and on its steep edge: every model that has the peak gives the
# amplifier within half a decibel over most of the band, and they diverge on the fall at
# 15 THz; the straight lines do not have the peak region at all.
compare(
    comnum.FiberParams(span_km=100, fc_hz=193.5e12),
    comnum.wdm_grid(40, 100e9),
    -3.0,
    [{"wavelength_nm": 1450, "power_mw": 500, "direction": "counter"}],
    "on_off_gain_db",
    lambda curve, ref: (
        f"{np.abs(curve - ref).max():.2f} dB off" if curve is not ref else f"{ref.mean():.1f} dB"
    ),
    "Counter-pump 1450 nm, 500 mW, 100 km, 40 × 100 GHz at −3 dBm",
    "frequency (THz)",
    "on-off gain (dB)",
)

# %%
# What the choice changes without pump: the tilt that the channels of a C+L comb give
# each other (inter-channel stimulated Raman scattering). 100 channels at 100 GHz and
# 0 dBm, 10 THz wide: the offsets are 0 to 10 THz, where the models differ most. A single
# oscillator has a quarter of the measured gain at 3 THz and underestimates the tilt;
# the boson peak of Lin & Agrawal restores it.
compare(
    comnum.FiberParams(span_km=100, fc_hz=190.0e12),
    comnum.wdm_grid(100, 100e9),
    0.0,
    [],
    "net_gain_db",
    lambda ref: f"tilt {np.ptp(ref):.2f} dB",
    lambda curve, ref: f"tilt {np.ptp(curve):.2f} dB",
    "ISRS tilt of 100 × 0 dBm at 100 GHz, no pump",
    "frequency (THz)",
    "net gain after 100 km (dB)",
)
