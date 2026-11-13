"""
The split-step Fourier method, checked against exact solutions
==============================================================

Light in a single-mode fibre obeys the nonlinear Schrödinger equation (NLSE),

.. math::

   \\frac{\\partial A}{\\partial z} = -\\frac{\\alpha}{2} A
   - j\\frac{\\beta_2}{2}\\frac{\\partial^2 A}{\\partial t^2} + j\\gamma|A|^2 A,

with the attenuation :math:`\\alpha`, the group-velocity dispersion :math:`\\beta_2` and
the Kerr coefficient :math:`\\gamma` (G. P. Agrawal, *Nonlinear Fiber Optics*, Ch. 2).
The linear part is diagonal in frequency, the nonlinear part is diagonal in time, and
neither has a closed-form solution together: the split-step Fourier method advances
the field by small steps :math:`h`, each a half linear step in the frequency domain, a
nonlinear phase rotation :math:`e^{j\\gamma|A|^2 h}` in the time domain, and another
half linear step (the *symmetric* split step, Agrawal Sec. 2.4). :class:`comnum.Ssfm`
is that solver, used by the link simulations of the other examples.

This tutorial checks it piece by piece against the cases where the NLSE has an exact
solution: dispersion alone (a Gaussian pulse broadens by a known factor), the Kerr
effect alone (self-phase modulation is a known phase), both together (the solitons),
and then measures how the error of the method falls with the step. Nothing here is
noise: every curve is deterministic and compared with a formula.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, MUTED

C = 299_792_458.0
FC = 193.1e12
T0 = 10e-12  # the width of the pulses (s)
BETA2 = -20e-24  # s²/km, anomalous dispersion
GAMMA = 1.3  # 1/(W km)


def fiber(
    beta2_s2_km: float, gamma_w_km: float, span_km: float, alpha_db_km: float = 0.0
) -> comnum.FiberParams:
    """The fibre of a span, its dispersion given as beta2 (the library takes D)."""
    lam = C / FC
    d_ps_nm_km = -2 * np.pi * C / lam**2 * beta2_s2_km * 1e3
    f = comnum.FiberParams(
        alpha_db_km=alpha_db_km,
        d_ps_nm_km=d_ps_nm_km,
        gamma_w_km=gamma_w_km,
        span_km=span_km,
        n_spans=1,
        fc_hz=FC,
    )
    assert abs(f.beta2_s2_km - beta2_s2_km) <= 1e-12 * abs(beta2_s2_km)
    return f


def propagate(a, fs, beta2, gamma, length_km, step_km, alpha_db_km=0.0, nonlinear_step="midpoint"):
    """The field after `length_km` of fibre, without amplifier (the sampling check of
    the link simulations is off: these pulses are not a modulated signal)."""
    link = comnum.Ssfm(
        fiber(beta2, gamma, length_km, alpha_db_km),
        step_km=step_km,
        amplification="none",
        nonlinear_step=nonlinear_step,
        check_sampling=False,
    )
    return link.propagate(a, fs)


# the time grid of the pulses: 2048 samples at 2 THz, a window of 1 ns around t = 0
N, FS = 2048, 2e12
t = (np.arange(N) - N // 2) / FS
L_D = T0**2 / abs(BETA2)  # the dispersion length, 5 km

# %%
# Dispersion alone: a Gaussian pulse broadens as :math:`\sqrt{1 + (z/L_D)^2}`
# ---------------------------------------------------------------------------
# Without the Kerr effect the linear step is exact, whatever its length: the simulated
# pulse and the closed form (Agrawal, eq. (3.2.10)) agree to the arithmetic. The
# dispersion length :math:`L_D = T_0^2/|\beta_2|` is 5 km for a 10 ps pulse.
p0 = 1e-3
gaussian = np.sqrt(p0) * np.exp(-(t**2) / (2 * T0**2)) + 0j
fig, ax = plt.subplots()
ax.plot(t * 1e12, np.abs(gaussian) ** 2 * 1e3, color=GREY, linewidth=1.2, label="$z$ = 0")
errors = []
for z in [1, 2, 4]:
    a = propagate(gaussian, FS, BETA2, 0.0, z * L_D, step_km=z * L_D)
    broadening = np.sqrt(1 + z**2)
    exact = p0 / broadening * np.exp(-(t**2) / (T0**2 * broadening**2))
    errors.append(np.sum((np.abs(a) ** 2 - exact) ** 2) / np.sum(exact**2))
    (line,) = ax.plot(t * 1e12, np.abs(a) ** 2 * 1e3, linewidth=1.4, label=f"$z$ = {z} $L_D$")
    ax.plot(
        t[::40] * 1e12,
        exact[::40] * 1e3,
        "o",
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        markersize=4,
    )
ax.plot([], [], "o", color=MUTED, markeredgecolor=BACKGROUND, markersize=4, label="closed form")
ax.set(
    xlim=(-100, 100),
    xlabel="time (ps)",
    ylabel="power (mW)",
    title="A Gaussian pulse in a dispersive fibre, no Kerr effect",
)
ax.legend()
fig.tight_layout()
print(
    "dispersion alone, NMSE of the power against the closed form:",
    ", ".join(f"{e:.1e}" for e in errors),
)

# %%
# The Kerr effect alone: self-phase modulation
# --------------------------------------------
# Without dispersion the nonlinear step is exact too: the field keeps its envelope and
# gains the phase :math:`\gamma|A|^2 L_{\mathrm{eff}}` (Agrawal, eq. (4.1.7)). That phase
# follows the pulse, so the instantaneous frequency sweeps across it and the spectrum
# broadens into lobes: about :math:`\phi_{\max}/\pi + 1/2` peaks for a maximum phase
# :math:`\phi_{\max} = \gamma P_0 L` (Agrawal, Sec. 4.1.2, Fig. 4.2).
length = 10.0
f = np.fft.fftshift(np.fft.fftfreq(N, 1 / FS))
fig, ax = plt.subplots()
errors = []
for k, phi in enumerate([0.5, 1.5, 2.5, 3.5]):
    p0 = phi * np.pi / (GAMMA * length)
    a0 = np.sqrt(p0) * np.exp(-(t**2) / (2 * T0**2)) + 0j
    a = propagate(a0, FS, 0.0, GAMMA, length, step_km=length)
    exact = a0 * np.exp(1j * GAMMA * np.abs(a0) ** 2 * length)
    errors.append(np.sum(np.abs(a - exact) ** 2) / np.sum(np.abs(exact) ** 2))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(a))) ** 2
    ax.plot(
        f / 1e9,
        spectrum / spectrum.max() + 1.1 * k,
        linewidth=1.4,
        label=rf"$\phi_{{\max}}$ = {phi}$\pi$",
    )
ax.set(
    xlim=(-150, 150),
    yticks=[],
    xlabel="frequency (GHz)",
    ylabel="power spectrum, offset per curve",
    title="Self-phase modulation of a Gaussian pulse, no dispersion",
)
ax.legend(loc="upper right")
fig.tight_layout()
print(
    "Kerr effect alone, NMSE of the field against the closed form:",
    ", ".join(f"{e:.1e}" for e in errors),
)

# %%
# Both together: solitons
# -----------------------
# When the chirps of the anomalous dispersion and of the Kerr effect cancel, the pulse
# keeps its shape: the fundamental soliton :math:`\sqrt{P_0}\,\mathrm{sech}(t/T_0)` with
# :math:`P_0 = |\beta_2|/(\gamma T_0^2)` (Agrawal, Sec. 5.2). With four times that
# power, the second-order soliton breathes: it narrows, splits, and recovers its
# initial shape after the soliton period :math:`z_0 = \frac{\pi}{2} L_D`. Neither is a
# small perturbation of the other: the simulation has to balance both terms, and it
# does with steps of 10 m (the titles give the error on the power after one period;
# the field also gains a constant phase, :math:`z/(2L_D)` for the fundamental soliton).
p1 = abs(BETA2) / (GAMMA * T0**2)
z0 = np.pi / 2 * L_D
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, order in zip(axes, [1, 2], strict=True):
    a0 = np.sqrt(order**2 * p1) / np.cosh(t / T0) + 0j
    ax.plot(t * 1e12, np.abs(a0) ** 2 / p1, color=GREY, linewidth=1.2, label="$z$ = 0")
    a = a0
    for frac in [0.25, 0.5, 0.75, 1.0]:
        a = propagate(a, FS, BETA2, GAMMA, 0.25 * z0, step_km=0.01)
        ax.plot(t * 1e12, np.abs(a) ** 2 / p1, linewidth=1.4, label=f"$z$ = {frac:g} $z_0$")
    # the power, not the field: a soliton gains a constant phase along the fibre
    nmse = np.sum((np.abs(a) ** 2 - np.abs(a0) ** 2) ** 2) / np.sum(np.abs(a0) ** 4)
    ax.set(
        xlim=(-40, 40),
        xlabel="time (ps)",
        title=f"$N$ = {order}: power NMSE {nmse:.1e} after one period",
    )
    ax.legend(fontsize="small")
axes[0].set(ylabel="power / $P_0$")
fig.tight_layout()

# %%
# The step: a second-order method
# -------------------------------
# The symmetric split step commits an error of order :math:`h^3` per step, hence
# :math:`h^2` over a fixed length (Agrawal, Sec. 2.4.1). Measured on the second-order
# soliton over one period against a reference at 1 m steps, the error falls by a
# factor 100 per decade of step.
reference = propagate(np.sqrt(4 * p1) / np.cosh(t / T0) + 0j, FS, BETA2, GAMMA, z0, step_km=0.001)
steps = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
nmse = []
for h in steps:
    a = propagate(np.sqrt(4 * p1) / np.cosh(t / T0) + 0j, FS, BETA2, GAMMA, z0, step_km=h)
    nmse.append(np.sum(np.abs(a - reference) ** 2) / np.sum(np.abs(reference) ** 2))
nmse = np.array(nmse)

fig, ax = plt.subplots()
ax.loglog(steps * 1e3, nmse, "o-", markeredgecolor=BACKGROUND, label="symmetric split step")
ax.loglog(
    steps * 1e3,
    nmse[-1] * (steps / steps[-1]) ** 4,
    ":",
    color=GREY,
    label=r"$\propto h^4$ (a field error $\propto h^2$)",
)
ax.set(
    xlabel="step (m)",
    ylabel="NMSE against the 1 m reference",
    title="Second-order soliton over one period",
)
ax.legend()
fig.tight_layout()
slope = np.polyfit(np.log(steps), np.log(nmse), 1)[0]
print(f"the NMSE grows as h^{slope:.2f}: the field error is of order h^{slope / 2:.2f}")

# %%
# With attenuation: where the nonlinear phase is taken
# ----------------------------------------------------
# In a lossy fibre the power falls within the step. The default step applies the phase
# of the power at the middle of the step; Optilux applies the phase of the power at the
# start over the effective length :math:`(1 - e^{-\alpha h})/\alpha` of the step. Both
# are second order and close to each other: on a 32 GBd 16-QAM signal at 5 dBm over
# 80 km of standard fibre (0.2 dB/km), a 1 km step leaves an NMSE below :math:`10^{-9}`
# with either, and the two curves are the same at coarse steps.
SPS, SYMBOLS = 4, 2048
rng = comnum.Rng(9)
c16 = comnum.Constellation("qam", 16).normalized()
_, s = comnum.symbols(c16, SYMBOLS, rng)
pulse = comnum.pulse_shape("rrc", SPS, n_taps=1025, rolloff=0.1)
x = comnum.pnorm(comnum.fir_filter(pulse, comnum.upsample(s, SPS))) * np.sqrt(comnum.dbm2w(5.0))
fs_link = 32e9 * SPS
link = dict(
    beta2=fiber(BETA2, GAMMA, 80.0).beta2_s2_km, gamma=GAMMA, length_km=80.0, alpha_db_km=0.2
)
reference = propagate(x, fs_link, step_km=0.01, **link)
steps = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.2])
fig, ax = plt.subplots()
for technique, label in [
    ("midpoint", "power at the middle of the step (default)"),
    ("effective_length", "effective length of the step (Optilux)"),
]:
    err = [
        np.sum(
            np.abs(propagate(x, fs_link, step_km=h, nonlinear_step=technique, **link) - reference)
            ** 2
        )
        / np.sum(np.abs(reference) ** 2)
        for h in steps
    ]
    ax.loglog(steps, err, "o-", markeredgecolor=BACKGROUND, label=label)
    print(f"{label}: NMSE {err[3]:.1e} at 1 km, {err[0]:.1e} at 10 km")
ax.set(
    xlabel="step (km)",
    ylabel="NMSE against the 10 m reference",
    title="16-QAM at 5 dBm over 80 km of lossy fibre",
)
ax.legend()
fig.tight_layout()

# %%
# What to remember
# ----------------
# - the linear and the nonlinear steps are each exact alone: the error of the method is
#   the error of splitting them, of order :math:`h^2` on the field;
# - the step must resolve the shortest of the dispersion and nonlinear lengths: the
#   second-order soliton, which has both at 5 km, needs steps of 50 m for an NMSE of
#   :math:`10^{-6}`; a 32 GBd signal at 5 dBm in standard fibre needs 1 km;
# - the time window must hold the broadened pulse and the sampling rate the broadened
#   spectrum: :class:`comnum.Ssfm` checks both on a modulated signal
#   (``check_sampling``), and the link examples rely on it;
# - the next examples use the same solver on links: the launch-power sweep and the
#   back-propagation, the GN model of WDM interference, and the cost of each
#   split-step technique.
