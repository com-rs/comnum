"""
Direct detection: sensitivity, quantum limit and power penalties
================================================================

The textbook treatment of an on–off keying receiver (G. P. Agrawal, *Fiber-Optic
Communication Systems*, 3rd ed., Wiley, 2002, Secs. 4.4–4.6 and 5.4), as the closed forms
of ``comnum.DirectDetectionReceiver`` and its companions. The receiver decides between
the photocurrents :math:`I_1` and :math:`I_0` with Gaussian noises :math:`\\sigma_1`,
:math:`\\sigma_0`; its Q factor and bit error rate are (eqs. (4.5.10), (4.5.11))

.. math::

    Q = \\frac{I_1 - I_0}{\\sigma_1 + \\sigma_0}, \\qquad
    \\mathrm{BER} = \\frac{1}{2}\\operatorname{erfc}\\!\\left(\\frac{Q}{\\sqrt{2}}\\right),

:math:`Q = 6` for :math:`10^{-9}`. With the shot noise :math:`2 q M^2 F_A R P \\Delta f`
of an avalanche photodiode of gain :math:`M` and excess noise :math:`F_A = k_A M + (1 -
k_A)(2 - 1/M)`, and the thermal noise :math:`\\sigma_T^2 = 4 k_B T F_n \\Delta f / R_L` of
its load, the average power that gives :math:`Q` is the sensitivity (eq. (4.5.17))

.. math::

    \\bar{P}_{\\mathrm{rec}} = \\frac{Q}{R}\\left(q F_A \\Delta f\\, Q + \\frac{\\sigma_T}{M}\\right),

far above the quantum limit of 10 photons per bit. The penalties of a finite extinction
ratio, of the intensity noise of the source, of the timing jitter and of dispersive
broadening are the extra power that restores :math:`Q`. The numbers the text reads on
its figures are the validation cards ``agrawal2002_*``; the formulas are also checked
against a simulated photodiode (``comnum.Photodiode``).
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, BLUE, GREEN, GREY, MUTED, ORANGE

Q9 = comnum.qfuncinv(1e-9)  # 5.998
C0 = 299_792_458.0

# %%
# Receiver sensitivity against the bit rate
# -----------------------------------------
#
# A p-i-n receiver (1 A/W, 50 Ω at 300 K, 3 dB of noise figure, :math:`\Delta f = B/2`)
# is thermal-noise limited: its sensitivity grows as :math:`\sqrt{B}`, 5 dB per decade.
# An InGaAs avalanche photodiode (:math:`k_A = 0.5`) at its optimum gain gains
# 9 to 16 dB on it (its gain 87 at 0.1 Gb/s down to 15 at 100 Gb/s); the quantum limit,
# 10 photons per bit at 1550 nm, grows as :math:`B`, 10 dB per decade, and stays 16 to
# 23 dB below the APD.

rates = np.logspace(8, 11, 61)
pin = comnum.DirectDetectionReceiver(responsivity=1.0)
apd = comnum.DirectDetectionReceiver(responsivity=1.0, ionization_ratio=0.5)
sens = {"p-i-n": [], "APD, optimum gain": [], "quantum limit": []}
gains = []
for b in rates:
    pin.bandwidth_hz = apd.bandwidth_hz = b / 2
    sens["p-i-n"].append(pin.sensitivity_dbm(Q9))
    apd.apd_gain = apd.optimum_apd_gain(Q9)
    gains.append(apd.apd_gain)
    sens["APD, optimum gain"].append(apd.sensitivity_dbm(Q9))
    sens["quantum limit"].append(comnum.w2dbm(comnum.quantum_limit_w(1e-9, b, C0 / 1550e-9)))

fig, ax = plt.subplots()
for (name, p), color in zip(sens.items(), [BLUE, ORANGE, GREY], strict=True):
    ax.semilogx(rates / 1e9, p, color=color, label=name)
ax.set(xlabel="bit rate (Gb/s)", ylabel="sensitivity at BER = 1e-9 (dBm)",
       title="Receiver sensitivity of on–off keying")  # fmt: skip
ax.legend()
fig.tight_layout()
print(f"optimum APD gain: {gains[0]:.1f} at 0.1 Gb/s, {gains[-1]:.1f} at 100 Gb/s")

# %%
# The bit error rate of a simulated photodiode
# --------------------------------------------
#
# NRZ on–off keying at 10 Gb/s through ``comnum.Photodiode`` (shot and thermal noise,
# four samples per bit integrated over the bit, so that the noise bandwidth is
# :math:`\Delta f = B/2`), decided at the optimum threshold, against the BER of the Q
# factor at every average power. The shot noise of the 1 bits makes the two noises
# unequal; the closed form carries both.

bit_rate, sps, n_bits = 10e9, 4, 100_000
fs = bit_rate * sps
pd = comnum.Photodiode(responsivity=0.8, temperature_k=300.0, dark_current_a=0.0, load_ohm=50.0,
                       bandwidth_hz=bit_rate / 2, bandwidth_limitation=False)  # fmt: skip
rx = comnum.DirectDetectionReceiver(responsivity=0.8, temperature_k=300.0, load_ohm=50.0,
                                    noise_figure_db=0.0, bandwidth_hz=bit_rate / 2)  # fmt: skip
rng = comnum.Rng(3)
bits = comnum.bits(n_bits, rng)
powers_dbm = np.arange(-26.0, -17.9, 1.0)
simulated, theory = [], []
for p_dbm in powers_dbm:
    p1 = 2 * comnum.dbm2w(p_dbm)
    i = pd.detect_power(np.repeat(bits * p1, sps), fs, rng).reshape(-1, sps).mean(axis=1)
    s1 = np.sqrt(rx.shot_variance(p1) + rx.thermal_variance())
    s0 = np.sqrt(rx.thermal_variance())
    threshold = s0 * rx.responsivity * p1 / (s0 + s1)
    simulated.append(np.mean((i > threshold) != bits.astype(bool)))
    theory.append(rx.ber(p1, 0.0))

fig, ax = plt.subplots()
ax.semilogy(powers_dbm, theory, color=MUTED, label=r"$\frac{1}{2}\,\mathrm{erfc}(Q/\sqrt{2})$")
ax.semilogy(powers_dbm, simulated, "o", color=BLUE, markeredgecolor=BACKGROUND,
            label="comnum.Photodiode, 1e5 bits")  # fmt: skip
ax.axvline(rx.sensitivity_dbm(Q9), color=GREY, linestyle=":", linewidth=1.0)
ax.annotate("sensitivity at 1e-9", (rx.sensitivity_dbm(Q9), 3e-2), fontsize=8, color=MUTED,
            rotation=90, ha="right")  # fmt: skip
ax.set(xlabel="average received power (dBm)", ylabel="BER", ylim=(1e-5, 1),
       title="10 Gb/s NRZ on a p-i-n receiver")  # fmt: skip
ax.legend()
fig.tight_layout()

# %%
# Power penalties
# ---------------
#
# Left (Figs. 4.20–4.22 of the book): a thermal-noise-limited receiver pays :math:`10
# \log_{10}[(1 + r_{ex})/(1 - r_{ex})]` for the power left in the 0 bits, 1 dB at
# :math:`r_{ex} = 0.12`; the intensity noise of the source, which grows with the signal,
# costs :math:`-10 \log_{10}(1 - r_I^2 Q^2)`, 2 dB at :math:`r_I = 0.1`, and no power
# restores :math:`Q = 6` beyond :math:`r_I = 1/6`; a Gaussian timing jitter of r.m.s.
# :math:`\tau_j` costs 2 dB at :math:`B \tau_j = 0.16` and everything beyond 0.2. Right
# (Figs. 5.7 and 5.11): a Gaussian pulse whose output width fills the bit, :math:`4 B
# \sigma = 1`, pays :math:`10 \log_{10}(\sigma/\sigma_0)`: for a source of spectral
# width :math:`\sigma_\lambda` the penalty is :math:`-5 \log_{10}[1 - (4 B L |D|
# \sigma_\lambda)^2]`, 2.2 dB at :math:`B L D \sigma_\lambda = 0.2`; for a linearly
# chirped pulse of a narrow-linewidth source it is :math:`5 \log_{10}[(1 + 8 C \beta_2
# B^2 L)^2 + (8 \beta_2 B^2 L)^2]`, more than 8 dB at :math:`|\beta_2| B^2 L = 0.2` for
# the :math:`C = -6` of a directly modulated laser, and negative at first for :math:`C
# > 0` (:math:`\beta_2 < 0`), where the chirp compresses the pulse.

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8))
r_ex = np.linspace(0.0, 0.5, 201)
left.plot(
    r_ex, comnum.extinction_ratio_penalty_db(r_ex), color=BLUE, label="extinction ratio $r_{ex}$"
)
r_i = np.linspace(0.0, 0.16, 201)
left.plot(r_i, [comnum.intensity_noise_penalty_db(r, Q9) for r in r_i], color=ORANGE,
          label="intensity noise $r_I$ at $Q = 6$")  # fmt: skip
jitter = np.linspace(0.0, 0.2, 201)
left.plot(jitter, [comnum.timing_jitter_penalty_db(j, Q9) for j in jitter], color=GREEN,
          label=r"timing jitter $B\tau_j$ at $Q = 6$")  # fmt: skip
left.set(xlabel=r"$r_{ex} = P_0/P_1$, $r_I$, or $B\tau_j$", ylabel="power penalty (dB)",
         ylim=(0, 5), title="Extinction ratio, intensity noise, jitter")  # fmt: skip
left.legend(fontsize=8)

x = np.linspace(0.0, 0.24, 121)
right.plot(x, [comnum.dispersion_penalty_db(1e9, 100.0, 1.0, 10 * v) for v in x], color=BLUE,
           label=r"source width: $BLD\sigma_\lambda$")  # fmt: skip
beta2 = -20e-24  # s²/km
for chirp, color in [(0.0, GREEN), (-6.0, ORANGE), (2.0, GREY)]:
    right.plot(x, [comnum.chirp_penalty_db(chirp, beta2, 10e9, v / 2.0) for v in x], color=color,
               linestyle="--", label=rf"chirped pulse, $|\beta_2| B^2 L$, $C = {chirp:+.0f}$")  # fmt: skip
right.set(xlabel=r"$BLD\sigma_\lambda$, or $|\beta_2| B^2 L$", ylabel="power penalty (dB)",
          ylim=(-3, 8), title="Dispersive broadening and chirp")  # fmt: skip
right.axhline(0.0, color=GREY, linewidth=0.5)
right.legend(fontsize=7)
fig.tight_layout()
print(
    f"dispersion penalty at BLDσ = 0.2: {comnum.dispersion_penalty_db(1e9, 100.0, 1.0, 2.0):.2f} dB"
)
print(
    f"chirp penalty at |β₂|B²L = 0.2, C = -6: {comnum.chirp_penalty_db(-6.0, beta2, 10e9, 100.0):.1f} dB"
)
print(f"dispersion limit of 10 Gb/s over standard fibre: "
      f"{comnum.dispersion_limited_length_km(10e9, -21.7e-24):.1f} km")  # fmt: skip
