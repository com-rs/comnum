"""
The noisy channel
=================

Every receiver adds noise: the thermal agitation of the electrons in its first
amplifier, to begin with. The model of that noise is the simplest channel of all and
the reference against which every link is measured, the **additive white Gaussian
noise** channel (AWGN):

.. math::

   y = x + w, \\qquad w \\sim \\mathcal{CN}(0, N_0).

Each word of the name is an assumption:

* **additive**: the noise is added to the signal, and does not depend on it;
* **white**: its samples are independent of one another (its spectrum is flat);
* **Gaussian**: it is the sum of very many small contributions, so the central limit
  theorem applies. For a complex sample, the real and imaginary parts are independent
  Gaussians of variance :math:`N_0/2` each, so that :math:`\\mathbb{E}|w|^2 = N_0`.

This chapter draws that noise, checks that it is what the model says, and sorts out
the three signal-to-noise ratios that the literature uses for the same thing.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, SERIES

# %%
# Drawing the noise
# -----------------
# ``Rng.complex_noise(n, n0)`` draws :math:`n` samples of :math:`\mathcal{CN}(0, N_0)`.
# Its power is :math:`N_0`, split evenly between the two parts, which are
# uncorrelated.
rng = comnum.Rng(3)
n0 = 0.5
w = rng.complex_noise(200_000, n0)
print(f"power           {comnum.power(w):.4f}   (N0 = {n0})")
print(f"real part       {np.var(w.real):.4f}   (N0/2 = {n0 / 2})")
print(f"imaginary part  {np.var(w.imag):.4f}")
print(f"correlation     {np.mean(w.real * w.imag):+.4f}")

# %%
# The histogram of the real part against the Gaussian density of variance
# :math:`N_0/2`, and the samples one after the other: no memory, each one is drawn
# afresh.
fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.6))
grid = np.linspace(-2.2, 2.2, 201)
left.hist(w.real, bins=80, density=True, color=SERIES[0], alpha=0.5, label="samples")
left.plot(grid, np.exp(-(grid**2) / n0) / np.sqrt(np.pi * n0), color=GREY, label="Gaussian")
left.set(xlabel="real part", ylabel="density", title="Gaussian")
left.legend()
lags = np.arange(-20, 21)
correlation = [np.mean(w[20:-20] * np.conj(np.roll(w, lag)[20:-20])).real / n0 for lag in lags]
right.stem(lags, correlation, basefmt=" ")
right.set(xlabel="lag (samples)", ylabel="normalised correlation", title="White")
fig.tight_layout()

# %%
# Three signal-to-noise ratios
# ----------------------------
# What matters is the size of the signal against the noise, and there are three ways
# to say it. For a link at one sample per symbol they are:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Ratio
#      - Definition
#      - Used for
#    * - :math:`\mathrm{SNR}`
#      - signal power over noise power, in the band of the signal
#      - what an instrument measures
#    * - :math:`E_s/N_0`
#      - energy per symbol over noise density
#      - the detector: it decides symbol by symbol
#    * - :math:`E_b/N_0`
#      - energy per bit over noise density
#      - comparing modulations and codes fairly
#
# With :math:`k` bits per symbol, :math:`E_s = kE_b`, so
#
# .. math::
#
#    \frac{E_s}{N_0} = k\,\frac{E_b}{N_0}, \qquad
#    \left.\frac{E_s}{N_0}\right|_{\mathrm{dB}} =
#    \left.\frac{E_b}{N_0}\right|_{\mathrm{dB}} + 10\log_{10}k ,
#
# and at one sample per symbol :math:`\mathrm{SNR} = E_s/N_0`. The reason for
# :math:`E_b/N_0` is the fairness: 16-QAM sends four bits in a symbol, so at the same
# :math:`E_s/N_0` as BPSK it spends a quarter of the energy on each bit. Plotting
# against :math:`E_b/N_0` charges every modulation the same price per bit delivered.
# (A channel code, which adds redundant bits, is charged the same way: chapter 7.)
#
# Decibels, :math:`10\log_{10}` of a power ratio, turn the products into sums: 3 dB is
# a factor 2, 10 dB a factor 10.
for kind, m in [("psk", 2), ("psk", 4), ("qam", 16), ("qam", 64)]:
    k = comnum.Constellation(kind, m).bits_per_symbol
    print(f"{m:>3}-{kind.upper()}: Eb/N0 = 6 dB  ->  Es/N0 = {6 + comnum.lin2db(k):5.2f} dB")

# %%
# Setting the noise for an Eb/N0
# ------------------------------
# With the constellation at :math:`E_s = 1`, the noise variance for a wanted
# :math:`E_b/N_0` follows from the definitions:
#
# .. math::
#
#    N_0 = \frac{E_s}{k \cdot E_b/N_0} = \frac{1}{k \cdot 10^{(E_b/N_0)_{\mathrm{dB}}/10}} .
#
# :func:`comnum.awgn_ebn0` does exactly that and adds the noise: it is the function to
# use, here and in every chapter that follows. The line written by hand below is only
# there to show what the call does; the measured :math:`E_s/N_0` of both confirms it.
qam16 = comnum.Constellation("qam", 16).normalized()
k = qam16.bits_per_symbol
ebn0_db = 8.0
x = qam16.map_bits(comnum.bits(k * 100_000, rng))

n0 = 1.0 / (k * comnum.db2lin(ebn0_db))
by_hand = x + rng.complex_noise(x.size, n0)
by_call = comnum.awgn_ebn0(x, ebn0_db, k, rng)

for name, y in [("by hand", by_hand), ("awgn_ebn0", by_call)]:
    measured = 1.0 / comnum.evm(y, qam16, tx=x)  # the EVM is noise power over signal power
    print(f"{name:>10}: Es/N0 = {comnum.lin2db(measured):.2f} dB")
print(f"  expected: Es/N0 = {ebn0_db + comnum.lin2db(k):.2f} dB")

# %%
# What it looks like
# ------------------
# 16-QAM at three values of :math:`E_b/N_0`. Each cloud is a two-dimensional Gaussian
# centred on a point of the constellation, of standard deviation
# :math:`\sqrt{N_0/2}` on each axis. Errors begin when the clouds touch.
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, e in zip(axes, (4.0, 10.0, 16.0), strict=True):
    y = comnum.awgn_ebn0(x[:6000], e, k, rng)
    ax.plot(y.real, y.imag, ".", color=SERIES[0], markersize=2, alpha=0.5)
    ax.plot(
        qam16.points.real, qam16.points.imag, "o", color=SERIES[1], markersize=4,
        markeredgecolor=BACKGROUND,
    )  # fmt: skip
    ax.set(aspect="equal", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), title=f"$E_b/N_0$ = {e:.0f} dB")
fig.tight_layout()

# %%
# The same seed, the same noise
# -----------------------------
# The generator is the only source of randomness. Two generators with the same seed
# give the same noise, sample for sample, so a result can be reproduced exactly, on
# another machine or a year later. Different seeds give independent noise.
a = comnum.Rng(42).complex_noise(3, 1.0)
b = comnum.Rng(42).complex_noise(3, 1.0)
c = comnum.Rng(43).complex_noise(3, 1.0)
print("seed 42:", np.round(a, 3))
print("seed 42:", np.round(b, 3))
print("seed 43:", np.round(c, 3))
assert np.array_equal(a, b)

# %%
# Summary
# -------
# * The AWGN channel adds independent complex Gaussian samples of power :math:`N_0`.
# * :math:`E_s/N_0 = k\,E_b/N_0`; error rates are compared against :math:`E_b/N_0`.
# * For a unit-energy constellation, :math:`N_0 = 1/(k\,E_b/N_0)`.
# * A seed fixes the noise: a simulation is an experiment that can be repeated.
#
# **Try it.** A link runs 64-QAM at :math:`E_b/N_0 = 12` dB. What is :math:`N_0`, and
# the standard deviation of the noise on each axis? Check your answer against the
# spacing of the points, :math:`d_{\min} = 0.309`.
#
# **Reference.** J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., Secs.
# 2.9 and 4.1.
#
# Next: :doc:`plot_03_decisions_and_error_rates`.
