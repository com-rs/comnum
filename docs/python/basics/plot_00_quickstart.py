"""
Quickstart: a whole link in twenty lines
========================================

A digital link carries bits from one place to another through a medium that damages
them. This chapter builds the smallest link that does it, end to end, and runs it;
the next chapters open each of its lines.

.. code-block:: text

   text ──► bits ──► symbols ──► channel ──► decisions ──► bits ──► text
            source    mapper      noise       detector

* The **source** turns the message into bits.
* The **mapper** groups the bits and sends each group as one *symbol*, a complex
  number: its real and imaginary parts are the amplitudes of two carriers in
  quadrature.
* The **channel** adds noise.
* The **detector** decides which symbol was sent from what it received, and gives the
  bits of its decision.

Whatever the link (a telephone, a satellite, an optical fibre), these four steps are
there. What changes is the mapper, the channel, and how much work the detector has
to do before it can decide.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, SERIES

# %%
# The link
# --------
# Every function that draws random numbers takes a generator, :class:`comnum.Rng`:
# there is no hidden global state, so the script gives the same numbers every time it
# runs. The modulation is QPSK, four points, two bits per symbol (the square
# constellation of four points, ``"qam", 4``).
message = "Digital communications: getting bits through noise."
rng = comnum.Rng(2026)
qpsk = comnum.Constellation("qam", 4).normalized()

bits = comnum.text_to_bits(message)  # the source
symbols = qpsk.map_bits(bits)  # the mapper
received = comnum.awgn_ebn0(symbols, 4.0, qpsk.bits_per_symbol, rng)  # the channel
decided = qpsk.demap_bits(received)  # the detector

print(f"sent:     {message!r}")
print(f"received: {comnum.bits_to_text(decided)!r}")
print(f"{bits.size} bits, {symbols.size} symbols, {np.count_nonzero(decided != bits)} bit errors")

# %%
# At this noise level a few bits are wrong, and each wrong bit damages one character
# (``\x..`` is a byte that is no longer a printable character, ``�`` one that is no
# longer valid text).
# The number that sets the noise level is :math:`E_b/N_0`, the energy spent per bit
# over the power density of the noise, here 4 dB: chapter 2 defines it.
#
# What the receiver sees
# ----------------------
# The four points sent, and what arrives: a cloud around each of them. The detector
# decides for the nearest point, so a received symbol that has crossed an axis is an
# error.
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.plot(received.real, received.imag, ".", color=SERIES[0], markersize=4, alpha=0.6)
ax.plot(qpsk.points.real, qpsk.points.imag, "o", color=SERIES[1], markeredgecolor=BACKGROUND)
ax.axhline(0, color=GREY, linewidth=0.8)
ax.axvline(0, color=GREY, linewidth=0.8)
ax.set(aspect="equal", xlabel="in phase", ylabel="quadrature", title="QPSK at $E_b/N_0$ = 4 dB")
fig.tight_layout()

# %%
# More power, fewer errors
# ------------------------
# The same message at four noise levels. Each 3 dB doubles the energy per bit.
for ebn0_db in (0.0, 3.0, 6.0, 9.0):
    noisy = comnum.awgn_ebn0(symbols, ebn0_db, 2, comnum.Rng(7))
    print(f"{ebn0_db:4.0f} dB: {comnum.bits_to_text(qpsk.demap_bits(noisy))!r}")

# %%
# How good is the link?
# ---------------------
# The figure of merit is the **bit error rate** (BER): the fraction of bits decided
# wrongly. A message of 400 bits cannot measure it, so the simulation sends random
# bits instead, many of them, and counts. For QPSK over this channel the answer is
# known in closed form,
#
# .. math::
#
#    P_b = Q\!\left(\sqrt{2E_b/N_0}\right), \qquad
#    Q(x) = \frac{1}{\sqrt{2\pi}}\int_x^\infty e^{-t^2/2}\,dt,
#
# the tail of a Gaussian (chapter 3 derives it), and :func:`comnum.theory_ber`
# evaluates it. A simulation that does not land on that curve has a mistake in it:
# comparing with the theory is how every chapter ends.
ebn0_points = np.arange(0.0, 9.0, 1.0)
measured = []
for ebn0_db in ebn0_points:
    tx = comnum.bits(400_000, rng)
    rx = qpsk.demap_bits(comnum.awgn_ebn0(qpsk.map_bits(tx), ebn0_db, 2, rng))
    measured.append(comnum.ber(rx, tx))

ebn0_fine = np.linspace(0, 10, 101)
fig, ax = plt.subplots()
ax.semilogy(ebn0_fine, comnum.theory_ber(4, ebn0_fine, "qam"), color=GREY, label="theory")
ax.plot(ebn0_points, measured, "o", color=SERIES[0], markeredgecolor=BACKGROUND, label="simulated")
ax.set(ylim=(1e-5, 0.2), xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate", title="QPSK over AWGN")
ax.legend()
fig.tight_layout()

gap = np.max(np.abs(np.array(measured) / comnum.theory_ber(4, ebn0_points, "qam") - 1))
print(f"simulated within {100 * gap:.0f} % of the closed form")

# %%
# The curve falls fast: from 0 to 8 dB the energy per bit is multiplied by six and the
# error rate divided by four hundred. That steepness is the reason digital links work
# at all, and the reason a decibel matters to the people who design them.
#
# Next
# ----
# * :doc:`plot_01_bits_and_symbols`: what a constellation is and how to choose one.
# * :doc:`plot_02_the_noisy_channel`: what :math:`E_b/N_0` means and how the noise is drawn.
# * :doc:`plot_03_decisions_and_error_rates`: where the formula comes from, and how
#   many errors to count.
