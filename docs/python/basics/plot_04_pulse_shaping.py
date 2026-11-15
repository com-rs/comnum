"""
Pulse shaping and the matched filter
====================================

So far a symbol was a number. On a cable or an antenna it has to become a signal that
lasts for a while: each symbol :math:`a_k` is carried by a **pulse** :math:`p(t)`,
and the transmitter sends the sum of the pulses, one every symbol period :math:`T`,

.. math::

   x(t) = \\sum_k a_k\\,p(t - kT).

The choice of :math:`p` settles two things at once: how much bandwidth the link
occupies, and whether the symbols can be told apart at the receiver. This chapter
shows why the obvious pulse is a poor one, what makes a pulse a good one (Nyquist's
criterion), and why the receiver should filter with the same pulse before it samples
(the matched filter).

In the simulation, time is sampled: ``SPS`` samples per symbol. The transmitter
inserts ``SPS - 1`` zeros between the symbols (:func:`comnum.upsample`) and filters
with the samples of the pulse (:func:`comnum.fir_filter`).
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, MUTED, SERIES

SPS = 8  # samples per symbol
ROLLOFF = 0.35
SPAN = 32  # length of the pulses, in symbols

rng = comnum.Rng(4)
qpsk = comnum.Constellation("qam", 4).normalized()


def shape(symbols: np.ndarray, pulse: np.ndarray) -> np.ndarray:
    """The signal that carries the symbols on the pulse, SPS samples per symbol."""
    return comnum.fir_filter(pulse, comnum.upsample(symbols, SPS))


# %%
# The rectangular pulse and its spectrum
# --------------------------------------
# The obvious pulse holds the symbol for one period: a rectangle. Its spectrum is a
# cardinal sine, whose side lobes fall as :math:`1/f` only: 13 dB down for the first,
# and still 20 dB down three symbol rates away. A neighbour on the next channel would
# not thank us. The raised-cosine family trades a longer pulse for a compact spectrum:
# the root-raised cosine below is confined to :math:`\pm(1 + \alpha)/(2T)`, with the
# **roll-off** :math:`\alpha` between 0 and 1.
# :func:`comnum.pulse_shape` gives the samples of a pulse; scaled to a unit energy, a
# symbol keeps its energy through the transmit filter and the matched filter
rect = comnum.pulse_shape("rect", SPS)
rect = rect / np.linalg.norm(rect)
rrc = comnum.pulse_shape("rrc", SPS, n_taps=SPAN * SPS + 1, rolloff=ROLLOFF)
rrc = rrc / np.linalg.norm(rrc)
_, symbols = comnum.symbols(qpsk, 20_000, rng)

fig, ax = plt.subplots()
for pulse, name, color in [
    (rect, "rectangular", SERIES[1]),
    (rrc, "root-raised cosine", SERIES[0]),
]:
    f, p = comnum.psd(shape(symbols, pulse), fs=SPS, nfft=1024)
    order = np.argsort(f)
    ax.plot(f[order], 10 * np.log10(p[order] / p.max()), color=color, label=name)
for edge in (-(1 + ROLLOFF) / 2, (1 + ROLLOFF) / 2):
    ax.axvline(edge, color=GREY, linewidth=0.8, linestyle=":")
ax.set(
    ylim=(-70, 3),
    xlim=(-SPS / 2, SPS / 2),
    xlabel="frequency (symbol rates)",
    ylabel="power spectral density (dB)",
    title=f"Spectrum of QPSK, roll-off {ROLLOFF}",
)
ax.legend()
fig.tight_layout()

# %%
# Nyquist's criterion
# -------------------
# A band-limited pulse is long: it overlaps its neighbours. That does no harm if, at
# the instant the receiver samples symbol :math:`k`, every other pulse is zero:
#
# .. math::
#
#    g(nT) = \begin{cases} 1 & n = 0 \\ 0 & n \neq 0 \end{cases}
#
# for the overall pulse :math:`g` seen by the sampler. This is **Nyquist's criterion**
# for no intersymbol interference (ISI). The smallest bandwidth that allows it is
# :math:`1/(2T)` on each side, with a cardinal sine; the raised cosine satisfies it
# with :math:`(1 + \alpha)/(2T)` and a pulse that decays fast enough to be truncated
# and to tolerate a small timing error.
#
# The raised cosine below crosses zero at every symbol instant but its own. The
# root-raised cosine does not: it is only *half* of the pulse.
rc = comnum.pulse_shape("rc", SPS, n_taps=SPAN * SPS + 1, rolloff=ROLLOFF)
t = (np.arange(rc.size) - rc.size // 2) / SPS
at_symbols = np.abs(t - np.round(t)) < 1e-9

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(t, rc / rc.max(), color=SERIES[0], label="raised cosine")
ax.plot(t, rrc / rrc.max(), color=SERIES[1], linewidth=1.0, label="root-raised cosine")
ax.plot(
    t[at_symbols], (rc / rc.max())[at_symbols], "o", color=SERIES[0], markeredgecolor=BACKGROUND
)
ax.axhline(0, color=GREY, linewidth=0.8)
ax.set(xlim=(-5, 5), xlabel="time (symbols)", ylabel="amplitude", title="Pulses in time")
ax.legend()
fig.tight_layout()

worst = np.max(np.abs((rc / rc.max())[at_symbols & (np.abs(t) > 0.5)]))
print(f"raised cosine at the other symbol instants: at most {worst:.1e}")

# %%
# The matched filter
# ------------------
# The receiver sees the signal plus white noise and must reduce each symbol to one
# number. Any filter followed by a sampler does that; which filter is best? The one
# that maximises the signal-to-noise ratio at the sampling instant is the **matched
# filter**, whose impulse response is the pulse itself, reversed in time and
# conjugated: :math:`p^*(-t)`. It weighs each instant by how much signal is expected
# there. Its output SNR is :math:`E_s/N_0`, the best any receiver can do, and it does
# not depend on the shape of the pulse, only on its energy.
#
# So the overall pulse is :math:`g = p * p^*(-\cdot)`, and it is :math:`g` that must
# satisfy Nyquist's criterion. Hence the root-raised cosine: its spectrum is the
# square root of the raised cosine's, and two of them in cascade, one at each end of
# the link, give a raised cosine. The transmitter gets its compact spectrum, the
# receiver its matched filter, and the samples are free of ISI.
#
# The **eye diagram** overlays the signal symbol after symbol. After the transmit
# filter alone the traces never meet: at the symbol instants each sample still carries
# a little of its neighbours. After the matched filter they all pass through
# :math:`\pm 0.707` at the sampling instant, and nowhere else.
transmitted = shape(symbols[:2000], rrc)
filtered = comnum.fir_filter(rrc, transmitted)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
comnum.plot.eye(comnum.pnorm(transmitted), ax=axes[0], sps=SPS, traces=200, skip=SPAN * SPS)
axes[0].set_title("after the transmit filter")
comnum.plot.eye(filtered, ax=axes[1], sps=SPS, traces=200, skip=SPAN * SPS)
axes[1].set_title("after the matched filter")
fig.tight_layout()

sampled = comnum.downsample(filtered, SPS)
residual = np.max(np.abs(sampled[SPAN:-SPAN] - symbols[SPAN : 2000 - SPAN]))
print(f"samples after the matched filter: the symbols, to within {residual:.3f}")

# %%
# The small residue is the truncation of the pulses to 32 symbols. It matters more
# than it looks: a residue of 1 % of the amplitude would move the error rate by 10 % at
# 8 dB, where the curve is steep.
#
# The link with pulses
# --------------------
# :func:`comnum.awgn_ebn0` adds the noise to the shaped signal (its argument ``sps``
# tells it that a symbol now lasts ``SPS`` samples), the receiver filters and takes one
# sample per symbol. With unit-energy pulses the noise
# after the matched filter still has the variance :math:`N_0` and the symbols their
# energy: the error rate is that of chapter 3, as if the pulses were not there.
#
# The second receiver skips the filter and simply samples a rectangular pulse once per
# symbol. It collects one sample of signal out of eight and all of the noise: it loses
# :math:`10\log_{10}8 = 9` dB, the curve of the theory shifted to the right.
MIN_ERRORS = 1000  # 3 % of standard error per point


def ber_with_pulses(pulse: np.ndarray, ebn0_db: float, *, matched: bool) -> float:
    """The bit error rate of QPSK on a pulse, with or without the matched filter."""
    errors = sent = 0
    while errors < MIN_ERRORS and sent < 8_000_000:
        tx = comnum.bits(2 * 20_000, rng)
        signal = shape(qpsk.map_bits(tx), pulse)
        noisy = comnum.awgn_ebn0(signal, ebn0_db, 2, rng, sps=SPS)
        if matched:
            samples = comnum.downsample(comnum.fir_filter(pulse[::-1], noisy), SPS)
        else:
            samples = comnum.downsample(noisy, SPS, offset=SPS // 2) * np.sqrt(SPS)
        keep = slice(2 * SPAN, -2 * SPAN)  # the ends, where the filters are still filling
        rx = qpsk.demap_bits(samples).reshape(-1, 2)[SPAN:-SPAN].ravel()
        errors += round(comnum.ber(rx, tx[keep]) * rx.size)
        sent += rx.size
    return errors / sent


points = np.arange(0.0, 9.0, 2.0)
fine = np.linspace(0, 20, 201)
with_filter = np.array([ber_with_pulses(rrc, e, matched=True) for e in points])
late_points = np.arange(9.0, 18.0, 2.0)
without = np.array([ber_with_pulses(rect, e, matched=False) for e in late_points])

fig, ax = plt.subplots()
ax.semilogy(fine, comnum.theory_ber(4, fine, "qam"), color=GREY, label="theory")
ax.semilogy(
    fine, comnum.theory_ber(4, fine - 10 * np.log10(SPS), "qam"), color=MUTED, linestyle=":",
    label="theory, 9 dB to the right",
)  # fmt: skip
ax.plot(
    points,
    with_filter,
    "o",
    color=SERIES[0],
    markeredgecolor=BACKGROUND,
    label="root-raised cosine, matched filter",
)
ax.plot(
    late_points,
    without,
    "s",
    color=SERIES[1],
    markeredgecolor=BACKGROUND,
    label="rectangle, one sample, no filter",
)
ax.set(ylim=(1e-5, 0.3), xlim=(0, 18), xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate")
ax.legend(fontsize="small")
fig.tight_layout()

gap = np.max(np.abs(with_filter / comnum.theory_ber(4, points, "qam") - 1))
print(f"with the matched filter: within {100 * gap:.0f} % of the closed form")
shifted = comnum.theory_ber(4, late_points - 10 * np.log10(SPS), "qam")
print(
    f"without it: within {100 * np.max(np.abs(without / shifted - 1)):.0f} % of the curve 9 dB away"
)

# %%
# How much bandwidth?
# -------------------
# The signal occupies :math:`(1 + \alpha)/T`: a symbol rate of 1 MBd with a roll-off of
# 0.35 takes 1.35 MHz. A small roll-off saves spectrum and costs a longer pulse, a
# larger peak power and a tighter timing. The **spectral efficiency** of the link is
# :math:`k/(1 + \alpha)` bits per second per hertz: 1.48 for this QPSK, 2.96 for
# 16-QAM on the same pulse.

# %%
# Summary
# -------
# * A symbol is sent on a pulse; the pulse sets the spectrum.
# * No ISI if the overall pulse is zero at the other sampling instants (Nyquist).
# * The matched filter gives the best SNR at the sampler, :math:`E_s/N_0`.
# * Root-raised cosine at both ends: compact spectrum, matched filter and no ISI
#   together. Bandwidth :math:`(1 + \alpha)/T`.
#
# **Try it.** Set ``ROLLOFF`` to 0.05 and look at the pulse in time and at the eye:
# what happens to the horizontal opening of the eye, and what does that say about the
# precision the receiver needs on its sampling instant? (Next chapter.)
#
# **Reference.** J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., Secs.
# 4.2 and 9.2.
#
# Next: :doc:`plot_05_synchronisation`.
