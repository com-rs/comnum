"""
Multipath and equalisation
==========================

A radio wave reaches the receiver by several paths: the direct one, and reflections
on the ground, buildings or walls that arrive later and weaker. A cable with a badly
matched end does the same, and so does a loudspeaker in a room. The channel is then a
filter,

.. math::

   y_k = \\sum_{l=0}^{L-1} h_l\\,a_{k-l} + n_k ,

written here at one sample per symbol, after the matched filter and the
synchronisation of the last two chapters: each received sample is the wanted symbol
plus *echoes of the previous ones*. This is intersymbol interference again, but this
time the pulse is not to blame and Nyquist's criterion cannot be designed in: the
channel is not known in advance and changes from one place to the next.

This chapter looks at what such a channel does, at the idea of undoing it with a
filter (an **equaliser**), at what that costs, and at an equaliser that learns the
channel by itself from a known sequence.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, MUTED, SERIES

rng = comnum.Rng(6)
qpsk = comnum.Constellation("qam", 4).normalized()

# %%
# A channel with two echoes
# -------------------------
# The direct path, an echo one symbol later at 50 % of the amplitude and another two
# symbols later at 30 %, each with its own phase. The taps are scaled to a unit
# energy, so that the received energy per symbol is the transmitted one and
# :math:`E_b/N_0` keeps its meaning.
channel = np.array([1.0, 0.5 * np.exp(0.8j), 0.3 * np.exp(-1.9j)])
channel = channel / np.linalg.norm(channel)


def through_channel(symbols: np.ndarray, ebn0_db: float) -> np.ndarray:
    """The symbols after the echoes and the noise, one sample per symbol."""
    echoed = comnum.convolve(channel, symbols)[: symbols.size]
    return comnum.awgn_ebn0(echoed, ebn0_db, 2, rng)


# %%
# In frequency, echoes are ripples: the paths add in phase at some frequencies and
# cancel at others. A channel that does not treat all frequencies alike is called
# **frequency-selective**. Here the gain varies by 17 dB across the band of the signal.
frequencies = np.linspace(-0.5, 0.5, 401)
response = np.array([np.sum(channel * np.exp(-2j * np.pi * f * np.arange(3))) for f in frequencies])
gain_db = 20 * np.log10(np.abs(response))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(frequencies, gain_db, color=SERIES[0])
ax.axhline(0, color=GREY, linewidth=0.8)
ax.set(
    xlabel="frequency (symbol rates)", ylabel="gain (dB)", title="Frequency response of the channel"
)
fig.tight_layout()
print(f"gain from {gain_db.min():.1f} to {gain_db.max():.1f} dB across the band")

# %%
# What the receiver sees
# ----------------------
# Each received sample depends on three symbols: four values each, sixty-four
# combinations, sixty-four points instead of four. For eight of those combinations
# the two echoes together outweigh the wanted symbol and the point lands in the wrong
# quadrant: the nearest-point detector of chapter 3 is wrong on one symbol in eight
# *without any noise*. No amount of power fixes it: the interference grows with the
# signal.
labels, symbols = comnum.symbols(qpsk, 6000, rng)
clean = comnum.convolve(channel, symbols)[: symbols.size]
received = through_channel(symbols, 14.0)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
for ax, (samples, title) in zip(
    axes, [(clean, "after the channel, no noise"), (received, "at $E_b/N_0$ = 14 dB")], strict=True
):
    ax.plot(samples.real, samples.imag, ".", color=SERIES[0], markersize=3)
    ax.plot(qpsk.points.real, qpsk.points.imag, "o", color=SERIES[1], markeredgecolor=BACKGROUND)
    ax.set(aspect="equal", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), title=title)
fig.tight_layout()
print(f"symbol errors without noise: {100 * np.mean(qpsk.decide(clean) != labels):.1f} %")
print(f"symbol errors at 14 dB:      {100 * np.mean(qpsk.decide(received) != labels):.1f} %")

# %%
# The idea: invert the channel
# ----------------------------
# If the channel is a filter :math:`H(f)`, follow it with the filter
# :math:`1/H(f)`: the cascade is flat and the echoes are gone. This is the
# **zero-forcing** equaliser. It has a price. The noise goes through :math:`1/H(f)`
# too, and where the channel attenuates, the equaliser amplifies: the noise is
# **enhanced**. A channel with a deep notch makes zero forcing useless.
#
# The better compromise does not insist on removing all the interference. It
# minimises the total error at its output, interference plus noise: the **minimum
# mean square error** (MMSE) equaliser,
#
# .. math::
#
#    W(f) = \frac{H^*(f)}{|H(f)|^2 + N_0/E_s} ,
#
# which is the inverse where the channel is strong and backs off where it is weak.
# At high SNR the two coincide.
n0 = 1.0 / (2 * comnum.db2lin(8.0))
zero_forcing = 1.0 / response
mmse = np.conj(response) / (np.abs(response) ** 2 + n0)

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(frequencies, -gain_db, color=SERIES[1], label="zero forcing, $1/|H|$")
ax.plot(frequencies, 20 * np.log10(np.abs(mmse)), color=SERIES[0], label="MMSE at $E_b/N_0$ = 8 dB")
ax.axhline(0, color=GREY, linewidth=0.8)
ax.set(xlabel="frequency (symbol rates)", ylabel="gain of the equaliser (dB)")
ax.legend()
fig.tight_layout()
enhancement = np.mean(np.abs(zero_forcing) ** 2)
print(
    f"zero forcing multiplies the noise power by {enhancement:.2f} ({10 * np.log10(enhancement):.1f} dB)"
)

# %%
# An equaliser that learns
# ------------------------
# The receiver does not know :math:`H`. It can learn the equaliser directly: a filter
# of a few taps :math:`w`, adjusted so that its output matches symbols it knows. The
# **least mean squares** (LMS) algorithm does it one symbol at a time,
#
# .. math::
#
#    e_k = a_k - w^{\mathsf T} y_k, \qquad w \leftarrow w + \mu\,e_k\,y_k^* ,
#
# a small step against the gradient of :math:`|e_k|^2`, and converges to the MMSE
# equaliser. It needs the true symbols :math:`a_k`: a **training sequence** at the
# start of the frame. Once the eye is open the receiver's own decisions are right
# almost always and can replace the training symbols: the **decision-directed** mode,
# which also follows a channel that changes slowly.
#
# :func:`comnum.adaptive_equalizer` runs such a schedule: here 500 symbols of training
# with the normalised LMS, then decision-directed.
TRAINING = 500
TAPS = 11
schedule = [("nlms", TRAINING), ("dd-lms", 0)]

equalised, taps, squared_error = comnum.adaptive_equalizer(
    received, qpsk, algorithm=schedule, n_taps=TAPS, sps=1, mu=[0.1, 5e-3],
    reference=symbols.reshape(1, -1),
)  # fmt: skip
output = equalised[0]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
smooth = np.convolve(squared_error[0], np.ones(50) / 50, mode="valid")
axes[0].semilogy(smooth, color=SERIES[0])
axes[0].axvline(TRAINING, color=GREY, linestyle=":", label="end of the training")
axes[0].set(xlabel="symbol", ylabel="squared error (mean of 50)", title="Learning curve")
axes[0].legend()
tail = output[2000:]
axes[1].plot(tail.real, tail.imag, ".", color=SERIES[0], markersize=3)
axes[1].plot(qpsk.points.real, qpsk.points.imag, "o", color=SERIES[1], markeredgecolor=BACKGROUND)
axes[1].set(aspect="equal", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), title="After the equaliser")
fig.tight_layout()

errors = np.count_nonzero(qpsk.decide(output[TRAINING:]) != labels[TRAINING:])
print(f"symbol errors after the equaliser at 14 dB: {errors} of {labels.size - TRAINING}")

# %%
# The four clusters are back. To see what the equaliser learnt, put it after the
# channel: the cascade should be a single tap, the echoes cancelled. (The equaliser
# output is :math:`\sum_t w_t\,y_{k+t}` with the centre tap at :math:`t = 0`, so the
# cascade is the channel convolved with the taps reversed.)
cascade = np.convolve(channel, taps[0][::-1])
lag = np.arange(cascade.size) - (TAPS // 2)

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.stem(lag, np.abs(cascade), basefmt=" ")
ax.set(xlabel="delay (symbols)", ylabel="magnitude", title="Channel followed by the equaliser")
fig.tight_layout()
residual = np.sqrt(np.sum(np.abs(cascade) ** 2) - np.abs(cascade).max() ** 2)
print(f"main tap {np.abs(cascade).max():.2f}, everything else together {residual:.2f}")

# %%
# What it costs
# -------------
# Three receivers against :math:`E_b/N_0`: the detector alone, which hits a floor set
# by the interference; the equaliser; and the closed form of a channel without
# echoes. The equaliser removes the floor and pays for it with the noise it
# enhances: its curve runs to the right of the theory by several decibels, of the
# order of the noise enhancement computed above. That gap is the cost of this channel
# for a linear equaliser. Detectors that search for the most
# likely *sequence* of symbols (Viterbi), or modulations that split the band into many
# narrow carriers each seeing a flat channel (OFDM, in the :doc:`gallery
# <../auto_examples/index>`), recover part of it.
MIN_ERRORS = 400
BLOCK = 8000


def error_rates(ebn0_db: float) -> tuple[float, float]:
    """The bit error rates without and with the equaliser, the training left out."""
    plain = equal = sent = 0
    while equal < MIN_ERRORS and sent < 3_000_000:
        tx = comnum.bits(2 * BLOCK, rng)
        a = qpsk.map_bits(tx)
        y = through_channel(a, ebn0_db)
        z, _, _ = comnum.adaptive_equalizer(
            y,
            qpsk,
            algorithm=schedule,
            n_taps=TAPS,
            sps=1,
            mu=[0.1, 5e-3],
            reference=a.reshape(1, -1),
        )
        data = slice(2 * TRAINING, None)
        plain += round(comnum.ber(qpsk.demap_bits(y)[data], tx[data]) * tx[data].size)
        equal += round(comnum.ber(qpsk.demap_bits(z[0])[data], tx[data]) * tx[data].size)
        sent += tx[data].size
    return plain / sent, equal / sent


points = np.arange(4.0, 19.0, 2.0)
rates = np.array([error_rates(e) for e in points])
fine = np.linspace(4, 18, 141)

fig, ax = plt.subplots()
ax.semilogy(fine, comnum.theory_ber(4, fine, "qam"), color=GREY, label="no echoes (theory)")
ax.plot(
    points,
    rates[:, 0],
    "s",
    color=SERIES[1],
    markeredgecolor=BACKGROUND,
    label="echoes, no equaliser",
)
ax.plot(
    points,
    rates[:, 1],
    "o",
    color=SERIES[0],
    markeredgecolor=BACKGROUND,
    label="echoes, adaptive equaliser",
)
ax.plot(points, rates[:, 0], color=SERIES[1], linewidth=0.8)
ax.plot(points, rates[:, 1], color=SERIES[0], linewidth=0.8)
ax.set(ylim=(1e-5, 0.3), xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate")
ax.plot([], [], color=MUTED, linewidth=0)
ax.legend(fontsize="small")
fig.tight_layout()

for e, (plain, equal) in zip(points, rates, strict=True):
    print(
        f"{e:3.0f} dB: no equaliser {plain:.1e}, equaliser {equal:.1e}, no echoes {comnum.theory_ber(4, e, 'qam'):.1e}"
    )

# %%
# Summary
# -------
# * Echoes make the channel a filter: frequency-selective, and a source of
#   intersymbol interference that power cannot overcome.
# * An equaliser is a filter that undoes it. Zero forcing inverts the channel and
#   enhances the noise; MMSE balances the two.
# * LMS learns the equaliser from a training sequence, then follows on its own
#   decisions.
# * The price is a loss against the channel without echoes, larger when the channel
#   has deep notches.
#
# **Try it.** Make the first echo as strong as the direct path (``1.0`` instead of
# ``0.5``): the notch of the channel gets much deeper. What happens to the noise
# enhancement of zero forcing, and to the curve of the equaliser? Then lower
# ``TRAINING`` to 50: is the eye open when the decision-directed mode takes over?
#
# **Reference.** J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., Secs.
# 9.4 and 10.1; S. Haykin, *Adaptive Filter Theory*, 4th ed., Ch. 5.
