"""
Synchronisation
===============

Until now the receiver knew everything except the bits: when each symbol was, where
the frame started, and the exact frequency and phase of the carrier. A real receiver
knows none of it. Its oscillator is not the transmitter's, the propagation delay is
arbitrary, and nothing marks the beginning of the message. After the matched filter,
what it has is

.. math::

   r(t) = e^{j(2\\pi \\Delta f\\,t + \\theta)} \\sum_k a_k\\,g(t - kT - \\tau) + n(t),

with three unknowns to remove before it can decide:

* the **timing** :math:`\\tau`: which sample of each symbol to keep;
* the **carrier**: a frequency offset :math:`\\Delta f` and a phase :math:`\\theta`;
* the **frame**: which symbol is the first of the message.

This chapter sends the message of the quickstart through a channel that hides all
three, looks at the damage, and repairs it one step at a time. Each step uses a
property of the signal rather than knowledge of the data, except the last.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, SERIES

SPS = 8
ROLLOFF = 0.35
SPAN = 32
PREAMBLE = 32  # known symbols at the start of the frame

rng = comnum.Rng(5)
qpsk = comnum.Constellation("qam", 4).normalized()
rrc = comnum.pulse_shape("rrc", SPS, n_taps=SPAN * SPS + 1, rolloff=ROLLOFF)
rrc = rrc / np.linalg.norm(rrc)  # unit energy

# %%
# The frame
# ---------
# A **preamble** of 32 symbols known to both ends, then the message. Random symbols
# before and after stand for the rest of the traffic.
message = "Digital communications: getting bits through noise."
_, preamble = comnum.symbols(qpsk, PREAMBLE, comnum.Rng(100))
payload = qpsk.map_bits(comnum.text_to_bits(message))
frame = np.concatenate([preamble, payload])
_, before = comnum.symbols(qpsk, 57, rng)
_, after = comnum.symbols(qpsk, 400, rng)
symbols = np.concatenate([before, frame, after])
print(f"preamble {PREAMBLE} symbols, message {payload.size} symbols, frame at symbol {before.size}")

# %%
# The channel
# -----------
# A delay of 3.4 samples (a fraction of a symbol the receiver cannot know), a carrier
# 0.4 % of the symbol rate off (4 kHz at 1 MBd, or 2 parts per million of a 2 GHz
# carrier), an arbitrary phase, and noise at :math:`E_b/N_0 = 10` dB.
EBN0_DB = 10.0
FREQUENCY_OFFSET = 0.004  # cycles per symbol
DELAY = 3.4  # samples

transmitted = comnum.fir_filter(rrc, comnum.upsample(symbols, SPS))
delayed = comnum.delay_fft(transmitted, DELAY, 1.0)
rotated = comnum.freq_shift(delayed, FREQUENCY_OFFSET, SPS) * np.exp(1j * 2.3)
received = comnum.awgn_ebn0(rotated, EBN0_DB, 2, rng, sps=SPS)
filtered = comnum.fir_filter(rrc, received)

# %%
# A receiver that ignores all this, samples where it believes the symbols are and
# decides, gets nothing: the frequency offset turns the constellation by 1.4° per
# symbol, a full turn every 250 symbols, and the points smear into a ring.
naive = comnum.downsample(filtered, SPS)
fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.plot(naive.real, naive.imag, ".", color=SERIES[0], markersize=3)
ax.set(aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), title="Sampled without synchronisation")
fig.tight_layout()

# %%
# Step 1: timing
# --------------
# At the right instant every sample is a symbol, of full amplitude. Half a symbol
# away the samples are mixtures of two neighbours, sometimes large, sometimes nearly
# zero: their mean power is lower. So the power of the samples taken at each of the
# ``SPS`` possible phases has a maximum at the right one, and it can be found without
# knowing the data or the carrier (a rotation does not change a modulus).
# :func:`comnum.decimate` keeps the phase of largest power.
power = [np.mean(np.abs(filtered[phase::SPS]) ** 2) for phase in range(SPS)]
timed = comnum.decimate(filtered, SPS, 1)

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(range(SPS), power, color=SERIES[0])
ax.axvline(DELAY, color=GREY, linestyle=":", label="true delay")
ax.set(
    xlabel="sampling phase (samples)", ylabel="mean power", title="Power against the sampling phase"
)
ax.legend()
fig.tight_layout()
print(f"largest power at phase {int(np.argmax(power))}, true delay {DELAY} samples")

# %%
# The estimate is the nearest sample, 0.4 sample from the truth, 5 % of a symbol: the
# eye of a roll-off of 0.35 is wide enough to forgive it. When the two clocks drift
# against each other the phase must be tracked instead: a loop driven by Gardner's
# timing error detector does it at two samples per symbol
# (:func:`comnum.gardner_clock_recovery`).
#
# Step 2: carrier frequency
# -------------------------
# The samples are now symbols, still spinning. A QPSK symbol is one of four phases
# 90° apart; raised to the fourth power, all four become the same number. The
# modulation disappears and what is left is a pure tone at four times the frequency
# offset: a peak in the spectrum of :math:`r^4`. :func:`comnum.fourth_power_foe` finds
# it and removes the offset.
steady, offset = comnum.fourth_power_foe(timed, 1.0, m=4)
print(f"frequency offset: estimated {offset:.5f}, true {FREQUENCY_OFFSET} cycles per symbol")

frequencies, spectrum = comnum.psd(timed**4, nfft=512)
frequencies = frequencies / 4  # the line of the fourth power is at four times the offset
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(frequencies, spectrum, color=SERIES[0])
ax.axvline(FREQUENCY_OFFSET, color=GREY, linestyle=":", label="true offset")
ax.set(xlim=(-0.02, 0.02), xlabel="frequency (cycles per symbol)", ylabel="$|\\mathrm{FFT}(r^4)|$",
       title="The fourth power has a line at the offset")  # fmt: skip
ax.legend()
fig.tight_layout()

# %%
# Step 3: carrier phase
# ---------------------
# The same trick gives the phase: the angle of the mean of :math:`r^4` over a window,
# divided by four. This is the Viterbi–Viterbi estimator
# (:func:`comnum.viterbi_viterbi`); a sliding window lets it follow a slow drift, such
# as what remains of the frequency offset.
phase = comnum.viterbi_viterbi(steady, n=20, m=4)
locked = comnum.derotate(steady, phase)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
for ax, (samples, title) in zip(
    axes,
    [(steady, "after the frequency correction"), (locked, "after the phase correction")],
    strict=True,
):
    ax.plot(samples.real, samples.imag, ".", color=SERIES[0], markersize=3)
    ax.plot(qpsk.points.real, qpsk.points.imag, "o", color=SERIES[1], markeredgecolor=BACKGROUND)
    ax.set(aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), title=title)
fig.tight_layout()

# %%
# Four clean clusters. But the fourth power cannot tell a constellation from the same
# constellation turned by 90°: the phase is known **up to a quarter turn**. Decided
# now, the bits would be right with one chance in four.
#
# Step 4: the frame, and the last ambiguity
# -----------------------------------------
# This is where the preamble comes in. The receiver slides the known symbols along
# what it received and correlates; the peak gives the position of the frame, and the
# phase of the correlation at the peak is the missing quarter turn.
# :func:`comnum.frame_sync` returns both, with the quality of the match.
start, rotation, quality = comnum.frame_sync(locked, frame, max_delay=120, stop=PREAMBLE)
print(f"frame found at symbol {start} (true {before.size}), quality {quality:.2f}")
print(f"rotation left by the phase recovery: {np.degrees(np.angle(rotation)) % 360:.0f} degrees")

recovered = locked[start + PREAMBLE : start + PREAMBLE + payload.size] * rotation
text = comnum.bits_to_text(qpsk.demap_bits(recovered))
print(f"received: {text!r}")
assert text == message

# %%
# What it costs
# -------------
# The same chain on random data, many frames, against the closed form: the estimates
# are not perfect, and their errors add a little to the noise. Two different things
# happen. In the frames that are found, the error rate is somewhat above the closed
# form: the timing is the nearest sample, not the exact instant, and the phase estimate
# is noisy. And as the noise grows, some frames are **lost** altogether: the phase
# estimate slips by a quarter turn in the middle of the frame and every bit after the
# slip is wrong. The two are counted apart below, because they are not the same
# failure: the first is a loss of a fraction of a decibel, the second a threshold.


def one_frame(ebn0_db: float, n_symbols: int) -> tuple[int, int]:
    """The bit errors and the bits of one frame through the whole receiver."""
    tx = comnum.bits(2 * n_symbols, rng)
    sent = np.concatenate([preamble, qpsk.map_bits(tx)])
    _, pad = comnum.symbols(qpsk, 2 * SPAN, rng)
    signal = comnum.fir_filter(rrc, comnum.upsample(np.concatenate([pad, sent, pad]), SPS))
    signal = comnum.delay_fft(signal, 8 * rng.uniform(1)[0], 1.0)
    signal = comnum.freq_shift(signal, FREQUENCY_OFFSET, SPS) * np.exp(
        2j * np.pi * rng.uniform(1)[0]
    )
    signal = comnum.awgn_ebn0(signal, ebn0_db, 2, rng, sps=SPS)
    samples = comnum.decimate(comnum.fir_filter(rrc, signal), SPS, 1)
    samples, _ = comnum.fourth_power_foe(samples, 1.0, m=4)
    samples = comnum.derotate(samples, comnum.viterbi_viterbi(samples, n=20, m=4))
    start, rotation, _ = comnum.frame_sync(samples, sent, max_delay=3 * SPAN, stop=PREAMBLE)
    data = samples[start + PREAMBLE : start + PREAMBLE + n_symbols] * rotation
    if data.size < n_symbols:  # the frame was not found where it could be
        return n_symbols, 2 * n_symbols
    return round(comnum.ber(qpsk.demap_bits(data), tx) * tx.size), tx.size


FRAMES = 300
points = np.arange(3.0, 10.0, 1.0)
measured, lost = [], []
for e in points:
    frames = np.array([one_frame(e, 2000)[0] for _ in range(FRAMES)]) / 4000
    failed = frames > 0.1  # a frame with more than 10 % of errors is a lost frame
    lost.append(int(np.count_nonzero(failed)))
    measured.append(frames[~failed].mean())

fine = np.linspace(3, 10, 71)
fig, ax = plt.subplots()
ax.semilogy(
    fine, comnum.theory_ber(4, fine, "qam"), color=GREY, label="theory, perfect synchronisation"
)
ax.plot(
    points, measured, "o", color=SERIES[0], markeredgecolor=BACKGROUND,
    label="timing, carrier and frame recovered",
)  # fmt: skip
ax.set(ylim=(1e-5, 0.1), xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate of the frames found")
ax.legend(fontsize="small")
fig.tight_layout()

for e, got, n_lost in zip(points, measured, lost, strict=True):
    ratio = got / comnum.theory_ber(4, e, "qam")
    print(
        f"{e:3.0f} dB: {n_lost:3d} frames lost of {FRAMES}, BER of the others {ratio:.2f} times the closed form"
    )

# %%
# Summary
# -------
# * Timing: the sampling phase of largest power; no data needed.
# * Carrier: the fourth power of QPSK removes the data and leaves the frequency and
#   the phase, up to a quarter turn.
# * Frame: correlation with a known preamble, which also resolves the quarter turn.
# * Blind where possible, data-aided where necessary; and every estimator has a
#   threshold below which it fails abruptly.
#
# **Try it.** Lower ``PREAMBLE`` to 8 and run the last section at 4 dB: how often is
# the frame found in the wrong place? Then raise ``FREQUENCY_OFFSET`` to 0.13: the
# fourth-power estimator can only see offsets below one eighth of the symbol rate
# (why?).
#
# **Reference.** U. Mengali, A. N. D'Andrea, *Synchronization Techniques for Digital
# Receivers*, Plenum, 1997; A. J. Viterbi, A. M. Viterbi, *IEEE Trans. Inf. Theory*
# 29(4), 1983.
#
# Next: :doc:`plot_06_multipath_and_equalisation`.
