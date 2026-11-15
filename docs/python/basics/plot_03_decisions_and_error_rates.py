"""
Decisions and error rates
=========================

The receiver holds a noisy number :math:`y` and must say which symbol was sent. This
chapter answers three questions: what the best decision is, how often it is wrong,
and how a simulation should measure that.

**The best decision.** Among the :math:`M` points, the receiver should pick the most
probable given :math:`y`. When the points are equally likely and the noise is
Gaussian, the probability density of :math:`y` given that :math:`x_m` was sent is

.. math::

   p(y \\mid x_m) = \\frac{1}{\\pi N_0}\\exp\\!\\left(-\\frac{|y - x_m|^2}{N_0}\\right),

largest for the :math:`x_m` closest to :math:`y`. The optimal detector is therefore
the **nearest point**: it cuts the plane into one *decision region* per point, and
needs to know neither :math:`N_0` nor anything else about the noise.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, MUTED, SERIES

# %%
# Decision regions
# ----------------
# ``Constellation.decide`` returns the index of the nearest point. Colouring a grid of
# the plane by that index shows the regions: quadrants for QPSK, squares for 16-QAM.
grid = np.linspace(-1.6, 1.6, 321)
plane = (grid[None, :] + 1j * grid[:, None]).ravel()
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
for ax, (name, kind, m) in zip(axes, [("QPSK", "qam", 4), ("16-QAM", "qam", 16)], strict=True):
    c = comnum.Constellation(kind, m).normalized()
    regions = c.decide(plane).reshape(grid.size, grid.size)
    ax.imshow(
        regions, extent=(-1.6, 1.6, -1.6, 1.6), origin="lower", cmap="tab20", vmin=0, vmax=19,
        alpha=0.35,
    )  # fmt: skip
    ax.plot(c.points.real, c.points.imag, "o", color=MUTED, markeredgecolor=BACKGROUND)
    ax.set(xlabel="in phase", ylabel="quadrature", title=f"{name}: nearest point")
fig.tight_layout()

# %%
# The error probability of BPSK, from a picture
# ---------------------------------------------
# BPSK sends :math:`\pm\sqrt{E_b}` on the real axis. Only the real part of the noise
# matters, a Gaussian of variance :math:`N_0/2`. If :math:`+\sqrt{E_b}` was sent, the
# decision is wrong when the noise pushes :math:`y` below zero: the shaded tail.
#
# .. math::
#
#    P_b = P\!\left(n < -\sqrt{E_b}\right)
#        = Q\!\left(\frac{\sqrt{E_b}}{\sqrt{N_0/2}}\right)
#        = Q\!\left(\sqrt{\frac{2E_b}{N_0}}\right).
#
# The argument of :math:`Q` is the distance to the boundary in standard deviations of
# the noise. That is the general pattern: an error rate is a Gaussian tail at *half
# the distance between two points over the noise standard deviation*.
ebn0_db = 4.0
sigma = np.sqrt(1.0 / (2.0 * comnum.db2lin(ebn0_db)))  # Eb = 1, variance N0/2
axis = np.linspace(-3, 3, 601)


def density(centre: float) -> np.ndarray:
    """The Gaussian density of the real part of y around a point."""
    return np.exp(-((axis - centre) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(axis, density(+1), color=SERIES[0], label="sent $+\\sqrt{E_b}$")
ax.plot(axis, density(-1), color=SERIES[1], label="sent $-\\sqrt{E_b}$")
ax.fill_between(axis, density(+1), where=axis < 0, color=SERIES[0], alpha=0.35)
ax.axvline(0, color=GREY, linewidth=0.8)
ax.set(xlabel="real part of $y$", ylabel="density", title=f"BPSK at $E_b/N_0$ = {ebn0_db:.0f} dB")
ax.legend()
fig.tight_layout()

tail = comnum.qfunc(np.sqrt(2 * comnum.db2lin(ebn0_db)))
print(f"shaded area: Q(sqrt(2 Eb/N0)) = {tail:.4f}")
print(f"theory_ber:                     {comnum.theory_ber(2, ebn0_db, 'psk'):.4f}")

# %%
# QPSK is two BPSK links at once, one on each axis, each with half the symbol energy,
# that is the same energy per bit: its bit error rate is the same
# :math:`Q(\sqrt{2E_b/N_0})`. For a square QAM the same reasoning on each axis gives
# a sum of such tails; :func:`comnum.theory_ber` has the exact expressions.
#
# Counting errors
# ---------------
# A simulation estimates an error rate by a count: :math:`\hat P = e/n`, errors over
# bits sent. The count :math:`e` is a random number, close to a Poisson variable when
# errors are rare, so its standard deviation is about :math:`\sqrt{e}` and
#
# .. math::
#
#    \frac{\text{standard error of } \hat P}{\hat P} \approx \frac{1}{\sqrt{e}} .
#
# **The precision of a point depends on the number of errors counted, not on the
# number of bits sent.** One hundred errors give 10 %, four hundred 5 %, ten thousand
# 1 %. A point with three errors is a rumour.
#
# The experiment below runs the same BPSK link fifty times with short blocks, then
# fifty times with blocks sixteen times longer: the spread of the estimates shrinks by
# four.
rng = comnum.Rng(5)
bpsk = comnum.Constellation("psk", 2).normalized()
truth = comnum.theory_ber(2, 6.0, "psk")


def one_run(n_bits: int) -> float:
    """The bit error rate of one block of BPSK at 6 dB."""
    tx = comnum.bits(n_bits, rng)
    return comnum.ber(bpsk.demap_bits(comnum.awgn_ebn0(bpsk.map_bits(tx), 6.0, 1, rng)), tx)


fig, ax = plt.subplots(figsize=(7, 3.4))
for row, n_bits in enumerate((20_000, 320_000)):
    runs = np.array([one_run(n_bits) for _ in range(50)])
    errors = truth * n_bits
    ax.plot(runs, np.full(runs.size, row), "|", color=SERIES[row], markersize=18)
    print(
        f"{n_bits:>7} bits, about {errors:4.0f} errors per run: "
        f"spread {100 * runs.std() / truth:4.1f} %, predicted {100 / np.sqrt(errors):4.1f} %"
    )
ax.axvline(truth, color=GREY, label="true value")
ax.set(
    yticks=[0, 1],
    yticklabels=["20 000 bits", "320 000 bits"],
    ylim=(-0.6, 1.6),
    xlabel="estimated bit error rate",
    title="Fifty runs of the same link",
)
ax.legend()
fig.tight_layout()

# %%
# Hence the rule used throughout this course: **simulate until a fixed number of
# errors**, not a fixed number of bits. The cost then grows as the error rate falls,
# which is why simulated curves stop around :math:`10^{-6}`: at that rate four hundred
# errors need four hundred million bits.
MIN_ERRORS = 400
BLOCK = 200_000
MAX_BITS = 40_000_000


def simulate(c: comnum.Constellation, ebn0_db: float) -> tuple[float, int]:
    """The bit error rate of the constellation over AWGN and the errors counted."""
    k = c.bits_per_symbol
    errors = sent = 0
    while errors < MIN_ERRORS and sent < MAX_BITS:
        tx = comnum.bits(k * BLOCK, rng)
        rx = c.demap_bits(comnum.awgn_ebn0(c.map_bits(tx), ebn0_db, k, rng))
        errors += round(comnum.ber(rx, tx) * tx.size)
        sent += tx.size
    return errors / sent, errors


# %%
# The curves
# ----------
# BPSK, QPSK and 16-QAM against :math:`E_b/N_0`, each point stopped at 400 errors and
# drawn with an interval of two standard errors. BPSK and QPSK coincide, as argued
# above. 16-QAM needs about 4 dB more for the same error rate: the price of two more
# bits per symbol found in chapter 1, :math:`d_{\min}^2/E_b` falling from 4 to 1.6.
cases = [("BPSK", "psk", 2, "o"), ("QPSK", "qam", 4, "x"), ("16-QAM", "qam", 16, "s")]
ebn0_fine = np.linspace(0, 14, 141)
fig, ax = plt.subplots()
for color, (name, kind, m, marker) in zip(SERIES, cases, strict=False):
    c = comnum.Constellation(kind, m).normalized()
    points = np.arange(0.0, 13.0, 2.0) if m == 16 else np.arange(0.0, 9.0, 2.0)
    results = [simulate(c, e) for e in points]
    rates = np.array([r for r, _ in results])
    counts = np.array([max(e, 1) for _, e in results])
    ax.semilogy(ebn0_fine, comnum.theory_ber(m, ebn0_fine, kind), color=color, linewidth=1.2)
    ax.errorbar(
        points, rates, yerr=2 * rates / np.sqrt(counts), fmt=marker, color=color,
        markeredgecolor=BACKGROUND if marker != "x" else color, capsize=2, label=name,
    )  # fmt: skip
    gap = np.max(np.abs(rates / comnum.theory_ber(m, points, kind) - 1))
    print(
        f"{name:>7}: within {100 * gap:.0f} % of the closed form ({MIN_ERRORS} errors: 5 % expected)"
    )
ax.plot([], [], color=MUTED, linewidth=1.2, label="closed forms")
ax.set(ylim=(1e-6, 0.3), xlim=(0, 14), xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate")
ax.legend()
fig.tight_layout()

# %%
# Bit errors and symbol errors
# ----------------------------
# The detector decides symbols; the user cares about bits. With Gray labels a symbol
# error is nearly always to a neighbour and flips one bit of the :math:`k`, so
#
# .. math::
#
#    P_b \approx \frac{P_s}{k} \quad \text{at high SNR.}
#
# 16-QAM at 10 dB: the measured ratio of the two rates is close to 4.
qam16 = comnum.Constellation("qam", 16).normalized()
labels, x = comnum.symbols(qam16, 500_000, rng)
y = comnum.awgn_ebn0(x, 10.0, 4, rng)
ser = np.mean(qam16.decide(y) != labels)
ber = comnum.ber(qam16.demap_bits(y), qam16.demap_bits(x))
print(f"symbol error rate {ser:.2e}, bit error rate {ber:.2e}, ratio {ser / ber:.2f}")

# %%
# Back to the message
# -------------------
# The text of the quickstart, at the error rates of the curve. Around
# :math:`10^{-2}` every other word is damaged; at :math:`10^{-4}` a message of 400
# bits usually arrives intact. Real links aim far lower, :math:`10^{-9}` and below,
# and get there with channel codes rather than power: chapter 7.
message = "Digital communications: getting bits through noise."
bits = comnum.text_to_bits(message)
qpsk = comnum.Constellation("qam", 4).normalized()
for e in (2.0, 5.0, 8.0):
    received = qpsk.demap_bits(comnum.awgn_ebn0(qpsk.map_bits(bits), e, 2, comnum.Rng(int(e))))
    rate = comnum.theory_ber(4, e, "qam")
    print(f"{e:3.0f} dB (BER {rate:.0e}): {comnum.bits_to_text(received)!r}")

# %%
# Summary
# -------
# * Over AWGN with equally likely symbols, the best decision is the nearest point.
# * An error rate is a Gaussian tail: :math:`Q` of half a distance over the noise
#   standard deviation. BPSK and QPSK: :math:`Q(\sqrt{2E_b/N_0})`.
# * The relative precision of a simulated error rate is :math:`1/\sqrt{\text{errors}}`:
#   stop on a number of errors.
# * With Gray labels, :math:`P_b \approx P_s/k`.
#
# **Try it.** Change ``MIN_ERRORS`` to 25 and run the curves again: the points scatter
# around the theory by 20 %. Then predict, before running it, how far 8-PSK sits from
# QPSK at a bit error rate of :math:`10^{-4}` using :math:`d_{\min}^2/E_b` from chapter
# 1 (1.76 against 4: 3.6 dB).
#
# **Reference.** J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., Secs.
# 4.1 to 4.3.
#
# Next: :doc:`plot_04_pulse_shaping`.
