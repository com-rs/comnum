"""
Bits and symbols
================

A transmitter cannot send a bit. It sends a waveform, and the waveform has to say
which bits it carries. The mapper is the step in between: it cuts the bit stream into
groups of :math:`k` bits and replaces each group by one of :math:`M = 2^k` complex
numbers, the *symbols*. The set of those numbers is the **constellation**.

Why complex? A radio signal around a carrier :math:`f_c` is
:math:`\\operatorname{Re}\\bigl(x(t)\\,e^{j2\\pi f_c t}\\bigr) = x_I(t)\\cos 2\\pi f_c t -
x_Q(t)\\sin 2\\pi f_c t`: two independent amplitudes, one on the cosine and one on the
sine. The complex number :math:`x = x_I + jx_Q` holds both, and everything in this
course happens on :math:`x`, the *baseband* signal, without the carrier.

This chapter covers three questions: what the bits of a message are, how they are
laid out on a constellation, and what it costs to put more bits in a symbol.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, MUTED, SERIES

# %%
# From a text to bits
# -------------------
# A text is a sequence of bytes (its UTF-8 encoding) and a byte is eight bits.
# :func:`comnum.text_to_bits` writes them most significant bit first; an accented
# letter takes two bytes.
for text in ("A", "Hi", "é"):
    print(f"{text!r:>5}: {''.join(map(str, comnum.text_to_bits(text)))}")
bits = comnum.text_to_bits("Hi")
assert comnum.bits_to_text(bits) == "Hi"

# %%
# A real source is compressed and encrypted before it reaches the mapper, so its bits
# look like the tosses of a fair coin. Simulations therefore draw random bits,
# :func:`comnum.bits`, which is also what makes a long simulation possible.
rng = comnum.Rng(1)
random_bits = comnum.bits(100_000, rng)
print(f"{random_bits[:24]}…  fraction of ones: {random_bits.mean():.3f}")

# %%
# Constellations
# --------------
# :class:`comnum.Constellation` holds the points and the bits each one stands for.
# Three families cover most links:
#
# * **PSK** (phase-shift keying): points on a circle. BPSK has two, QPSK four. QPSK is
#   also the smallest QAM, and is built here as ``"qam", 4``: the four corners of a
#   square (``"psk", 4`` is the same constellation turned by 45°).
# * **QAM** (quadrature amplitude modulation): a square grid, 16, 64, 256 points.
# * **PAM** (pulse amplitude modulation): points on a line, a real signal.
#
# The label next to each point is the group of bits it carries.
families = [
    ("BPSK", comnum.Constellation("psk", 2)),
    ("QPSK", comnum.Constellation("qam", 4)),
    ("8-PSK", comnum.Constellation("psk", 8)),
    ("16-QAM", comnum.Constellation("qam", 16)),
]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
for ax, (name, c) in zip(axes, families, strict=True):
    c = c.normalized()
    labels = c.demap_bits(c.points).reshape(-1, c.bits_per_symbol)
    ax.plot(c.points.real, c.points.imag, "o", color=SERIES[0], markeredgecolor=BACKGROUND)
    for p, label in zip(c.points, labels, strict=True):
        ax.annotate(
            "".join(map(str, label)),
            (p.real, p.imag),
            (4, 5),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    ax.set(aspect="equal", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), title=name)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
fig.tight_layout()

# %%
# Mapping is a table lookup: ``map_bits`` reads :math:`k` bits and returns the point
# with that label; ``demap_bits`` goes back.
qpsk = comnum.Constellation("qam", 4).normalized()
symbols = qpsk.map_bits(bits)
print(f"{bits.size} bits -> {symbols.size} symbols")
for group, s in zip(bits.reshape(-1, 2)[:4], symbols[:4], strict=True):
    print(f"  {group[0]}{group[1]} -> {s.real:+.3f} {s.imag:+.3f}j")
assert np.array_equal(qpsk.demap_bits(symbols), bits)

# %%
# Energy
# ------
# The energy of a symbol is :math:`|x|^2`, and the **average energy per symbol** of a
# constellation used with equally likely points is
#
# .. math::
#
#    E_s = \frac{1}{M}\sum_{m=1}^{M} |x_m|^2 .
#
# Comparing two modulations is only fair at the same energy, so the convention is
# :math:`E_s = 1`: ``normalized()`` scales the points to it. Since a symbol carries
# :math:`k` bits, the **energy per bit** is :math:`E_b = E_s / k`. It is the quantity
# a link pays for, and the one error rates are plotted against.
for name, c in families:
    raw = float(np.mean(np.abs(c.points) ** 2))
    unit = float(np.mean(np.abs(c.normalized().points) ** 2))
    k = c.bits_per_symbol
    print(
        f"{name:>7}: Es = {raw:5.2f} as built, {unit:.2f} normalised, Eb = Es/{k} = {unit / k:.3f}"
    )

# %%
# The price of a bit
# ------------------
# With the energy fixed, adding points to a constellation pushes them together. What
# protects a symbol from the noise is the distance to its nearest neighbour,
# :math:`d_{\min}`: the noise has to move the received point by :math:`d_{\min}/2` to
# cause an error. More bits per symbol means a higher rate in the same bandwidth, and
# a smaller :math:`d_{\min}` at the same energy. That trade is the first design choice
# of any link.


def min_distance(c: comnum.Constellation) -> float:
    """The smallest distance between two points of a constellation."""
    gaps = np.abs(c.points[:, None] - c.points[None, :])
    return float(gaps[gaps > 0].min())


orders = [("psk", 2), ("qam", 4), ("psk", 8), ("qam", 16), ("psk", 16), ("qam", 64), ("qam", 256)]
print(" modulation   bits/symbol   d_min at Es = 1   d_min^2 / Eb")
for kind, m in orders:
    c = comnum.Constellation(kind, m).normalized()
    d = min_distance(c)
    k = c.bits_per_symbol
    print(f"{m:>4}-{kind.upper():<5}        {k}            {d:.3f}           {d * d * k:5.2f}")

# %%
# The last column is :math:`d_{\min}^2/E_b`, the distance bought per unit of energy
# *per bit*. BPSK and QPSK tie at 4: QPSK carries twice the bits at the same cost per
# bit, which is why BPSK is rarely used alone. 16-QAM beats 16-PSK (1.6 against 0.61)
# because a grid fills the plane better than a circle. And each step up the QAM
# ladder costs 4 to 5 dB (a factor 2.5 to 3) per two extra bits.
#
# Gray labels
# -----------
# When the noise causes an error, it is almost always to a nearest neighbour. If the
# labels of two neighbours differ in one bit only, a symbol error costs one bit error
# and not :math:`k`. A labelling with that property is a **Gray code**; the
# constellations of the library use one.


def neighbour_bit_differences(c: comnum.Constellation) -> list[int]:
    """The number of differing bits between every pair of nearest neighbours."""
    labels = c.demap_bits(c.points).reshape(-1, c.bits_per_symbol)
    gaps = np.abs(c.points[:, None] - c.points[None, :])
    near = np.argwhere(np.isclose(gaps, min_distance(c)))
    return [int(np.count_nonzero(labels[i] != labels[j])) for i, j in near if i < j]


for kind, m in [("psk", 8), ("qam", 16), ("qam", 64)]:
    differences = neighbour_bit_differences(comnum.Constellation(kind, m).normalized())
    print(
        f"{m}-{kind.upper()}: {len(differences)} pairs of neighbours, all {set(differences)} bit apart"
    )

# %%
# Summary
# -------
# * A mapper sends :math:`k` bits as one of :math:`2^k` complex numbers.
# * Constellations are compared at unit average energy; :math:`E_b = E_s/k`.
# * More bits per symbol: more rate, less distance, more energy needed.
# * Gray labels make a symbol error cost one bit.
#
# **Try it.** Build ``comnum.Constellation("pam", 4)`` and find its
# :math:`d_{\min}^2/E_b`: why is it worse than QPSK, which also carries two bits?
# (QPSK uses two dimensions, 4-PAM one.)
#
# **Reference.** J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., Secs.
# 3.2 and 4.3.
#
# Next: :doc:`plot_02_the_noisy_channel`.
