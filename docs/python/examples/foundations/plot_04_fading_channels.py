"""
Error rates over AWGN and fading channels
=========================================

The performance of a link is a curve of error rate against the SNR, and the first
thing to check about a simulated curve is its closed form. This tutorial builds the
curves of a coherent receiver step by step, over the additive white Gaussian noise
channel first, then over the flat fading channels of the library, and compares every
one of them with the theory it should follow.

**Over AWGN**, the received symbol is :math:`y = x + w` with
:math:`w \\sim \\mathcal{CN}(0, N_0)`, and the bit error rate of Gray-mapped
constellations is a Gaussian tail, :math:`Q(\\sqrt{2E_b/N_0})` for BPSK and QPSK
(Proakis & Salehi, *Digital Communications*, 5th ed., Sec. 4.3).

**Over flat fading**, a gain :math:`h` multiplies every symbol, :math:`y = hx + w`,
known at the receiver, which divides by it. The conditional error rate is that of
AWGN at the instantaneous SNR :math:`\\gamma = |h|^2\\,E_s/N_0`, and the average error
rate is its expectation over the distribution of :math:`\\gamma`. The unified way to
write that expectation (M. K. Simon, M.-S. Alouini, *Proc. IEEE* 86(9), 1998; M.-S.
Alouini, A. J. Goldsmith, *IEEE Trans. Commun.* 47(9), 1999) uses Craig's form of the
:math:`Q` function, which puts :math:`\\gamma` inside an exponential: the average is a
finite integral of the moment generating function (MGF) of the SNR,
:math:`M_\\gamma(s) = \\mathbb{E}\\,e^{s\\gamma}`,

.. math::

   P_b = \\frac{1}{\\pi}\\int_0^{\\pi/2}
   M_\\gamma\\!\\left(-\\frac{g}{\\sin^2\\theta}\\right) d\\theta,
   \\qquad g = 1 \\text{ for BPSK},

and the MGF is known in closed form for every usual channel. In the library a channel
is one object, :class:`comnum.Fading`, which draws the gains of a simulation
(``gains``) and evaluates the averages (``ber``, ``ser``, ``ber_noncoherent``), for

.. list-table::
   :header-rows: 1

   * - Channel
     - Gain :math:`h`
     - MGF :math:`M_\\gamma(s)` of mean :math:`\\bar\\gamma`
   * - Rayleigh
     - :math:`\\mathcal{CN}(0, 1)`: scattered paths only
     - :math:`(1 - s\\bar\\gamma)^{-1}`
   * - Rice, factor :math:`K`
     - a line of sight of power :math:`K/(1+K)` plus a scattered term
     - :math:`\\frac{1+K}{1+K-s\\bar\\gamma}\\exp\\frac{Ks\\bar\\gamma}{1+K-s\\bar\\gamma}`
   * - Nakagami-m
     - :math:`|h|^2 \\sim \\Gamma(m)/m`, the fit of many measured channels
     - :math:`(1 - s\\bar\\gamma/m)^{-m}`
   * - Hoyt, parameter :math:`q`
     - a Gaussian with unequal variances on the two axes
     - :math:`\\bigl(1 - 2s\\bar\\gamma + \\tfrac{4q^2s^2\\bar\\gamma^2}{(1+q^2)^2}\\bigr)^{-1/2}`

Rayleigh is Rice at :math:`K = 0`, Nakagami at :math:`m = 1` and Hoyt at :math:`q = 1`;
:math:`K \\to \\infty` or :math:`m \\to \\infty` is the AWGN channel. The simulations
below draw the gains with the generator of the library, so every point is reproducible
from its seed, and stop at a fixed number of errors.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, MUTED

MIN_ERRORS = 400
BLOCK = 100_000
MAX_DECISIONS = 4e7


def error_rate(trial, stream: str, tag: int) -> float:
    """The error rate of `trial`, which draws one block from the generator it is given
    and returns its errors and its decisions: blocks on fresh streams until
    `MIN_ERRORS` errors or `MAX_DECISIONS` decisions."""
    errors = decisions = t = 0
    while errors < MIN_ERRORS and decisions < MAX_DECISIONS:
        e, n = trial(comnum.Rng.stream(4, stream, tag, t))
        errors, decisions, t = errors + e, decisions + n, t + 1
    return errors / decisions


def simulate(c: comnum.Constellation, ebn0_db: float, channel: comnum.Fading, tag: int) -> float:
    """The BER of the constellation `c` at the mean Eb/N0 over the channel, with
    coherent detection: the receiver divides by the gain and decides."""
    n0 = 1.0 / (comnum.db2lin(ebn0_db) * c.bits_per_symbol)

    def trial(rng: comnum.Rng) -> tuple[int, int]:
        tx = comnum.bits(BLOCK * c.bits_per_symbol, rng)
        h = channel.gains(BLOCK, rng)
        y = h * c.map_bits(tx) + rng.complex_noise(BLOCK, n0)
        return int(np.count_nonzero(c.demap_bits(y / h) != tx)), tx.size

    return error_rate(trial, f"ber-{channel.kind}", tag)


# %%
# AWGN: the waterfall of BPSK and 16-QAM
# ---------------------------------------
# The simulated points fall on the closed forms of :func:`comnum.theory_ber`.
bpsk = comnum.Constellation("psk", 2).normalized()
qam16 = comnum.Constellation("qam", 16).normalized()
eb_awgn = np.arange(0.0, 15.0, 2.0)
eb = np.linspace(0, 40, 401)

fig, ax = plt.subplots()
for c, name, kind, marker in [(bpsk, "BPSK", "psk", "o"), (qam16, "16-QAM", "qam", "s")]:
    sim = [simulate(c, e, comnum.Fading("awgn"), i) for i, e in enumerate(eb_awgn)]
    (line,) = ax.semilogy(
        eb, comnum.theory_ber(len(c), eb, kind), linewidth=1.4, label=f"{name}, theory"
    )
    keep = np.array(sim) > 0
    ax.plot(
        eb_awgn[keep],
        np.array(sim)[keep],
        marker,
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        label=f"{name}, simulated",
    )
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 14),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="Coherent detection over AWGN",
)
ax.legend()
fig.tight_layout()

# %%
# Fading: BPSK over Rayleigh, Rice, Nakagami-m and Hoyt
# ------------------------------------------------------
# Deep fades dominate the errors: over Rayleigh fading the BER decays by one decade
# per 10 dB (:math:`\approx 1/(4\bar\gamma)`) instead of exponentially. A line of
# sight (Rice) or a less severe distribution (Nakagami with :math:`m > 1`) brings the
# curve back towards AWGN; the one-sided Gaussian (Nakagami :math:`m = 1/2`, Hoyt
# :math:`q = 0`) is worse than Rayleigh. Every simulated point sits on the MGF average
# of :meth:`comnum.Fading.ber`.
rayleigh = comnum.Fading("rayleigh")
rice10 = comnum.Fading("rice", 10.0)
nakagami2 = comnum.Fading("nakagami", 2.0)
channels = [
    (rayleigh, "Rayleigh", "o"),
    (comnum.Fading("rice", 3.0), "Rice, $K$ = 3", "s"),
    (rice10, "Rice, $K$ = 10", "D"),
    (comnum.Fading("nakagami", 0.5), "Nakagami, $m$ = 0.5", "v"),
    (nakagami2, "Nakagami, $m$ = 2", "^"),
    (comnum.Fading("nakagami", 4.0), "Nakagami, $m$ = 4", ">"),
    (comnum.Fading("hoyt", 0.5), "Hoyt, $q$ = 0.5", "p"),
]
eb_fading = np.arange(0.0, 41.0, 4.0)
fading_sims = {}

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(eb, comnum.theory_ber(2, eb, "psk"), color=GREY, linewidth=1.2, label="AWGN")
for i, (channel, name, marker) in enumerate(channels):
    sim = np.array([simulate(bpsk, e, channel, 100 * i + j) for j, e in enumerate(eb_fading)])
    fading_sims[name] = (channel.ber(eb_fading), sim)
    (line,) = ax.semilogy(eb, channel.ber(eb), linewidth=1.4)
    keep = sim > 0
    ax.plot(
        eb_fading[keep],
        sim[keep],
        marker,
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        label=name,
    )
ax.plot([], [], color=MUTED, linewidth=1.4, label="MGF average")
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 40),
    xlabel=r"average $E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="BPSK with coherent detection over flat fading",
)
ax.legend(ncol=2, fontsize="small")
fig.tight_layout()

# %%
# Every point stops at 400 errors, a relative standard deviation of 5 %, unless it
# reaches the budget of symbols first (a BER below 1e-5): over the points with their
# 400 errors, the largest gap to the MGF average is the sampling error.
for name, (expected, sim) in fading_sims.items():
    keep = expected >= MIN_ERRORS / MAX_DECISIONS
    gap = np.max(np.abs(sim[keep] / expected[keep] - 1))
    print(f"{name:>20}: simulated within {100 * gap:.1f} % of the MGF average")

# %%
# Symbol errors of 16-QAM
# -----------------------
# The same integral gives the symbol error rate of square QAM (two intervals, Alouini &
# Goldsmith 1999, eq. (48)), against the mean :math:`E_s/N_0`.
es = np.linspace(0, 40, 401)
es_points = np.arange(0.0, 41.0, 4.0)


def simulate_ser(
    c: comnum.Constellation, esn0_db: float, channel: comnum.Fading, tag: int
) -> float:
    """The SER of the constellation `c` at the mean Es/N0 over the channel."""
    n0 = 1.0 / comnum.db2lin(esn0_db)

    def trial(rng: comnum.Rng) -> tuple[int, int]:
        labels, x = comnum.symbols(c, BLOCK, rng)
        h = channel.gains(BLOCK, rng)
        y = h * x + rng.complex_noise(BLOCK, n0)
        return int(np.count_nonzero(c.decide(y / h) != labels)), BLOCK

    return error_rate(trial, f"ser-{channel.kind}", tag)


fig, ax = plt.subplots()
ax.semilogy(es, comnum.theory_ser_esn0(16, es, "qam"), color=GREY, linewidth=1.2, label="AWGN")
for i, (channel, name, marker) in enumerate(
    [
        (rayleigh, "Rayleigh", "o"),
        (rice10, "Rice, $K$ = 10", "D"),
        (nakagami2, "Nakagami, $m$ = 2", "^"),
    ]
):
    (line,) = ax.semilogy(es, channel.ser(16, es, "qam"), linewidth=1.4)
    sim = np.array([simulate_ser(qam16, e, channel, 10 * i + j) for j, e in enumerate(es_points)])
    keep = sim > 0
    ax.plot(
        es_points[keep],
        sim[keep],
        marker,
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        label=name,
    )
ax.plot([], [], color=MUTED, linewidth=1.4, label="MGF average")
ax.set(
    ylim=(1e-5, 1),
    xlim=(0, 40),
    xlabel=r"average $E_s/N_0$ (dB)",
    ylabel="symbol error rate",
    title="16-QAM over flat fading",
)
ax.legend()
fig.tight_layout()

# %%
# Noncoherent detection: DPSK
# ---------------------------
# A differential receiver needs no channel estimate: it decides on
# :math:`\mathrm{Re}\{y_k y_{k-1}^*\}`. Its conditional BER is
# :math:`\frac{1}{2}e^{-\gamma}` over AWGN, so the average over the fading is simply
# :math:`\frac{1}{2}M_\gamma(-1)`: :math:`1/(2(1 + \bar\gamma))` over Rayleigh fading,
# a decade per 10 dB as well but 3 dB behind coherent BPSK at high SNR. The gain is
# constant over a block of 64 symbols here (slow fading), drawn anew for every block.
SLOW_BLOCK = 64


def simulate_dpsk(ebn0_db: float, channel: comnum.Fading, tag: int) -> float:
    """The BER of binary DPSK at the mean Eb/N0, the gain constant over `SLOW_BLOCK`
    symbols."""
    n0 = 1.0 / comnum.db2lin(ebn0_db)
    blocks = BLOCK // SLOW_BLOCK

    def trial(rng: comnum.Rng) -> tuple[int, int]:
        tx = comnum.bits(blocks * SLOW_BLOCK, rng).reshape(blocks, SLOW_BLOCK)
        # the differential encoding, a reference symbol first
        x = np.cumprod(np.hstack([np.ones((blocks, 1)), 1.0 - 2.0 * tx]), axis=1)
        h = channel.gains(blocks, rng)[:, None]
        y = h * x + rng.complex_noise(x.size, n0).reshape(x.shape)
        rx = (np.real(y[:, 1:] * np.conj(y[:, :-1])) < 0).astype(np.uint8)
        return int(np.count_nonzero(rx != tx)), tx.size

    return error_rate(trial, f"dpsk-{channel.kind}", tag)


fig, ax = plt.subplots()
ax.semilogy(eb, comnum.theory_ber_dpsk(2, eb), color=GREY, linewidth=1.2, label="AWGN")
for i, (channel, name, marker) in enumerate(
    [(rayleigh, "Rayleigh", "o"), (rice10, "Rice, $K$ = 10", "D")]
):
    (line,) = ax.semilogy(eb, channel.ber_noncoherent(eb), linewidth=1.4)
    sim = np.array([simulate_dpsk(e, channel, 10 * i + j) for j, e in enumerate(eb_fading)])
    keep = sim > 0
    ax.plot(
        eb_fading[keep],
        sim[keep],
        marker,
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        label=name,
    )
ax.semilogy(
    eb,
    rayleigh.ber(eb),
    color=MUTED,
    linestyle=":",
    linewidth=1.2,
    label="coherent BPSK, Rayleigh",
)
ax.plot([], [], color=MUTED, linewidth=1.4, label=r"$\frac{1}{2}M_\gamma(-1)$")
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 40),
    xlabel=r"average $E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="Binary DPSK with differential detection",
)
ax.legend()
fig.tight_layout()

# %%
# Diversity: maximal-ratio combining
# ----------------------------------
# With :math:`L` independent Rayleigh branches combined by maximal ratio, the SNR is
# the sum of the branch SNRs and the MGF is the product of theirs: the BER decays by
# :math:`L` decades per 10 dB. The MRC of :math:`L` Rayleigh branches of mean
# :math:`\bar\gamma/L` has exactly the distribution of a Nakagami-m channel of
# :math:`m = L` and mean :math:`\bar\gamma` (Alouini & Goldsmith 1999, Appendix B).


def simulate_mrc(ebn0_db: float, branches: int, tag: int) -> float:
    """BPSK with maximal-ratio combining of `branches` Rayleigh branches, each of the
    mean Eb/N0 `ebn0_db`."""
    n0 = 1.0 / comnum.db2lin(ebn0_db)

    def trial(rng: comnum.Rng) -> tuple[int, int]:
        tx = comnum.bits(BLOCK, rng)
        x = 1.0 - 2.0 * tx
        z = np.zeros(BLOCK, dtype=complex)
        for _ in range(branches):
            h = rng.complex_noise(BLOCK, 1.0)
            z += np.conj(h) * (h * x + rng.complex_noise(BLOCK, n0))
        return int(np.count_nonzero((z.real < 0) != (tx == 1))), BLOCK

    return error_rate(trial, "mrc", tag)


fig, ax = plt.subplots()
ax.semilogy(eb, comnum.theory_ber(2, eb, "psk"), color=GREY, linewidth=1.2, label="AWGN")
for i, (branches, marker) in enumerate([(1, "o"), (2, "s"), (4, "D")]):
    (line,) = ax.semilogy(eb, rayleigh.ber(eb, branches=branches), linewidth=1.4)
    sim = np.array([simulate_mrc(e, branches, 10 * i + j) for j, e in enumerate(eb_fading)])
    keep = sim > 0
    ax.plot(
        eb_fading[keep],
        sim[keep],
        marker,
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        label=f"$L$ = {branches}",
    )
nakagami = comnum.Fading("nakagami", 4.0).ber(eb + 10 * np.log10(4))
ax.semilogy(
    eb,
    nakagami,
    color=MUTED,
    linestyle=":",
    linewidth=2.0,
    label=r"Nakagami, $m$ = 4, mean $4\bar\gamma$: the same curve as $L$ = 4",
)
ax.plot([], [], color=MUTED, linewidth=1.4, label="MGF average")
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 40),
    xlabel=r"average $E_b/N_0$ per branch (dB)",
    ylabel="bit error rate",
    title="BPSK with maximal-ratio combining over Rayleigh fading",
)
ax.legend()
fig.tight_layout()

# %%
# The price of fading: the Eb/N0 of a BER of 1e-3
# ------------------------------------------------
# Rayleigh fading costs 17 dB at this error rate, and the one-sided Gaussian
# (Nakagami :math:`m = 1/2`) 40 dB; a line of sight of :math:`K = 10` gives all but 3 dB
# back. Two combined branches give 13 dB back, and four bring the :math:`E_b/N_0` per
# branch below that of AWGN, since the combined SNR is :math:`L` times the branch SNR.
TARGET = 1e-3


def ebn0_at(f, target: float = TARGET) -> float:
    lo, hi = -5.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return lo


rows = [("AWGN, BPSK", lambda e: comnum.theory_ber(2, e, "psk"))]
rows += [(name.replace("$", ""), channel.ber) for channel, name, _ in channels]
rows += [
    ("Rayleigh, DPSK", rayleigh.ber_noncoherent),
    ("Rayleigh, MRC L = 2", lambda e: rayleigh.ber(e, branches=2)),
    ("Rayleigh, MRC L = 4", lambda e: rayleigh.ber(e, branches=4)),
]
reference = ebn0_at(rows[0][1])
for name, f in rows:
    e = ebn0_at(f)
    print(f"{name:>22}: {e:5.1f} dB ({e - reference:+5.1f} dB against AWGN)")
