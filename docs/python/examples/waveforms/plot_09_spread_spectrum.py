"""
Spread spectrum: codes, processing gain, CDMA and the rake receiver
===================================================================

A spread-spectrum link sends every symbol on many more degrees of freedom than it
needs: :math:`N` chips per symbol, the *spreading factor* or *processing gain*. The
receiver, which knows the code, collapses the :math:`N` chips back into one symbol
and whatever does not carry the code (a jammer, another user, a delayed echo) is left
spread out, at about :math:`1/N` of its power. This tutorial follows the idea through
direct-sequence spreading: the codes and their correlations, the rejection of a
jammer, several users on the same band (CDMA) with the receivers that separate them,
and the rake receiver that turns the echoes of a multipath channel into diversity.
The chirp form of spread spectrum is in :doc:`plot_07_lora`.

**Direct sequence.** User :math:`k` sends the symbol :math:`b_k` on the signature
:math:`s_k \\in \\mathbb{C}^N`, :math:`\\lVert s_k\\rVert = 1`, with the amplitude
:math:`A_k`. With :math:`K` synchronous users, the :math:`N` chips of one symbol
period are

.. math::

   y = \\sum_{k=1}^{K} A_k b_k s_k + w = S A b + w,
   \\qquad w \\sim \\mathcal{CN}(0, \\sigma^2 I_N),

which :class:`comnum.Dsss` builds (``modulate``) from binary codes sent as the chips
:math:`\\pm 1/\\sqrt{N}`. Despreading is the matched filter :math:`s_k^H y`
(``demodulate``): the symbol comes back with its full energy, the noise keeps its
variance, so over white noise alone spreading changes nothing to the error rate.

**Codes.** What the matched filter leaves of user :math:`j` is the correlation
:math:`\\rho_{jk} = s_j^H s_k`, and of an echo delayed by :math:`l` chips the
autocorrelation of the code at the lag :math:`l`. The classical families trade the two
(D. V. Sarwate, M. B. Pursley, *Proc. IEEE* 68(5), 1980):

.. list-table::
   :header-rows: 1

   * - Codes
     - Length
     - Number
     - Periodic correlations
   * - m-sequence (:func:`comnum.m_sequence`)
     - :math:`2^r - 1`
     - 1
     - autocorrelation :math:`-1` off the peak
   * - Gold (:func:`comnum.gold_codes`; R. Gold, *IEEE Trans. Inf. Theory* 13(4), 1967)
     - :math:`2^r - 1`
     - :math:`2^r + 1`
     - :math:`\\{-1, -t, t - 2\\}`, :math:`t = 1 + 2^{\\lfloor (r+2)/2 \\rfloor}`
   * - Kasami, small set (:func:`comnum.kasami_codes`)
     - :math:`2^r - 1`
     - :math:`2^{r/2}`
     - :math:`\\{-1, -s, s - 2\\}`, :math:`s = 1 + 2^{r/2}`
   * - Walsh–Hadamard (:func:`comnum.walsh_codes`)
     - :math:`2^m`
     - :math:`2^m`
     - orthogonal when aligned, not otherwise
   * - Barker (:func:`comnum.barker_code`)
     - up to 13
     - 1
     - aperiodic sidelobes at most 1

**Multiuser receivers.** The model above is :math:`y = Hx + w` with :math:`H = SA`:
the chips play the receive antennas of a MIMO link and the users its streams, so the
receivers of CDMA are the detectors of :func:`comnum.mimo_detect` applied to the
signature matrix (``Dsss.detect``): the conventional receiver (``"mr"``, the bank of
matched filters), the decorrelator (``"zf"``; R. Lupas, S. Verdú, *IEEE Trans. Inf.
Theory* 35(1), 1989), the linear MMSE receiver, successive cancellation and the optimum
detector.

**Rake.** Over :math:`L` resolvable paths :math:`r[n] = \\sum_l g_l\\,x[n - d_l] +
w[n]`, one matched filter per path (a *finger*) sees its own path at full strength and
the others through the autocorrelation of the code; combining the fingers with the
weights :math:`g_l^*` (``Dsss.rake``; R. Price, P. E. Green, *Proc. IRE* 46(3), 1958) is
maximal-ratio combining of :math:`L` branches.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, GREY, MUTED, SERIES

bpsk = comnum.Constellation("psk", 2)


def bpsk_symbols(shape: tuple[int, ...], rng: comnum.Rng) -> tuple[np.ndarray, np.ndarray]:
    """Random labels of the given shape and their BPSK symbols."""
    labels = rng.integers(2, int(np.prod(shape))).reshape(shape)
    return labels, bpsk.points[labels]


# %%
# Codes and their correlations
# ----------------------------
# The m-sequence of order 7 (127 chips) has the ideal two-valued autocorrelation. A
# single register gives one such code; the 129 Gold codes of the same length keep the
# cross-correlation between any two of them, at any shift, on three values (here
# :math:`-17`, :math:`-1` and 15 out of 127), at the price of the same three values off
# the peak of their own autocorrelation. Two m-sequences taken without care do worse:
# the sequence and its mirror image, the m-sequence of the reciprocal polynomial, reach 21.
order = 7
n_chips = 2**order - 1
u = comnum.m_sequence(order)
gold = comnum.gold_codes(order)
t = 1 + 2 ** ((order + 2) // 2)
lags = np.arange(n_chips)

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
left.plot(lags, comnum.periodic_correlation(u, u), color=SERIES[0], label="m-sequence")
left.plot(lags, comnum.periodic_correlation(gold[5], gold[5]), color=SERIES[1], label="a Gold code")
left.set(xlabel="lag (chips)", ylabel="periodic correlation", title="Autocorrelation")
left.legend()
reversed_u = u[::-1].copy()  # the reciprocal polynomial: another m-sequence
right.plot(
    lags,
    comnum.periodic_correlation(u, reversed_u),
    color=GREY,
    label="two m-sequences",
)
right.plot(
    lags, comnum.periodic_correlation(gold[5], gold[20]), color=SERIES[1], label="two Gold codes"
)
for level in (-t, -1, t - 2):
    right.axhline(level, color=MUTED, linewidth=0.6, linestyle=":")
right.set(xlabel="lag (chips)", title="Cross-correlation")
right.legend()
fig.tight_layout()

values = set()
for i in range(0, len(gold), 7):
    for j in range(i + 1, len(gold), 5):
        values |= set(comnum.periodic_correlation(gold[i], gold[j]).tolist())
print(f"Gold codes of length {n_chips}: cross-correlations in {sorted(values)}")
worst = np.abs(comnum.periodic_correlation(u, reversed_u)).max()
print(f"the m-sequence and its mirror image: up to {worst}")

# %%
# Processing gain: a tone jammer
# ------------------------------
# A continuous tone falls in the band, 6 dB *above* the signal. Without spreading the
# link is lost. With spreading, the matched filter multiplies the tone by the code and
# sums :math:`N` chips: the jammer comes out with the power :math:`J\,|C(f)|^2/N^2`,
# :math:`C(f)` the spectrum of the code at the frequency of the tone, about :math:`J/N`.
# The bit error rate is then the average of a Gaussian tail over the phase
# :math:`\varphi` of the despread tone,
# :math:`\mathbb{E}_\varphi\,Q\bigl(\sqrt{2E_b/N_0}\,(1 + \rho\cos\varphi)\bigr)`, with
# :math:`\rho^2` the jammer-to-signal ratio after despreading.
JAMMER_DB = 6.0
TONE = 0.1234  # cycles per chip
N_SYMBOLS = 200_000
ebn0_db = np.arange(0.0, 13.0, 2.0)
ebn0_fine = np.linspace(0, 12, 121)
phases = np.linspace(0, 2 * np.pi, 720, endpoint=False)


def jammed_ber(link: comnum.Dsss, ebn0: float, tag: int) -> float:
    """The simulated BER of one user under the tone and white noise."""
    rng = comnum.Rng.stream(9, "jammer", link.spreading_factor, tag)
    n = N_SYMBOLS * 127 // link.spreading_factor  # the same number of chips for every code
    labels, symbols = bpsk_symbols((n,), rng)
    chips = link.modulate(symbols)
    # the same jammer-to-signal ratio per chip whatever the code: a chip has the power 1/N
    amplitude = np.sqrt(comnum.db2lin(JAMMER_DB) / link.spreading_factor)
    tone = amplitude * np.exp(2j * np.pi * TONE * np.arange(chips.size))
    received = chips + tone + rng.complex_noise(chips.size, 1.0 / comnum.db2lin(ebn0))
    return float(np.mean(bpsk.decide(link.demodulate(received)[0]) != labels))


def jammed_theory(link: comnum.Dsss, ebn0: np.ndarray) -> np.ndarray:
    """The average of the Gaussian tail over the phase of the despread tone."""
    code = link.signatures[:, 0]
    n = code.size
    rho = np.sqrt(comnum.db2lin(JAMMER_DB) / n) * abs(
        np.vdot(code, np.exp(2j * np.pi * TONE * np.arange(n)))
    )
    snr = np.sqrt(2 * comnum.db2lin(ebn0))[:, None]
    return comnum.qfunc(snr * (1 + rho * np.cos(phases))).mean(axis=1)


fig, ax = plt.subplots()
ax.semilogy(ebn0_fine, comnum.theory_ber(2, ebn0_fine, "psk"), color=GREY, label="no jammer")
links = [
    ("no spreading", comnum.Dsss([0])),
    ("$N$ = 7", comnum.Dsss(comnum.m_sequence(3))),
    ("$N$ = 31", comnum.Dsss(comnum.m_sequence(5))),
    ("$N$ = 127", comnum.Dsss(comnum.m_sequence(7))),
]
for color, (name, link) in zip(SERIES, links, strict=False):
    ax.semilogy(ebn0_fine, jammed_theory(link, ebn0_fine), color=color, linewidth=1.4)
    sim = np.array([jammed_ber(link, e, i) for i, e in enumerate(ebn0_db)])
    keep = sim > 0
    ax.plot(ebn0_db[keep], sim[keep], "o", color=color, markeredgecolor=BACKGROUND, label=name)
ax.set(
    ylim=(1e-5, 1),
    xlim=(0, 12),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title=f"BPSK under a tone jammer {JAMMER_DB:.0f} dB above the signal",
)
ax.legend()
fig.tight_layout()

# %%
# CDMA and the near–far problem
# -----------------------------
# Eight synchronous users share the band on Gold codes of length 31. The user of
# interest is received at :math:`E_b/N_0 = 8` dB; the seven others are all received
# :math:`\Delta` dB above or below it. The conventional receiver treats them as noise:
# as soon as they are a few dB stronger its error rate climbs towards 1/2, whatever
# the SNR. The decorrelator projects them out, so its error rate does not depend on
# their power at all; it pays a fixed noise enhancement, the near–far resistance
# :math:`1/(G_{kk}(G^{-1})_{kk})`. The MMSE receiver moves from one to the other: the
# matched filter when the others are weak, the decorrelator when they are strong.
# The curves are the exact error rates (:func:`comnum.theory_ber_cdma`: for the
# conventional receiver, the mean over the 128 patterns of the interfering bits) and,
# for the MMSE receiver, the Gaussian approximation :math:`Q(\sqrt{2\,\mathrm{SINR}})`
# with the SINR of ``Dsss.sinr``.
codes = comnum.gold_codes(5)[1:9]
EBN0_USER = 8.0
noise_var = 1.0 / comnum.db2lin(EBN0_USER)
delta_db = np.arange(-10.0, 21.0, 5.0)
delta_fine = np.linspace(-10, 20, 61)
receivers = [("mr", "conventional", "o"), ("zf", "decorrelator", "s"), ("mmse", "MMSE", "^")]


def near_far_link(delta: float) -> comnum.Dsss:
    """The eight users, the seven interferers `delta` dB above the first."""
    return comnum.Dsss(codes, amplitudes=[1.0] + [10 ** (delta / 20)] * 7)


def cdma_theory(cdma: comnum.Dsss, receiver: str) -> float:
    if receiver == "mmse":
        return float(comnum.qfunc(np.sqrt(2 * cdma.sinr(noise_var, "mmse")[0])))
    return float(comnum.theory_ber_cdma(cdma.signatures, noise_var, receiver))


fig, ax = plt.subplots()
ax.axhline(comnum.theory_ber(2, EBN0_USER, "psk"), color=GREY, linewidth=1.2, label="single user")
near_far = {}
for color, (receiver, name, marker) in zip(SERIES, receivers, strict=False):
    ax.semilogy(
        delta_fine,
        [cdma_theory(near_far_link(d), receiver) for d in delta_fine],
        color=color,
        linewidth=1.4,
    )
    sim = []
    for i, d in enumerate(delta_db):
        cdma = near_far_link(d)
        rng = comnum.Rng.stream(9, "near-far", i, 0)
        labels, symbols = bpsk_symbols((8, 400_000), rng)
        chips = cdma.modulate(symbols)
        chips += rng.complex_noise(chips.size, noise_var)
        decided = cdma.detect(chips, bpsk, noise_var, receiver)
        sim.append(np.mean(decided[0] != labels[0]))
    near_far[name] = (np.array([cdma_theory(near_far_link(d), receiver) for d in delta_db]), sim)
    ax.plot(delta_db, sim, marker, color=color, markeredgecolor=BACKGROUND, label=name)
ax.set(
    ylim=(1e-4, 1),
    xlabel=r"power of the seven interferers over the user, $\Delta$ (dB)",
    ylabel="bit error rate of the user",
    title="Eight users on Gold codes of length 31, $E_b/N_0$ = 8 dB",
)
ax.legend()
fig.tight_layout()

# %%
# The simulated points of the conventional receiver and of the decorrelator sit on
# their exact error rates (the sampling error of the number of errors counted). The
# curve of the MMSE receiver is an approximation, and an optimistic one: what is left
# of seven BPSK interferers is not Gaussian.
resistance = comnum.cdma_near_far_resistance(near_far_link(0.0).signatures)
print(f"near-far resistance of the user: {resistance:.3f} ({10 * np.log10(resistance):.2f} dB)")
for name, (expected, sim) in near_far.items():
    gap = np.max(np.abs(np.array(sim) / expected - 1))
    kind = "the Gaussian approximation" if name == "MMSE" else "the exact error rate"
    print(f"{name:>13}: simulated within {100 * gap:.0f} % of {kind}")

# %%
# How many users? Random signatures and large systems
# ---------------------------------------------------
# With long pseudo-random codes the signatures are, in effect, drawn at random, and
# the SINR of each user depends on the draw. In a large system it no longer does: at
# the load :math:`\beta = K/N` it converges to a limit that depends on the receiver
# only (D. N. C. Tse, S. V. Hanly, *IEEE Trans. Inf. Theory* 45(2), 1999; S. Verdú,
# S. Shamai, same issue):
#
# .. math::
#
#    \mathrm{SINR}_{\mathrm{MF}} = \frac{\mathrm{SNR}}{1 + \beta\,\mathrm{SNR}}, \qquad
#    \mathrm{SINR}_{\mathrm{dec}} = \mathrm{SNR}\,(1 - \beta), \qquad
#    \mathrm{SINR}_{\mathrm{MMSE}} = \mathrm{SNR} - \tfrac14\mathcal{F}(\mathrm{SNR}, \beta).
#
# On the left, the SINR of one user for 40 draws of random codes of 64 chips at
# :math:`\mathrm{SNR} = 10` dB, around the limits of
# :func:`comnum.cdma_sinr_large_system`. On the right, what it means for the total
# rate: the spectral efficiency (bits per chip) at :math:`E_b/N_0 = 10` dB, Fig. 1 of
# Verdú and Shamai. The conventional receiver saturates; the decorrelator has an
# optimum load near 0.65 and nothing left at :math:`K = N`; the MMSE receiver peaks
# near 0.75; only joint decoding keeps growing with the number of users.
N_RANDOM = 64
SNR_DB = 10.0
loads = np.arange(0.125, 1.51, 0.125)
load_fine = np.linspace(0.02, 2.0, 200)
rng = comnum.Rng(1999)

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4))
for color, (receiver, name, _marker) in zip(SERIES, receivers, strict=False):
    top = 0.98 if receiver == "zf" else 2.0
    fine = load_fine[load_fine <= top]
    left.plot(
        fine,
        [comnum.cdma_sinr_large_system(b, SNR_DB, receiver) for b in fine],
        color=color,
        linewidth=1.4,
        label=name,
    )
    for load in loads[loads <= (0.9 if receiver == "zf" else 1.5)]:
        k = int(load * N_RANDOM)
        draws = [
            10
            * np.log10(
                comnum.Dsss(rng.integers(2, k * N_RANDOM).reshape(k, N_RANDOM)).sinr(
                    1.0 / comnum.db2lin(SNR_DB), receiver
                )[0]
            )
            for _ in range(40)
        ]
        left.plot([load] * 40, draws, ".", color=color, markersize=2.5, alpha=0.5)
left.set(
    xlim=(0, 1.55),
    ylim=(-6, 11),
    xlabel=r"load $\beta = K/N$",
    ylabel="SINR (dB)",
    title=f"Random codes, $N$ = {N_RANDOM}, SNR = {SNR_DB:.0f} dB",
)
left.legend()
efficiency = [("optimal", "optimum (joint decoding)", MUTED)] + [
    (receiver, name, color) for color, (receiver, name, _) in zip(SERIES, receivers, strict=False)
]
for receiver, name, color in efficiency:
    right.plot(
        load_fine,
        comnum.cdma_spectral_efficiency(load_fine, receiver, ebn0_db=10.0),
        color=color,
        linewidth=1.4,
        label=name,
    )
right.set(
    xlim=(0, 2),
    ylim=(0, 3),
    xlabel=r"load $\beta = K/N$",
    ylabel="spectral efficiency (bits per chip)",
    title="Large systems, $E_b/N_0$ = 10 dB",
)
right.legend(fontsize="small")
fig.tight_layout()

for receiver, name, _ in receivers:
    c = comnum.cdma_spectral_efficiency(load_fine, receiver, ebn0_db=10.0)
    print(f"{name:>13}: at most {c.max():.2f} bit/chip, at K/N = {load_fine[c.argmax()]:.2f}")

# %%
# The rake receiver: echoes as diversity
# --------------------------------------
# One user on an m-sequence of 127 chips, over three Rayleigh paths of equal mean
# power delayed by 0, 2 and 5 chips, constant over a burst of 20 symbols. A narrowband
# signal would see the sum of the three paths, one Rayleigh gain; the spread signal
# resolves them, because the code is nearly orthogonal to its own shifts. A rake of
# one, two or three fingers that knows the gains then has the error rate of
# maximal-ratio combining of as many independent Rayleigh branches, each at a third
# of the total :math:`E_b/N_0` (:func:`comnum.theory_ber_rayleigh_mrc`): the slope of the
# curve, the diversity order, is the number of fingers.
link = comnum.Dsss(comnum.m_sequence(7))
DELAYS = np.array([0, 2, 5])
BURSTS, BURST_SYMBOLS = 6000, 20
rake_db = np.arange(0.0, 17.0, 4.0)
rake_fine = np.linspace(0, 16, 81)


def rake_ber(ebn0: float, tag: int) -> np.ndarray:
    """The BER with one, two and three fingers at the total Eb/N0."""
    rng = comnum.Rng.stream(9, "rake", tag, 0)
    errors = np.zeros(3)
    for _ in range(BURSTS):
        labels, symbols = bpsk_symbols((BURST_SYMBOLS,), rng)
        x = link.modulate(symbols)
        gains = rng.complex_noise(3, 1.0 / 3.0)
        received = rng.complex_noise(x.size + DELAYS[-1], 1.0 / comnum.db2lin(ebn0))
        for delay, gain in zip(DELAYS, gains, strict=True):
            received[delay : delay + x.size] += gain * x
        for fingers in (1, 2, 3):
            z = link.rake(received, DELAYS[:fingers], gains[:fingers])
            errors[fingers - 1] += np.count_nonzero(bpsk.decide(z) != labels)
    return errors / (BURSTS * BURST_SYMBOLS)


rake_sim = np.array([rake_ber(e, i) for i, e in enumerate(rake_db)])
per_path = 10 * np.log10(1.0 / 3.0)
fig, ax = plt.subplots()
ax.semilogy(rake_fine, comnum.theory_ber(2, rake_fine, "psk"), color=GREY, label="AWGN")
for fingers, color in zip((1, 2, 3), SERIES, strict=False):
    ax.semilogy(
        rake_fine,
        comnum.theory_ber_rayleigh_mrc(rake_fine + per_path, branches=fingers),
        color=color,
        linewidth=1.4,
    )
    ax.plot(
        rake_db,
        rake_sim[:, fingers - 1],
        "o",
        color=color,
        markeredgecolor=BACKGROUND,
        label=f"{fingers} finger{'s' if fingers > 1 else ''}",
    )
ax.plot([], [], color=MUTED, linewidth=1.4, label="maximal-ratio combining")
ax.set(
    ylim=(1e-5, 0.5),
    xlim=(0, 16),
    xlabel="total $E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="Rake receiver over three Rayleigh paths, $N$ = 127",
)
ax.legend()
fig.tight_layout()

# %%
# The rake stays on the curves of maximal-ratio combining as long as the noise
# dominates what the other paths leak through the code (:math:`1/127` of their
# amplitude here); a shorter code, or more paths, raises that floor.
for fingers in (1, 2, 3):
    expected = comnum.theory_ber_rayleigh_mrc(rake_db + per_path, branches=fingers)
    keep = expected * BURSTS * BURST_SYMBOLS >= 400
    gap = np.max(np.abs(rake_sim[keep, fingers - 1] / expected[keep] - 1))
    print(f"{fingers} finger(s): simulated within {100 * gap:.0f} % of maximal-ratio combining")
