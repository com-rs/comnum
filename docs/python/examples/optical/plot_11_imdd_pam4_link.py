"""
IM-DD PAM4 over dispersive fibre: FFE, DFE and MLSE
===================================================

A short-reach intensity-modulated, directly detected link: PAM4 at 25 GBd from a
directly modulated laser (:class:`comnum.DirectlyModulatedLaser`, linewidth
enhancement factor 3), standard single-mode fibre with its dispersion and dispersion
slope (:class:`comnum.FiberParams`, :math:`D = 17` ps/(nm·km), :math:`S = 0.06`
ps/(nm²·km)), a p-i-n photodiode (:class:`comnum.Photodiode`, shot and thermal noise,
0.7 of the symbol rate of bandwidth), and three receivers at one sample per symbol: a
feedforward equaliser of 21 taps (:func:`comnum.adaptive_equalizer` on one stream, NLMS
on 4000 training symbols then decision-directed), a decision-feedback equaliser of 21
feedforward and 3 feedback taps (:func:`comnum.dfe`) and maximum-likelihood sequence
estimation (:func:`comnum.mlse`) over the three taps around the symbol that a
least-squares fit of the training symbols gives.

Square-law detection of a chirped, dispersed field is not a linear channel in the
symbols: the power fading of the chirp–dispersion interaction (Agrawal 2002,
Sec. 5.4.4) grows with the distance and the equalisers differ in how they survive
it. The figure reports the symbol error rate against the distance, with the error
rate of an ideal receiver on the back-to-back channel (:func:`comnum.theory_ser`) as
the floor the noise alone would give.
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BLUE, GREEN, GREY, ORANGE

RS, SPS, N_SYMBOLS = 25e9, 4, 20_000
FS = RS * SPS
fiber = comnum.FiberParams(d_ps_nm_km=17.0, slope_ps_nm2_km=0.06, fc_hz=193.4e12)
c = comnum.Constellation("pam", 4)
dml = comnum.DirectlyModulatedLaser(linewidth_enhancement=3.0, bandwidth_hz=0.8 * RS)
pd = comnum.Photodiode(responsivity=0.8, bandwidth_hz=0.7 * RS, n_taps=63, filter_kind="gauss")
P_MEAN, EXTINCTION_DB = 2e-3, 6.0  # 3 dBm into the fibre, 6 dB of extinction ratio
rng = comnum.Rng(7)
idx, s = comnum.symbols(c, N_SYMBOLS, rng)

# the drive: four power levels between P_min and P_max = r P_min, r the extinction ratio
r_ex = 10 ** (EXTINCTION_DB / 10)
p_min = 2 * P_MEAN / (1 + r_ex)
levels = p_min + (p_min * (r_ex - 1)) * (s.real + 3) / 6
drive = np.repeat(levels, SPS)  # NRZ
field = dml.field(drive, FS)


def receive(distance_km):
    """One sample per symbol of the photocurrent after the fibre, and the noise variance."""
    e = comnum.linear_fiber_channel(field, fiber, distance_km, FS) if distance_km > 0 else field
    i = pd.detect_power(np.abs(e) ** 2, FS, rng)
    # centre of the symbol, then an AC-coupled, unit-power real signal
    x = i.reshape(-1, SPS)[:, SPS // 2]
    x = (x - x.mean()) / x.std()
    return x.astype(complex)


def ser(decisions, offset=1000):
    return float(np.mean(decisions[offset:] != idx[offset:]))


TRAIN = 4000
cn = c.normalized()  # the equalisers work on unit-power signals and points
sn = cn.symbols(idx)
distances = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
results = {"FFE (21 taps)": [], "DFE (21 + 3 taps)": [], "MLSE (3 taps)": []}
for d in distances:
    x = receive(d)
    y, _, _ = comnum.adaptive_equalizer(
        x,
        cn,
        algorithm=[("nlms", TRAIN), ("dd-lms", 0)],
        n_taps=21,
        sps=1,
        mu=[2e-3, 5e-4],
        reference=sn,
    )
    results["FFE (21 taps)"].append(ser(cn.decide(y[0])))
    _, dec, _, _, _ = comnum.dfe(
        x, cn, n_feedforward=21, n_feedback=3, mu=2e-3, reference=sn, training=TRAIN
    )
    results["DFE (21 + 3 taps)"].append(ser(dec))
    # the MLSE needs the channel: a least-squares fit of 3 taps around the symbol (one
    # precursor, one post-cursor) on the training symbols, then r_j = x_{j-1}
    a = np.stack([np.roll(sn, lag) for lag in [-1, 0, 1]], axis=1)[TRAIN // 2 : TRAIN]
    taps, *_ = np.linalg.lstsq(a, x[TRAIN // 2 : TRAIN], rcond=None)
    dec = np.append(comnum.mlse(x[:-1], taps, cn), 0)
    results["MLSE (3 taps)"].append(float(np.mean(dec[1000:-1] != idx[1001:])))

fig, ax = plt.subplots()
for (name, v), color in zip(results.items(), [BLUE, ORANGE, GREEN], strict=True):
    ax.semilogy(distances, np.maximum(v, 1e-5), "o-", color=color, label=name)
# the noise floor: the error rate of the back-to-back samples through a slicer
x0 = receive(0.0)
snr_db = comnum.snr_db(x0[TRAIN:], s[TRAIN:])
ax.axhline(comnum.theory_ser_esn0(4, snr_db, "pam"), color=GREY, linestyle=":",
           label=f"slicer, SNR {snr_db:.1f} dB")  # fmt: skip
ax.set(xlabel="distance (km)", ylabel="symbol error rate", ylim=(1e-5, 1),
       title="PAM4 at 25 GBd, DML (β_c = 3), SSMF, p-i-n")  # fmt: skip
ax.legend()
fig.tight_layout()
for name, v in results.items():
    print(f"{name}: " + ", ".join(f"{e:.1e}" for e in v))

# %%
# Soft outputs for a channel decoder
# ----------------------------------
#
# On–off keying over the same fibre with the (133, 171) convolutional code: the
# MLSE returns the reliability of every symbol (the soft-output Viterbi algorithm of
# Hagenauer & Hoeher 1989, :func:`comnum.mlse_soft`), which the decoder
# takes as LLRs; the hard decisions of the MLSE are decoded for comparison.

code = comnum.ConvolutionalCode(2000)
ook = comnum.Constellation("ook", 2)
info = comnum.bits(code.k * 4, rng)
coded = code.encode(info)
s_ook = ook.symbols(coded)
levels = p_min + (p_min * (r_ex - 1)) * s_ook.real
field = dml.field(np.repeat(levels, SPS), FS)
idx = coded.astype(np.int64)
for d in [10.0, 20.0]:
    x = receive(d)
    # a 3-tap channel around the symbol (one precursor), plus its DC, fitted on 1000 symbols
    lags = [-1, 0, 1]
    a = np.stack([np.roll(s_ook, lag) for lag in lags] + [np.ones(len(s_ook))], axis=1)[200:1200]
    fit, *_ = np.linalg.lstsq(a, x[200:1200], rcond=None)
    taps, dc = fit[:3], fit[3]
    r = (x - dc)[:-1]  # r_j = x_{j-1} = Σ_m taps[m] s_{j-m}
    noise_var = 2 * float(np.var(x[200:1200] - a @ fit))  # real noise: twice its variance
    hard, rel = comnum.mlse_soft(r, taps, ook, noise_var)
    llr = np.where(hard == 0, rel, -rel)  # positive for a 0, as the decoder takes them
    uncoded = float(np.mean(hard != idx[1:]))
    # decision j is the symbol j + 1: the first coded bit gets no information
    soft_ber = comnum.ber(code.decode(np.concatenate(([0.0], llr))), info)
    hard_ber = comnum.ber(code.decode(np.concatenate(([0.0], 1.0 - 2.0 * hard))), info)
    print(f"{d:.0f} km: uncoded BER {uncoded:.1e}, decoded from the SOVA reliabilities {soft_ber:.1e}, "
          f"from hard decisions {hard_ber:.1e}")  # fmt: skip
