"""
CCSDS: the codes of telemetry against the normal approximation
===============================================================

CCSDS 131.0-B-5 (TM Synchronization and Channel Coding, 2023) specifies the channel codes
of space telemetry: a convolutional code (Sec. 3), interleaved Reed–Solomon codeblocks
(Sec. 4), turbo codes (Sec. 6) and LDPC codes (Sec. 7), then the markers and the
randomizer that frame them on the channel (Secs. 9 and 10). comnum reproduces every table
the standard prints exactly (card ``ccsds131_exact``): the dual basis of Table F-1, the
generators of Annex G, the lengths of Tables 6-2 and 7-5, the parity checks of the LDPC
codes, the first bits of the randomizers.

**The standard gives no error-rate curve.** The curves are in the Green Book CCSDS
130.1-G, which comnum does not have yet. This script is therefore a check of
self-consistency, not a validation: it draws the frame error rates of the AR4JA codes of
:math:`k = 1024` bits and of the turbo code of :math:`k = 1784` bits over BPSK and AWGN,
next to the normal approximation of Polyanskiy, Poor & Verdú (2010), the smallest frame
error rate any code of the same length and rate can reach (to second order). A good
code sits a fraction of a dB to the right of it; a code built wrongly sits far away, or
never falls. With 400 frames per point, the three codes reach a frame error rate of
:math:`10^{-2}` 0.6 to 0.8 dB to the right of the approximation (about 1.45, 3.15 and
0.55 dB against 0.69, 2.58 and −0.11 dB).
"""

import matplotlib.pyplot as plt
import numpy as np

import comnum
from comnum.plot import BACKGROUND, BLUE, RUST, TEAL

# %%
# Frame error rates: the information bits are random, the decoders run 50 iterations of
# belief propagation (LDPC) or 8 of log-MAP (turbo), 400 frames per point at most and
# stop after 40 frame errors.
rng, noise = comnum.Rng(131), np.random.default_rng(131)


def fer(code, ebn0_db, frames=400, max_errors=40):
    """The frame error rate of ``code`` over BPSK and AWGN at every Eb/N0."""
    out = []
    for e in ebn0_db:
        noise_var = comnum.bpsk_noise_variance(e, code.k / code.n)
        errors = sent = 0
        while sent < frames and errors < max_errors:
            b = comnum.bits(code.k, rng)
            y = 1.0 - 2.0 * code.encode(b) + np.sqrt(noise_var) * noise.normal(size=code.n)
            errors += int(np.any(code.decode(comnum.bpsk_llr(y, noise_var)) != b))
            sent += 1
        out.append(errors / sent if errors else np.nan)
    return np.array(out)


codes = [
    (comnum.CcsdsLdpc("1/2", 1024), np.arange(0.8, 1.81, 0.2), BLUE, "AR4JA 1/2"),
    (comnum.CcsdsLdpc("4/5", 1024), np.arange(2.4, 3.61, 0.2), RUST, "AR4JA 4/5"),
    (comnum.CcsdsTurbo(1784, "1/3"), np.arange(0.0, 0.81, 0.2), TEAL, "turbo 1/3"),
]
fig, ax = plt.subplots()
for code, ebn0, color, label in codes:
    ax.semilogy(ebn0, fer(code, ebn0), marker="o", color=color, markeredgecolor=BACKGROUND,
                label=f"{label}, ({code.n}, {code.k})")  # fmt: skip
    fine = np.linspace(ebn0[0] - 1.0, ebn0[-1], 60)
    ax.semilogy(fine, comnum.theory_normal_approximation(code.n, code.k, fine), color=color,
                linestyle="--", linewidth=1)  # fmt: skip
ax.set(xlabel="Eb/N0 (dB)", ylabel="frame error rate", ylim=(2e-3, 1),
       title="CCSDS codes, BPSK, AWGN; dashed: normal approximation")  # fmt: skip
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# **On the channel.** A codeword leaves after its attached sync marker, randomized by the
# sequence of Sec. 10.4 (the marker is not). The receiver negates the LLRs where the
# sequence is one; here the 40 first bits of the short sequence, ``FF 48 0E C0 9A``, as
# NOTE 2 of Sec. 10.4.3 prints them.
cadu = comnum.ccsds_cadu(comnum.ccsds_marker("asm"), np.zeros(40, np.uint8), "short")
print(np.packbits(cadu).tobytes().hex())
