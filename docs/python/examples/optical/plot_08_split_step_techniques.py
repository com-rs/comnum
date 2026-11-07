"""
Split-step techniques: how many steps, and at what cost
=======================================================

The link is that of Fig. 8 of C. Häger and H. D. Pfister, "Physics-based deep learning
for fiber-optic communication systems", *IEEE J. Sel. Areas Commun.* 39(1), 2021
(arXiv:2010.14258): one channel of Gaussian symbols at 32 GBd, root-raised-cosine pulses
of roll-off 0.1 at 6 samples per symbol, frames of 1024 symbols, 10 spans of 100 km of
standard fibre (:math:`\\beta_2 = -21.683` ps²/km at 194.6 THz, :math:`\\gamma = 1.3`
/W/km, 0.2 dB/km), amplifiers of noise figure 5 dB. The receivers are those of the
figure: linear equalisation, and digital back-propagation (DBP) with 500 steps per span.

**The reference** is the split step with uniform steps and many of them: 500 steps per
span of 200 m. More steps change nothing: 2000 steps per span move its effective SNR by
:math:`10^{-7}` dB at 6 dBm. The card ``hager2021_convergence`` checks this convergence
in CI. Every other technique is measured by how much effective SNR it loses against the
reference, in dB.

**The first figure** runs the specification of Fig. 8
(``specs/validation/hager2021_fig8.yaml``, 8 frames per point) with only the forward
steps changed. The reference follows the published points within 0.04 dB (linear
equalisation) and 0.07 dB (DBP) up to 6 dBm. With 10 steps per span the DBP curve
falls: by 0.30 dB at 6 dBm and 0.88 dB at 8 dBm with uniform steps, by 1.07 and 2.13 dB
with logarithmic ones. A back-propagation that knows the fibre exactly removes the
nonlinearity of the true link, not that of a coarse simulation, so the error of the
forward steps stays in the SNR. Linear equalisation barely sees it: its nonlinear
interference is much larger than the error.

**The second figure** gives the SNR lost with the DBP receiver at 6 dBm, the peak of the
DBP curve, against the steps per span and against the computation time of the
propagation, for every technique. To compare them on the same noise, the forward models
run here without noise and the ASE of the link is added at the receiver; the losses
with 10 steps per span are those of the first figure within 0.1 dB (0.21 and 1.12 dB).
Every technique converges as the square of the step. For 0.01 dB:

- uniform steps with the phase at the middle of the step, the default: about 22 steps
  per span, 12 ms; the effective length of Optilux: about 20, 10 ms;
- single precision: the same 22 steps in 7 ms, with a floor near
  :math:`2 \\cdot 10^{-4}` dB;
- Bosco's logarithmic steps, those of Häger & Pfister: about 45 steps per span, 30 ms.
  They keep the spurious four-wave mixing of the split step low in WDM (Bosco et al.,
  *IEEE Photon. Technol. Lett.* 12(5), 2000), not the error of one channel; the
  paper's 500 take 325 ms;
- the trapezoidal mean of OptiCommPy: about 50 steps per span and four times the time
  per step (it iterates), 110 ms;
- the adaptive rules run in the Manakov solver, which carries two polarisations (the
  second empty), about twice the time per step: the four-wave-mixing rule of Musetti et
  al. (*J. Lightw. Technol.* 36(23), 2018) needs the fewest steps, about 16 per span,
  30 ms; the nonlinear-phase rule of OptiCommPy about 100, 420 ms. The default of
  :class:`comnum.ManakovSsfm` (20 rad, at most 0.02 rad of nonlinear phase per step)
  takes 107 steps per span and loses less than :math:`10^{-4}` dB.
"""

import re
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from _gallery_utils import SPECS

import comnum
from comnum.plot import AMBER, BACKGROUND, BLUE, DEEP_GREEN, GREEN, GREY, INK, ORANGE, PINK, VIOLET

RATE, SPS, SYMBOLS = 32e9, 6, 1024
FS = RATE * SPS
FIBER = dict(n_spans=10, span_km=100.0, alpha_db_km=0.2, gamma_w_km=1.3, d_ps_nm_km=17.20934,
             fc_hz=194.6e12, nf_db=5.0)  # fmt: skip
fiber = comnum.FiberParams(**FIBER)
# the Manakov solver with one polarisation empty is the scalar equation with gamma 9/8
manakov_fiber = comnum.FiberParams(**{**FIBER, "gamma_w_km": FIBER["gamma_w_km"] * 9 / 8})
SPAN = FIBER["span_km"]

# Gaussian symbols shaped by a root-raised cosine on the periodic block, unit power.
rng = np.random.default_rng(8)
symbols = (rng.standard_normal(SYMBOLS) + 1j * rng.standard_normal(SYMBOLS)) / np.sqrt(2)
shape = comnum.rrc_spectrum(SYMBOLS, SPS, 0.1)
unit = np.fft.ifft(np.tile(np.fft.fft(symbols), SPS) * shape)
unit /= np.sqrt(np.mean(np.abs(unit) ** 2))

# The noise of the 10 amplifiers in the band of the channel (W).
edfa = comnum.Edfa(gain_db=FIBER["alpha_db_km"] * SPAN, nf_db=FIBER["nf_db"], fc_hz=FIBER["fc_hz"])
ase_w = FIBER["n_spans"] * edfa.ase_psd * RATE
receiver_link = comnum.Ssfm(fiber, step_km=SPAN / 500)


def effective_snr_db(y, power_w):
    """DBP with 500 uniform steps per span, the matched filter, and the SNR against the
    sent symbols with the ASE of the link added."""
    z = receiver_link.backpropagate(y, FS, steps_per_span=500)
    s = np.fft.ifft(np.fft.fft(z) * shape)[::SPS]
    c = np.vdot(symbols, s) / np.vdot(symbols, symbols)
    distortion = np.mean(np.abs(s - c * symbols) ** 2) / (
        abs(c) ** 2 * np.mean(np.abs(symbols) ** 2)
    )
    return -10 * np.log10(distortion + ase_w / power_w)


def scalar(k, nonlinear_step="midpoint", precision="f64", logarithmic=False):
    """The scalar solver with `k` uniform (or logarithmic) steps per span, noiseless."""
    if logarithmic:
        ssfm = comnum.Ssfm(fiber, steps_per_span=k, amplification="ideal")
    else:
        ssfm = comnum.Ssfm(fiber, step_km=SPAN / k, nonlinear_step=nonlinear_step,
                           amplification="ideal")  # fmt: skip
    return lambda x: (ssfm.propagate(x, FS, precision=precision), k)


def manakov(**options):
    """The Manakov solver with its own step rule, one polarisation, noiseless."""
    solver = comnum.ManakovSsfm(manakov_fiber, amplification="ideal", **options)

    def run(x):
        y = solver.propagate(np.vstack([x, np.zeros_like(x)]), FS)[0]
        return y, solver.last_steps / FIBER["n_spans"]

    return run


def measure(technique, power_dbm, repeats=3):
    """The effective SNR, the steps per span and the best computation time (s)."""
    power_w = 1e-3 * 10 ** (power_dbm / 10)
    x = unit * np.sqrt(power_w)
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        y, steps = technique(x)
        best = min(best, time.perf_counter() - start)
    return effective_snr_db(y, power_w), steps, best


# %%
# The curves of Fig. 8, from its specification with only the forward steps changed:
# the reference (500 uniform steps per span), 10 uniform and 10 logarithmic steps.
text = (SPECS / "validation/hager2021_fig8.yaml").read_text(encoding="utf-8")
expect = {e["where"]["receiver"]: e["curve"] for e in yaml.safe_load(text)["expect"]}
body = text.split("\nexpect:")[0].replace("stop: {max_trials: 16}", "stop: {max_trials: 8}")
body = re.sub(r"cases:\n(  - .*\n)+", "cases:\n  - {receiver: linear, steps: 1, nonlinear: false}\n"
              "  - {receiver: dbp500, steps: 500, nonlinear: true}\n", body)  # fmt: skip
body = re.sub(
    r"  - \{type: dbp,.*\}", "  - {type: dbp, steps_per_span: $steps, nonlinear: $nonlinear}", body
)
forward = {
    "500 uniform steps per span (reference)": (INK, None, 2.4, "step_km: 0.2"),
    "10 uniform steps per span": (BLUE, "o", 1.2, "step_km: 10"),
    "10 logarithmic steps per span": (ORANGE, "D", 1.2, "steps_per_span: 10"),
}
curves = {}
fig, ax = plt.subplots()
for label, (color, marker, width, steps) in forward.items():
    results = comnum.run_spec(
        body.replace("{type: ssfm, steps_per_span: 500}", f"{{type: ssfm, {steps}}}")
    )
    for receiver, style, name in [
        ("linear", "--", "linear equalisation"),
        ("dbp500", "-", "DBP, 500 steps"),
    ]:
        x, snr = results.series("power_dbm", "snr_eff_db", receiver=receiver)
        curves[(label, receiver)] = dict(zip(x, snr, strict=True))
        ax.plot(x, snr, style, marker=marker, color=color, lw=width, markeredgecolor=BACKGROUND,
                label=f"{label}, {name}")  # fmt: skip
for receiver, marker, name in [
    ("linear", "x", "linear equalisation"),
    ("dbp500", "+", "DBP, 500 steps"),
]:
    c = expect[receiver]
    ax.plot(
        c["x"], c["y"], marker, color=GREY, mew=1.6, ms=8, label=f"Häger & Pfister, Fig. 8, {name}"
    )
ax.set(xlabel="launch power (dBm)", ylabel="effective SNR (dB)", ylim=(0, 32),
       title="10 × 100 km, 32 GBd (Häger & Pfister 2021, Fig. 8)")  # fmt: skip
ax.legend(loc="upper left", fontsize=6.5)
fig.tight_layout()

reference_label = "500 uniform steps per span (reference)"
for receiver in ("linear", "dbp500"):
    c, ref = expect[receiver], curves[(reference_label, receiver)]
    gaps = [abs(ref[x] - y) for x, y in zip(c["x"], c["y"], strict=True) if x <= 6]
    print(f"reference, {receiver}: within {max(gaps):.2f} dB of Fig. 8 up to 6 dBm")
for label in list(forward)[1:]:
    ref, coarse = curves[(reference_label, "dbp500")], curves[(label, "dbp500")]
    print(f"{label}, DBP: {ref[6.0] - coarse[6.0]:.2f} dB below the reference at 6 dBm, "
          f"{ref[8.0] - coarse[8.0]:.2f} dB at 8 dBm")  # fmt: skip

# %%
# The error of the steps at 6 dBm with the DBP receiver, for every technique.
POWER = 6.0
reference = measure(scalar(500), POWER)[0]
finer = measure(scalar(2000), POWER, repeats=1)[0]
print(f"2000 against 500 uniform steps per span: {finer - reference:+.1e} dB")
counts = [5, 10, 20, 50, 100, 200, 500]
sweeps = {
    "uniform, midpoint (default)": (BLUE, "o", [scalar(k) for k in counts]),
    "uniform, effective length (Optilux)": (GREEN, "s", [scalar(k, "effective_length") for k in counts]),
    "uniform, trapezoidal (OptiCommPy)": (AMBER, "^", [scalar(k, "trapezoidal") for k in counts]),
    "logarithmic (Bosco; Häger & Pfister)": (ORANGE, "D", [scalar(k, logarithmic=True) for k in counts]),
    "uniform, single precision": (VIOLET, "v", [scalar(k, precision="f32") for k in counts]),
    "nonlinear phase (OptiCommPy)": (PINK, "P", [
        manakov(step_rule="nonlinear_phase", max_nl_phase=phi, nonlinear_step="trapezoidal")
        for phi in [0.1, 0.03, 0.01, 0.003]
    ]),
    "four-wave mixing (Musetti et al.)": (DEEP_GREEN, "X", [
        manakov(step_rule="fwm", fwm_phase=phi, max_nl_phase=10.0) for phi in [40.0, 20.0, 10.0, 5.0, 2.0]
    ]),
}  # fmt: skip
rows = {label: [measure(t, POWER) for t in runs] for label, (_, _, runs) in sweeps.items()}
default = measure(manakov(), POWER)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
for label, (color, marker, _) in sweeps.items():
    snr, steps, seconds = np.array(rows[label]).T
    error = np.maximum(reference - snr, 1e-4)
    for ax, x in zip(axes, [steps, seconds], strict=True):
        ax.plot(x, error, marker=marker, color=color, markeredgecolor=BACKGROUND, label=label)
for ax, x in zip(axes, [default[1], default[2]], strict=True):
    ax.plot(x, max(reference - default[0], 1e-4), "*", ms=13, color=DEEP_GREEN, markeredgecolor=BACKGROUND,
            label="ManakovSsfm default" if ax is axes[0] else None)  # fmt: skip
for ax in axes:
    for level in (0.1, 0.01):
        ax.axhline(level, color=GREY, lw=1.0, ls="--")
    ax.set(xscale="log", yscale="log")
axes[0].text(4, 0.12, "0.1 dB", color=GREY, fontsize=8)
axes[0].text(4, 0.012, "0.01 dB", color=GREY, fontsize=8)
axes[0].set(xlabel="steps per span", ylabel="SNR lost against the reference (dB)", ylim=(1e-4, 10))
axes[1].set(xlabel="computation time of the propagation (s)")
fig.legend(
    *axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7.5, frameon=False
)
fig.suptitle(
    "The error of the steps at 6 dBm, DBP receiver (reference: 500 uniform steps per span)"
)
fig.tight_layout(rect=(0, 0.12, 1, 1))

# %%
# The table: steps per span, SNR lost against the reference, time of the propagation.
print(f"reference at {POWER:.0f} dBm, DBP: {reference:.3f} dB")
for label, measured in rows.items():
    print(label)
    for snr, steps, seconds in measured:
        print(f"  {steps:6.1f} steps/span  lost {reference - snr:7.4f} dB  {1e3 * seconds:7.1f} ms")
print(f"ManakovSsfm default: {default[1]:.1f} steps/span, lost {reference - default[0]:.4f} dB, "
      f"{1e3 * default[2]:.1f} ms")  # fmt: skip
