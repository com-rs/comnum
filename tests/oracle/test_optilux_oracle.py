"""comnum's split-step against the ``fiber`` of Optilux (GPL-3.0, Octave), recorded.

``tools/record_optilux.py`` runs Optilux (Paolo Serena, University of Parma, the release
of April 2021) under Octave on the input field of the split-step scenario of the
library benchmark (``benchmarks/libraries/s5_ssfm.py`` at its quick size: 16-QAM at
32 GBd, root-raised cosine of roll-off 0.1, 4 samples per symbol, 4 dBm, 2¹¹ symbols)
over the link of that scenario (80 km spans, alpha 0.2 dB/km, D 17 ps/nm/km, no slope,
gamma 1.3 /W/km, a fixed symmetric step of 1 km, noiseless amplification of alpha L at
the end of every span) and records, in ``tests/oracle/optilux_ssfm.npz`` (the provenance
in the ``.json``), the input and the output fields of three cases: one span without
Kerr effect, one span with it, and the ten spans of the benchmark.

What the comparison shows, with ``comnum.Ssfm`` at the same step on the same field:

- the linear span agrees to machine precision (NMSE of the fields below 10⁻²⁴): same
  dispersion operator, same attenuation, same amplifier;
- the Kerr spans differ by the nonlinear step: comnum rotates the phase by the power at
  the middle of the step over its length, Optilux by the power at the start over the
  effective length :math:`(1 - e^{-\\alpha h}) / \\alpha` of the step, which weighs the
  attenuation within it ([Sha14] in its help). At 1 km the two differ by an NMSE of
  about 10⁻¹⁰ after one span and 1.5 · 10⁻⁸ after ten, both within 4 · 10⁻⁸ of the
  step-converged solution (comnum at 0.05 km): Optilux's scheme is the closer of the
  two at this step (1.1 · 10⁻⁸ against 3.7 · 10⁻⁸).

Conventions: Optilux writes the nonlinear Schrödinger equation with the opposite sign
of the imaginary unit, so the recorder propagated the conjugate of the field and
conjugated the result; its fields are in sqrt(mW), converted to sqrt(W).
"""

import json

import numpy as np
import pytest

import comnum
from oracle import DIR, missing

if not (DIR / "optilux_ssfm.npz").exists():
    missing("tests/oracle/optilux_ssfm.npz is missing: run tools/record_optilux.py", module=True)
V = dict(np.load(DIR / "optilux_ssfm.npz"))
META = json.loads((DIR / "optilux_ssfm.json").read_text(encoding="utf-8"))
LINK = META["link"]
REF_STEP_KM = 0.05


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


def propagate(gamma_w_km: float, n_spans: int, step_km: float = LINK["step_km"]) -> np.ndarray:
    fibre = comnum.FiberParams(
        alpha_db_km=LINK["alpha_db_km"],
        d_ps_nm_km=LINK["d_ps_nm_km"],
        gamma_w_km=gamma_w_km,
        span_km=LINK["span_km"],
        n_spans=n_spans,
        fc_hz=LINK["fc_hz"],
    )
    link = comnum.Ssfm(fibre, step_km=step_km, amplification="ideal")
    return link.propagate(V["input"], LINK["fs_hz"], precision="f64")


def test_the_recorded_steps_are_those_of_the_benchmark():
    """80 steps of 1 km per span with the Kerr effect; without it Optilux takes the
    whole span in one step, which is exact (dispersion and attenuation commute)."""
    for case, steps in META["steps_per_span"].items():
        spans = META["cases"][case]["spans"]
        per_span = (
            int(LINK["span_km"] / LINK["step_km"]) if META["cases"][case]["gamma_scale"] else 1
        )
        assert steps == [per_span] * spans, case


def test_the_linear_span_agrees_to_machine_precision():
    ours = propagate(0.0, 1)
    assert nmse(V["linear_1span"], ours) < 1e-24


@pytest.mark.parametrize(
    ("case", "spans", "bound"), [("kerr_1span", 1, 1e-9), ("kerr_10spans", 10, 1e-7)]
)
def test_the_kerr_spans_agree_within_the_nonlinear_step(case, spans, bound):
    ours = propagate(LINK["gamma_w_km"], spans)
    assert nmse(V[case], ours) < bound


def test_both_solvers_are_within_the_step_error_of_the_converged_solution():
    ref = propagate(LINK["gamma_w_km"], 10, REF_STEP_KM)
    ours = propagate(LINK["gamma_w_km"], 10)
    assert nmse(V["kerr_10spans"], ref) < 4e-8
    assert nmse(ours, ref) < 4e-8
