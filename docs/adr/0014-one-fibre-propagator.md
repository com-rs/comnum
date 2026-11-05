# 0014 — One split-step engine for the fibre

Status: accepted (2026-10-31) — changes `comnum_math::nlse` and the propagators of
`comnum_channel::optical` (`Ssfm`, `ManakovSsfm`, `ManakovPmd`)

## Context

The fibre had three forward propagators, each with its own scheme:

| | `Ssfm` (one polarisation) | `ManakovSsfm` (two) | `ManakovPmd` (two, PMD) |
|---|---|---|---|
| scheme | symmetric, midpoint phase | symmetric, midpoint or trapezoidal | asymmetric, one step per PMD section |
| step | fixed (`SplitStep`) | `StepRule` (Musetti et al. 2018 by default) | the section length |
| Raman gain | yes | yes | no |
| PMD | no | no | yes (coarse steps) |

Each was checked against references (`docs/adr/0013-manakov-defaults.md`), but they
could not be combined: PMD with nonlinearity used steps of a whole section, which the
step rule of Musetti et al. finds unconverged in WDM, and had no Raman gain; the step
rule existed for the Manakov solver only; and nothing guarded against the sampling and
sequence-length conditions under which a split step aliases the four-wave mixing
(Musetti et al. 2018, Table I and Appendix B; 0.3 dB of error found on
`specs/wdm_gn.yaml`, `validation/cards/poggiolini2012_band.yaml`).

## Decision

One engine, `comnum_math::nlse::ManakovSplitStep` generalised, propagates every forward
link:

- **fields**: one polarisation (the scalar NLSE, $\gamma$) or two (the Manakov equation,
  $\frac{8}{9}\gamma$);
- **nonlinear step**: `NonlinearStep::Midpoint` (default) or `Trapezoidal` (OptiCommPy);
- **steps**: one `StepRule` for all (`Fixed`, `NonlinearPhase`, `Fwm`); `Fixed(h)` covers
  a span with $\lceil L/h \rceil$ equal steps, as `Ssfm` did;
- **linear operator of a step**: loss and dispersion (scalar, in the frequency domain),
  the distributed gain of a Raman profile (`FieldGain`), and the birefringence of the
  PMD sections (`Birefringence`): the steps are cut at the section boundaries, the
  rotation of a section is applied at its start (frequency independent, per sample), and
  its DGD, diagonal in the current frame, is distributed along it as phase factors
  $e^{\mp j \omega \tau' \Delta z / 2}$ on the two polarisations in the linear half steps.
  Without nonlinearity this is the lumped response $\prod_{k} D_{k}(\omega) R_{k}$ of
  `PmdLink`;
- `comnum-math` knows the sections as rotations (`[C64; 4]`) and DGDs per kilometre only;
  `comnum-channel` builds them from a `PmdLink`.

The propagators become configurations of it: `Ssfm` (one polarisation, fixed steps by
default, OptiCommPy's `ssfm`), `ManakovSsfm` (two polarisations, now with an optional
`PmdLink`), and `ManakovPmd` (two polarisations with the PMD sections, the step rule and
the Raman gain of the engine; the former coarse-step scheme stays available as
`coarse_steps` for Marcuse et al. 1997). `SplitStep` and `SplitStepF32` stay as the
fixed-step kernels of the digital back-propagation and of the single-precision path,
checked equal to the engine with fixed steps.

The engine adds the third-order dispersion $\beta_{3}$ (`Medium::beta3_s3_km`), and a
sampling check of the launched field: its bandwidth $B$ (`signal_bandwidth`) against the
rate $\frac{3}{2} B + \frac{1}{2} B_{\mathrm{CUT}}$ and the sequence length against the
walk-off across $B$ (Musetti et al. 2018).

## Consequences

- Every forward propagation has the same scheme, step rule and checks; PMD,
  nonlinearity, Raman gain and converged steps combine.
- The references already checked are the tests of the engine: Hirota (1973) and
  Sinkin et al. (2003), Radhakrishnan et al. (1997), Chiang et al. (1996), Dar et al.
  (2014), Musetti et al. (2018), Poggiolini (2012), Gordon & Kogelnik (2000), Sunnerud
  et al. (2000), and the recorded outputs of OptiCommPy (`ssfm`, `manakovSSF`).
- `ManakovPmd` with the engine gives other results than its coarse steps (shorter
  steps, symmetric): the coarse steps remain selectable.
- The results of `Ssfm` and `ManakovSsfm` with their former settings do not change
  beyond rounding.
