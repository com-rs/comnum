# 0015 — Adaptive algorithms: error laws, update rules and loop filters

Status: accepted (2026-11-02) — adds `comnum_dsp::adaptive`, changes the internals of
`comnum_dsp::equalizer::AdaptiveEqualizer`, `comnum_dsp::sync::timing::GardnerLoop` and
`comnum_dsp::sync::carrier::ddpll`; no result changes

## Context

The adaptive algorithms of the library were written case by case:

- the butterfly equaliser hard-coded four error laws (CMA, RDE, NLMS, DD-LMS) in a
  private `ErrorLaws` struct, and the normalisation of NLMS inside its update loop;
- the Gardner timing loop and the decision-directed PLL each wrote their own loop
  filter (a proportional-integral filter, and its bilinear discretisation).

Every adaptive filter is the same stochastic gradient, $h \leftarrow h + \mu_{k}\, g(e)\,
x^{*}$, where two choices are independent: what is minimised (the error law) and how far
the taps move (the update rule). Every tracking loop is a detector, a loop filter and an
oscillator. Writing these pieces once lets a new algorithm be a new combination (a
sign-error CMA, a CMA normalised by the input energy, the RLS-CMA of Fatadin et al.
2009 later) and lets the coming receivers (decision-directed channel tracking of OFDM,
the DVB-T and NR receivers) reuse them instead of a third copy.

## Decision

`comnum_dsp::adaptive` holds:

- the trait `ErrorLaw` (`error(y, stream, k) -> (e, |e|²)`) and its laws
  `ConstantModulus`, `RadiusDirected`, `Reference`, `Decision`;
- the `UpdateRule` (`Lms`, `Normalized`, `SignError`) and its `update` of the taps;
- the trait `LoopFilter` and the filters `ProportionalIntegral` and `BilinearPi`.

`AdaptiveEqualizer` keeps its public interface: its `Algorithm`s map to the usual pairs
(CMA and RDE with LMS, NLMS on a reference, DD-LMS); `run_with` runs any law with any rule;
the field `update_rule` overrides the rule of every stage (block parameter `update_rule`,
Python `update_rule=`). `GardnerLoop` uses `ProportionalIntegral`, `ddpll` uses
`BilinearPi::active_pi`.

Two traits rather than one: a filter updates a vector of taps from a complex error, a
loop updates a phase or an instant from a real one; a common interface would be
artificial. The laws are called through `&dyn ErrorLaw` once per output, outside the loop
over the taps (the cost is unchanged); the update rule is an enum.

## Consequences

- The order of the operations of every update is kept (ADR 0005): the outputs of the
  equaliser (symbols, taps and errors of the four algorithms in 1 × 1 and 2 × 2, a
  schedule with Kikuchi's initialisation), of the Gardner loop (both detectors) and of
  the DD-PLL (with and without pilots), 33 arrays on fixed inputs, are bit-identical
  before and after the change; the comparisons with OptiCommPy are unchanged.
- The sign-error rule is new: `update_rule: sign` in a specification, with its unit test.
- A new error law or loop filter is a type implementing the trait, documented with its
  reference, not a change of the equaliser or of a loop.
