# 0009 — `comnum-waveform` depends on `comnum-mimo`

Status: accepted (2026-10-26) — new sibling dependency in layer 2

## Context

The precedent table of `docs/ARCHITECTURE.md` places per-subcarrier MIMO detection in
`comnum_waveform::ofdm::detection`, as a thin wrapper applying the detectors of
`comnum_mimo` across the resource grid. That wrapper needs a `comnum-waveform → comnum-mimo`
edge, which the crate table does not list.

## Decision

`comnum-waveform` depends on `comnum-mimo` for the per-resource-element detection of OFDM only:
`comnum_waveform::ofdm::detection` (the detectors of `comnum_mimo::detection` across the grid)
and `comnum_waveform::ofdm::link` (the receiver of `comnum_mimo::receiver`, whose
`Receiver::detect_uses` detects the data resource elements of a frame). The detectors
and the receiver stay waveform-agnostic in `comnum-mimo`; the resource mapping stays in
`comnum-waveform`.

## Consequences

`comnum-mimo` must never depend on `comnum-waveform` (it does not: it depends on layer 0 and
on `comnum-mapping`, ADR 0008). Layer 2 now has four sibling edges, all
pointing to `comnum-mapping` or `comnum-mimo`, both free of waveform knowledge.
