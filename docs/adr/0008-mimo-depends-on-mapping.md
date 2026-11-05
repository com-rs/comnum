# 0008 — `comnum-mimo` depends on `comnum-mapping`

Status: accepted (2026-10-26) — new sibling dependency in layer 2

## Context

The guide places the MIMO detectors in `comnum_mimo::detection`, operating on the generic
model $y = H x + w$, and lets `comnum-mimo` depend on layers 0 and 1. A detector works on
an alphabet: the tree searches enumerate its points, the linear detectors decide on it
and their soft outputs need its bit labels; the space-time decoders hand their
estimates to the same decisions and LLRs.

## Decision

`comnum-mimo` depends on `comnum-mapping` and takes a `Constellation` (points, Gray labels,
nearest point, LLRs). The linear algebra the detectors need (QR, Cholesky, SVD) is
added to `comnum_math::linalg`, where every crate can reuse it. The MIMO channel models
stay in `comnum-channel` (`fading::Mimo`, `spatial`), as the guide places them.

## Consequences

Layer 2 has three sibling edges, all towards `comnum-mapping`, which stays a leaf of the
layer: `comnum-waveform`, `comnum-metrics` (ADR 0007) and `comnum-mimo`.
