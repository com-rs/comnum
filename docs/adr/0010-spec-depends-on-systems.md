# 0010 — `comnum-spec` depends on `comnum-systems`

Status: accepted (2026-10-27) — new edge between two crates of layer 4

## Context

`comnum-systems` holds the standard-compliant chains (`comnum_systems::dvbs2`), as the guide
requires: what only makes sense for one standard goes there, not to the functional
crates. The declarative campaigns of `comnum-spec` must run these chains as blocks
(`dvbs2_tx`, `dvbs2_rx`), and both crates sit in layer 4, whose crates do not depend on
each other.

## Decision

`comnum-spec` depends on `comnum-systems`. The blocks of `crates/comnum-spec/src/blocks/systems.rs`
are thin wrappers that parse their parameters and call the chains; the chains keep no
knowledge of the specifications, frames or metrics of `comnum-spec`.

## Consequences

`comnum-systems` must never depend on `comnum-spec`: it depends on layers 0 to 3 only. The
facade `comnum` re-exports both. A standard chain becomes available to specifications by
a block of `blocks/systems.rs`, like any other block (ADR 0004).
