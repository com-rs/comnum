# 0004 — The crate `comnum-spec` (layer 4)

Status: accepted (2026-10-26) — new crate

## Context

comnum runs declarative campaigns: a YAML, JSON or TOML file describes the grid, the
link as a list of blocks, the metrics, the stopping rule and the expected results
(`comnum run`, `comnum verify`, `comnum.run_spec` in Python). This engine, its block
registry, the recordings I/O (SigMF, raw IQ, NPY) and the `comnum` binary are not in
the crate table of the guide.

## Decision

They form the crate `comnum-spec`, in layer 4 next to `comnum-systems`: it assembles blocks
of layers 0 to 2 into links and runs them with `comnum-sim`, which is what an integration
crate does. It depends on layers 0 to 3 and nothing depends on it but the facade
(`comnum::spec`, `comnum::io`, feature `spec`) and the bindings. Its binary keeps the name
`comnum`. Blocks of other projects join its registry from their own crates (ADR 0006).

## Consequences

- `serde`, `serde_json`, `serde_norway` (the YAML reader, a fork of `serde_yaml`, which
  is no longer maintained) and `toml` stay out of the lower layers.
- A block becomes available to specifications by an entry of
  `comnum_spec::blocks::registry`; the algorithm itself is placed by the placement rules.
- The recordings I/O lives here because the specifications are its only user; it moves
  down if a lower crate needs it.
