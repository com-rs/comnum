# 0003 — One error type shared by the crates

Status: accepted (2026-10-26) — exception to `docs/ARCHITECTURE.md` ("Errors and panics")

## Context

The guide asks for one `Error` enum per crate. A simulation trial calls blocks of many
crates and returns one `Result`; with an enum per crate, every crate above layer 0
would wrap the enums below it, and user code would match nested variants. All the
failures of comnum are of five kinds: invalid configuration, inconsistent shapes,
numerical failure, missing condition value, input/output.

## Decision

`comnum_core::Error` (`thiserror`, `#[non_exhaustive]`) with the variants `Config`,
`Shape`, `Numerical`, `Condition` and `Io` is the error of every crate, re-exported as
`comnum::Error`. No `String` errors, no `Box<dyn Error>` in public signatures.

## Consequences

- `?` composes across crates without conversions.
- `Condition` is a simulation concept that lives in `comnum-core`; acceptable for one
  variant.
- A crate that needs a kind of its own (a decoder failing to converge, say) adds a
  variant to `comnum_core::Error` (allowed by `#[non_exhaustive]`) or introduces its own
  enum with an ADR.
