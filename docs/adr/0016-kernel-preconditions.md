# 0016 — The preconditions of kernels and accessors panic; nothing a user reaches does

Status: accepted (2026-11-03) — refines `docs/ARCHITECTURE.md` ("Errors and panics")

## Context

The guide says that invalid user input returns an error and never panics. On
3 November 2026 the library code had 35 panics left (3 `panic!`, 15
`unreachable!`, 17 `assert!`):

- the 15 `unreachable!` and the `panic!` of `Checked` (a debug build only) are broken
  invariants: constructors of fixed codes, CRCs and constellations, or a checked state
  of a decoder or a block;
- the others are the length and index preconditions of the low-level kernels and
  accessors: `Fft::forward` and `inverse`, `fft::filter`, `linalg::matvec` and
  `matmul`, the steps of `SplitStep`, `manakov_rotation`, the arithmetic of
  `BigUint`, `Taps::coefficient`, `frequency_correlation`, the interleavers. They are
  called in the inner loops of blocks that check their inputs once, when they are built
  or applied (the rule "hot loops never re-validate"), and behave as the slices and
  matrices of `std` and `ndarray` do;
- two functions reachable from Python panicked on input that the user chooses:
  `Constellation::energy_with` (a probability vector of another length) and
  `LowRank::mse_floor` (a correlation of another size), which reached Python as a
  `PanicException` that `except Exception` does not catch.

## Decision

1. **A kernel or an accessor may state a precondition** on the lengths or indices of
   its arguments and panic when it is broken, documented under `# Panics`, when its
   callers check the inputs once outside the loop that calls it. Such a function
   validates nothing that a block above it does not check first.
2. **A function that takes what a user chooses returns an error**: a block, a
   constructor, a method of a closed form or an estimator, and everything the Python
   bindings and the specifications call with the user's values. `energy_with` and
   `mse_floor` now return `Result`.
3. **The bindings never panic.** `tests/test_no_panic.py` calls every function of the
   Python package and every method of an instance of every class with degenerate
   arguments (empty, NaN, infinite, negative, zero, of the wrong type or shape) and
   fails on a `PanicException`; it found the two functions above.

## Consequences

The kernels keep their signatures and their speed. A precondition is a documented
contract of the Rust API, not a path a user of Python or of a specification can take;
the probe of the bindings checks it, and a new class whose docstrings construct no
instance is added to its `EXTRA` table.
