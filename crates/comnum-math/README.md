# comnum-math

[![crates.io](https://img.shields.io/crates/v/comnum-math.svg)](https://crates.io/crates/comnum-math)
[![docs.rs](https://img.shields.io/docsrs/comnum-math)](https://docs.rs/comnum-math)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 0 (the foundation) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

The numerical methods without a communication meaning: decibel conversions, special functions, Gauss–Hermite quadrature, FFT plans, complex linear algebra (QR, Cholesky, SVD), nearest-point search, the split-step solvers of the nonlinear Schrödinger and Manakov equations, arithmetic over GF(2) and GF(2^m), exact integers, the Gray code, bisection and golden-section search.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::math`. The API is documented on
[docs.rs/comnum-math](https://docs.rs/comnum-math), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
