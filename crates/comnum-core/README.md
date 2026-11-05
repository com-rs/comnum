# comnum-core

[![crates.io](https://img.shields.io/crates/v/comnum-core.svg)](https://crates.io/crates/comnum-core)
[![docs.rs](https://img.shields.io/docsrs/comnum-core)](https://docs.rs/comnum-core)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 0 (the foundation) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

The vocabulary every block of comnum shares: the multi-stream containers (`Bits`, `Llr`, `Symbols`, `Signal`, `DualPol`), the verb traits (`Map`, `Demap`, `Propagate`, `Impair`, …), the unit newtypes (`Db`, `EbN0`, `EsN0`, `Hertz`), the seeded generator `SimRng` (xoshiro256\*\*, one stream per trial) and the error type.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::core`. The API is documented on
[docs.rs/comnum-core](https://docs.rs/comnum-core), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
