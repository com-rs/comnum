# comnum-mapping

[![crates.io](https://img.shields.io/crates/v/comnum-mapping.svg)](https://crates.io/crates/comnum-mapping)
[![docs.rs](https://img.shields.io/docsrs/comnum-mapping)](https://docs.rs/comnum-mapping)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

QAM, PSK, PAM, OOK and custom constellations with Gray labels, hard decisions and LLRs (exact or max-log), differential M-PSK, the sources (PRBS of ITU-T O.150, shaped symbols) and the closed-form error and information rates.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::mapping`. The API is documented on
[docs.rs/comnum-mapping](https://docs.rs/comnum-mapping), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
