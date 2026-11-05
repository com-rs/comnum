# comnum-systems

[![crates.io](https://img.shields.io/crates/v/comnum-systems.svg)](https://crates.io/crates/comnum-systems)
[![docs.rs](https://img.shields.io/docsrs/comnum-systems)](https://docs.rs/comnum-systems)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 4 (integration) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

Reference chains of standards and assemblies of several link-block crates: the physical layer of DVB-S2 (ETSI EN 302 307-1), from the BBFRAME to the scrambled PLFRAME and back, with its BCH and LDPC codes, the 28 MODCODs, normal and short frames and pilots; probabilistic amplitude shaping (Böcherer, Steiner & Schulte 2015) with the CCDM and the LDPC codes of `comnum-fec`.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::systems`. The API is documented on
[docs.rs/comnum-systems](https://docs.rs/comnum-systems), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
