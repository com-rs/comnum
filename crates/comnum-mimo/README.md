# comnum-mimo

[![crates.io](https://img.shields.io/crates/v/comnum-mimo.svg)](https://crates.io/crates/comnum-mimo)
[![docs.rs](https://img.shields.io/docsrs/comnum-mimo)](https://docs.rs/comnum-mimo)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

MIMO detection (ZF, MMSE, V-BLAST SIC, ML, sphere decoder, K-best) with soft outputs, the receiver of a link (spatial layers or Alamouti blocks), orthogonal space-time block codes (Alamouti, rate 3/4), eigenmode and zero-forcing precoding, water filling and capacity.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::mimo`. The API is documented on
[docs.rs/comnum-mimo](https://docs.rs/comnum-mimo), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
