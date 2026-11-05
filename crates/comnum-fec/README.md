# comnum-fec

[![crates.io](https://img.shields.io/crates/v/comnum-fec.svg)](https://crates.io/crates/comnum-fec)
[![docs.rs](https://img.shields.io/docsrs/comnum-fec)](https://docs.rs/comnum-fec)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

Channel coding: the CRCs and the scrambler of 5G NR, block, random and QPP interleavers, convolutional codes (Viterbi, BCJR) and their puncturing, the LTE turbo code, LDPC codes (PEG, repeat-accumulate, alist files; sum-product, min-sum, layered decoding), polar codes (SC, CRC-aided SCL), Reed–Solomon and binary BCH codes, constant-composition distribution matching, and the BI-AWGN capacity and normal approximation.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::fec`. The API is documented on
[docs.rs/comnum-fec](https://docs.rs/comnum-fec), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
