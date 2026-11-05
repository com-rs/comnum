# comnum-channel

[![crates.io](https://img.shields.io/crates/v/comnum-channel.svg)](https://crates.io/crates/comnum-channel)
[![docs.rs](https://img.shields.io/docsrs/comnum-channel)](https://docs.rs/comnum-channel)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

Propagation media: AWGN, flat Rayleigh and Rician fading (SISO and MIMO, Kronecker correlation), tapped delay lines (EPA, EVA, ETU, ITU-R profiles) and the clustered delay lines CDL-A to CDL-E of 3GPP TR 38.901; optical fibre with the feature `optical`: linear fibre, split-step Fourier, Manakov-PMD, EDFA, distributed Raman amplification and inter-channel stimulated Raman scattering, the GN and EGN models of the nonlinear interference.

The crate ships the Raman gain spectrum of GNPy (BSD 3-Clause): see [`THIRD_PARTY.md`](https://github.com/com-rs/comnum/blob/main/THIRD_PARTY.md), included in the package.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::channel`. The API is documented on
[docs.rs/comnum-channel](https://docs.rs/comnum-channel), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
