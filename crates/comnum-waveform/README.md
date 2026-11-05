# comnum-waveform

[![crates.io](https://img.shields.io/crates/v/comnum-waveform.svg)](https://crates.io/crates/comnum-waveform)
[![docs.rs](https://img.shields.io/docsrs/comnum-waveform)](https://docs.rs/comnum-waveform)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

OFDM and DFT-spread OFDM with resource grids, pilots, LS and LMMSE channel estimation, Schmidl–Cox synchronisation and MIMO-OFDM detection; continuous-phase modulation (CPFSK, MSK, GMSK) with its detectors; constant-envelope OFDM and the theory of its error rates; the distribution of the PAPR and its reduction (clipping, SLM, PTS, tone reservation, ACE, companding).

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::waveform`. The API is documented on
[docs.rs/comnum-waveform](https://docs.rs/comnum-waveform), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
