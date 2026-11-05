# comnum-hardware

[![crates.io](https://img.shields.io/crates/v/comnum-hardware.svg)](https://crates.io/crates/comnum-hardware)
[![docs.rs](https://img.shields.io/docsrs/comnum-hardware)](https://docs.rs/comnum-hardware)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

Transmitter and receiver non-idealities: power amplifiers (Rapp, Saleh, memory polynomial) and the static predistorter of the Rapp model, Wiener phase noise and frequency offset, transmitter and receiver IQ imbalance; with the feature `optical`, lasers, phase, Mach–Zehnder and IQ modulators, photodiodes with shot and thermal noise, the 90° hybrid and the coherent receiver.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::hardware`. The API is documented on
[docs.rs/comnum-hardware](https://docs.rs/comnum-hardware), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
