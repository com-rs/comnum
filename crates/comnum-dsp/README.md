# comnum-dsp

[![crates.io](https://img.shields.io/crates/v/comnum-dsp.svg)](https://crates.io/crates/comnum-dsp)
[![docs.rs](https://img.shields.io/docsrs/comnum-dsp)](https://docs.rs/comnum-dsp)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 1 (generic signal processing) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

Waveform-agnostic signal processing: pulse shapes and FIR filtering, resampling, carrier, timing and frame synchronisation, the adaptive n×n butterfly equaliser (CMA, RDE, NLMS, DD-LMS), LS and LMMSE estimation, chromatic-dispersion compensation, digital back-propagation and digital predistortion.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::dsp`. The API is documented on
[docs.rs/comnum-dsp](https://docs.rs/comnum-dsp), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
