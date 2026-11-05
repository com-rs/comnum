# comnum-metrics

[![crates.io](https://img.shields.io/crates/v/comnum-metrics.svg)](https://crates.io/crates/comnum-metrics)
[![docs.rs](https://img.shields.io/docsrs/comnum-metrics)](https://docs.rs/comnum-metrics)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

Error rates, SNR, EVM, the mutual information and the generalised mutual information, PAPR and CCDF, power spectral density (Welch, aligned periodogram) and ACLR.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::metrics`. The API is documented on
[docs.rs/comnum-metrics](https://docs.rs/comnum-metrics), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
