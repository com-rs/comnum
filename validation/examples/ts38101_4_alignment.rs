//! An exploration, not a validation: test 1-2 of TS 38.101-4 under the conditions of the
//! RAN4 alignment (R4-1811723: the TR 38.901 channel without simplification, a transmitter
//! EVM of 6 %, the TRS in symbols 4 and 8) and with the receivers that bound it (an
//! estimator designed for the 2.6 us of TDLC300, the channel known), whose results are
//! recorded in the findings of the card `ts38101_4_pdsch`:
//!
//! ```text
//! cargo run --release -p comnum-validation --example ts38101_4_alignment
//! ONLY_GENIE=1 cargo run --release -p comnum-validation --example ts38101_4_alignment
//! ```
#![allow(clippy::unwrap_used, reason = "an exploration: a failure stops it")]

use comnum::channel::multipath::Profile;
use comnum_validation::ts38101_4::{Conditions, Rx, TS_38101_4, throughput};

fn main() {
    let tdlc = Profile::TdlC { delay_spread_s: 300e-9 };
    let only_genie = std::env::var("ONLY_GENIE").is_ok();
    for (n_rx, start) in if only_genie { vec![(2, -6.0), (4, -9.5)] } else { vec![(2, -4.0), (4, -7.0)] } {
        let grid: Vec<f64> = (0..9).map(|i| start + 0.5 * f64::from(i)).collect();
        for (name, c) in [
            ("TS 38.101-4", TS_38101_4),
            ("TR 38.901 TDL-C 300 ns", Conditions { profile: tdlc, ..TS_38101_4 }),
            ("+ EVM 6 %", Conditions { profile: tdlc, evm: 0.06, ..TS_38101_4 }),
            ("+ TRS 4, 8", Conditions { profile: tdlc, evm: 0.06, trs_symbols: [4, 8], ..TS_38101_4 }),
            ("estimator for 2.6 us", Conditions { receiver: Rx::Estimated { max_delay_s: 2.6e-6 }, ..TS_38101_4 }),
            ("genie channel", Conditions { receiver: Rx::Genie, ..TS_38101_4 }),
        ] {
            if only_genie && !name.starts_with("genie") {
                continue;
            }
            let t: Vec<f64> = grid.iter().map(|&s| throughput(c, n_rx, s, 20, 200, 38_101).unwrap()).collect();
            let i = t.windows(2).position(|w| w[0] < 0.7 && w[1] >= 0.7).unwrap();
            let s70 = grid[i] + (0.7 - t[i]) / (t[i + 1] - t[i]) * 0.5;
            println!(
                "{n_rx} RX {name:24}: SNR70 {s70:.2} dB  {:?}",
                t.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
            );
        }
    }
}
