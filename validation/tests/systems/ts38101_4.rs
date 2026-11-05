//! The PDSCH of `comnum::systems::nr` against the demodulation requirements of 3GPP TS
//! 38.101-4 V19.4.0, clause 5.2.2.1.1 (FDD, 2 RX, normal performance), test 1-2:
//! R.PDSCH.1-1.2 (6 PRBs from PRB 23 of 52, QPSK, MCS 4 of Table 5.1.3.1-1, TBS 480, one
//! code block, 12 PDSCH symbols, DM-RS in symbols 2 and 11), TDLC300-100, 2 TX × 2 RX of
//! low correlation (ULA Low: independent links), 70 % of the maximum throughput at
//! 0.2 dB (Table 5.2.2.1.1-3), and the same test with 4 receive antennas, 70 % at -2.9 dB
//! (2x4, Table 5.2.3.1.1-3). The requirement is an upper bound: a conformant receiver reaches the
//! throughput at that SNR or below; the card `validation/cards/ts38101_4_pdsch.yaml`
//! compares the SNR of 70 % with it and records the margin.
//!
//! The link, from the test parameters of Tables 5.2-1 and 5.2.2.1.1-2:
//!
//! - every slot of 1 ms carries a PDSCH but slot 0 of every 20 (the SS/PBCH block);
//!   slots 10 and 11 carry the TRS (symbols 6 and 10, every fourth subcarrier, its
//!   subcarrier offset not given: 0), around which the PDSCH is rate-matched;
//! - the precoder of every PRG (2 PRBs, counted from common resource block 0) is drawn
//!   uniformly among the four of TS 38.214 Table 5.2.2.2.1-1 at every slot;
//! - HARQ: 4 processes served in turn, at most 4 transmissions of redundancy versions 0,
//!   2, 3, 1, the soft buffers combined; the throughput is the fraction of the PDSCH
//!   slots that deliver a transport block;
//! - the channel: TDLC300 (Table B.2.1.1-4) at a maximum Doppler of 100 Hz, its taps at
//!   the middle of every OFDM symbol, applied per subcarrier with the delays of the table
//!   (rounded to 1/983.04 MHz, 0.5 ns), without inter-carrier interference (100 Hz over
//!   15 kHz); the noise white of variance $10^{-\mathrm{SNR}/10}$ per RE and receive
//!   antenna, the signal of unit power per RE at every antenna;
//! - the receiver of `Pdsch::receive`: the precoded channel estimated in every PRG by a
//!   2-D Wiener interpolator designed for delays up to the cyclic prefix (4.69 µs) and a
//!   Doppler of 100 Hz, the noise variance known, maximum ratio combining.
//!
//! The industry reference is the alignment of RAN4 #88bis (Chengdu, October 2018), case 2
//! of FR1 FDD (the same test, 6 PRBs, MMSE-IRC): the results of four companies,
//! anonymised in the card as companies 1 to 4, from the contributions R4-1812164,
//! R4-1812461, R4-1813439 and R4-1813632 (a table, a spreadsheet, vector curves and a
//! raster figure read on its pixels), under the assumptions of R4-1811723. The bounds of
//! the card: the channel known (genie) and the estimator designed for TDLC300 (2.6 us).
//!
//! The SNR of 70 % throughput is interpolated linearly between the points of a 0.5 dB
//! grid, every point simulated over the same 20 realisations of 200 slots (3 800 PDSCH
//! slots, a standard error of the throughput under 0.8 %).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum_validation::card::Card;
use comnum_validation::ts38101_4::{Conditions, Rx, TS_38101_4, crossing, crossing_with};
use serde_json::json;

#[test]
#[ignore = "campaign: about seven minutes in release (just card ts38101_4_pdsch)"]
fn ts_38101_4_test_1_2_tdlc300_100() {
    let mut card = Card::load("ts38101_4_pdsch").unwrap();
    let grid = |a: f64| (0..9).map(|i| a + 0.5 * f64::from(i)).collect::<Vec<f64>>();
    // the curves, recorded for the documentation (docs/python/examples/systems)
    let mut curves = serde_json::Map::new();
    let mut record = |antennas: &str, receiver: &str, snrs: &[f64], t: &[f64], snr70: f64| {
        let entry = curves.entry(antennas.to_string()).or_insert_with(|| json!({}));
        entry[receiver] = json!({"snr_db": snrs, "throughput": t, "snr_70_db": snr70});
    };
    for (n_rx, antennas, start) in [(2, "2x2", -4.0), (4, "2x4", -7.0)] {
        let snrs = grid(start);
        let (t, snr70) = crossing(n_rx, &snrs, 0.7).unwrap();
        record(antennas, "robust", &snrs, &t, snr70);
        card.check(&json!({"test": "1-2", "antennas": antennas, "quantity": "snr_70_db"}), snr70);
        card.check(
            &json!({"test": "1-2", "antennas": antennas, "quantity": "snr_70_db", "reference": "company 4"}),
            snr70,
        );
        card.check(
            &json!({"test": "1-2", "antennas": antennas, "quantity": "snr_70_db", "reference": "RAN4 #88bis"}),
            snr70,
        );
        // the curves of companies 4 and 2 at the integer SNRs of the grid
        for (s, v) in snrs.iter().zip(&t).filter(|(s, _)| s.fract() == 0.0) {
            let inputs = json!({"test": "1-2", "antennas": antennas, "quantity": "throughput", "snr_db": s, "reference": "company 4"});
            card.check(&inputs, *v);
            let inputs = json!({"test": "1-2", "antennas": antennas, "quantity": "throughput", "snr_db": s, "reference": "company 2"});
            card.check(&inputs, *v);
        }
        for c in 1..=3 {
            let inputs = json!({"test": "1-2", "antennas": antennas, "quantity": "snr_70_db", "reference": format!("company {c}")});
            card.check(&inputs, snr70);
        }
    }
    // the bounds: the channel known (genie) and the estimator matched to TDLC300
    for (n_rx, antennas, genie_start, matched_start) in [(2, "2x2", -6.0, -4.0), (4, "2x4", -9.5, -7.0)] {
        for (receiver, name, start) in
            [(Rx::Genie, "genie", genie_start), (Rx::Estimated { max_delay_s: 2.6e-6 }, "matched", matched_start)]
        {
            let c = Conditions { receiver, ..TS_38101_4 };
            let snrs = grid(start);
            let (t, snr70) = crossing_with(c, n_rx, &snrs, 0.7).unwrap();
            record(antennas, name, &snrs, &t, snr70);
            card.check(&json!({"test": "1-2", "antennas": antennas, "quantity": "snr_70_db", "receiver": name}), snr70);
        }
    }
    let doc = json!({
        "test": "TS 38.101-4 V19.4.0, clause 5.2.2.1.1 test 1-2 (R.PDSCH.1-1.2, TDLC300-100)",
        "budget": "20 realisations of 200 slots per point, seed 38101",
        "produced_by": "validation/tests/systems/ts38101_4.rs::ts_38101_4_test_1_2_tdlc300_100",
        "curves": curves,
    });
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results/ts38101_4_pdsch.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
    card.finish().unwrap();
}
