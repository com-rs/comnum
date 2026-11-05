//! DVB-S (`comnum_systems::dvbs`) against ETSI EN 300 421 V1.1.2, *Digital Video
//! Broadcasting (DVB); Framing structure, channel coding and modulation for 11/12 GHz
//! satellite services*, 1997, Table 3 ("IF-Loop performance of the System").
//!
//! Table 3 gives, for each inner code rate, the $E_{b}/N_{0}$ (per useful bit, before
//! the outer code) at which the BER after the Viterbi decoder is $2 \cdot 10^{-4}$, the
//! input of the outer decoder for a quasi-error-free output: 4.5, 5.0, 5.5, 6.0 and
//! 6.4 dB from 1/2 to 7/8. Its NOTE 1 says the figures include "a modem implementation
//! margin of 0,8 dB and the noise bandwidth increase due to the outer code (10 log
//! 188/204 = 0,36 dB)": an ideal modem should reach the BER at Table 3 − 0.8 dB, the
//! outer code being already counted by referring $E_{b}$ to the useful bits.
//!
//! The link: bursts of 200 random transport packets through the whole chain of
//! [`Dvbs`] (energy dispersal, RS(204, 188), the interleaver $I = 12$, the punctured
//! inner code, the QPSK of Figure 5), white Gaussian noise, exact LLRs, soft Viterbi
//! decoding over the whole burst; the BER is counted on the bytes entering the inner
//! coder, the flush of the interleaver included. The $E_{b}/N_{0}$ of the crossing is
//! interpolated between points 0.1 dB apart, with $10^{6}$ bits at least per point.
//! The values and tolerances are in the card `validation/cards/dvbs_table3.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::awgn::add_noise;
use comnum::core::rng::Rng;
use comnum::math::db2lin;
use comnum::systems::dvb::{self, InnerRate};
use comnum::systems::dvbs::Dvbs;
use comnum_validation::card::Card;
use comnum_validation::sweep_crossing;
use serde_json::json;

const PACKETS: usize = 200;
const MIN_BITS: usize = 1_000_000;
const TARGET: f64 = 2e-4;

/// The BER after the Viterbi decoder at `ebn0_db` per useful bit.
fn viterbi_ber(link: &Dvbs, ebn0_db: f64, rng: &mut Rng) -> f64 {
    let noise_var = 1.0 / db2lin(link.esn0_db(ebn0_db));
    let (mut errors, mut bits) = (0usize, 0usize);
    while bits < MIN_BITS {
        let packets = dvb::random_packets(PACKETS, rng);
        let tx = dvb::burst_encode(&packets).unwrap();
        let mut x = link.inner(&tx).unwrap();
        add_noise(&mut x, noise_var, rng);
        let rx = link.inner_decode(&x, noise_var).unwrap();
        errors += tx.iter().zip(&rx).map(|(a, b)| (a ^ b).count_ones() as usize).sum::<usize>();
        bits += 8 * tx.len();
    }
    errors as f64 / bits as f64
}

#[test]
#[ignore = "validation against the standard: run with --release --ignored"]
fn the_required_ebn0_follows_table_3_of_en_300_421() {
    let mut card = Card::load("dvbs_table3").unwrap();
    let mut rng = Rng::new(421);
    for (rate, table) in InnerRate::ALL.into_iter().zip([4.5, 5.0, 5.5, 6.0, 6.4]) {
        let link = Dvbs::new(rate).unwrap();
        let start = table - 2.0;
        let crossing = sweep_crossing(start, 0.1, 40, TARGET, |_, e| viterbi_ber(&link, e, &mut rng)).unwrap();
        println!(
            "rate {}: BER 2e-4 at Eb/N0 = {crossing:.2} dB (Table 3: {table}, minus 0.8 dB: {:.1})",
            rate.name(),
            table - 0.8
        );
        card.check(&json!({"rate": rate.name()}), crossing);
    }
    card.finish().unwrap();
}
