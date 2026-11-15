//! A simulated frequency-hopping link (`comnum::waveform::fhss::Fhss`) under a
//! partial-band noise jammer (`comnum::channel::jamming::partial_band_noise`) against
//! the error rate of slow FH with noncoherent $M$-FSK, $P_{b}(\rho) = \rho\\,
//! P_{b}(\rho E_{b}/N_{J})$ without thermal noise (Proakis and Salehi, *Digital
//! Communications*, 5th ed., Sec. 12.3; `fhss::theory::ber_partial_band`), and its
//! worst case $e^{-1} N_{J}/E_{b}$ at $\rho = 2 N_{J}/E_{b}$ for binary FSK (Viterbi and
//! Jacobs 1975).
//!
//! 50 channels, one symbol per hop on a random pattern as long as the run, the jammer on the first
//! channels of the band: the whole band, a fifth of it and one channel. The expected
//! values and the tolerances are in the card
//! `validation/cards/proakis2008_fh_partial_band.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::jamming::partial_band_noise;
use comnum::core::rng::Rng;
use comnum::math::db2lin;
use comnum::waveform::fhss::{Fhss, theory};
use comnum_validation::card::Card;
use serde_json::json;

#[test]
fn a_hopping_link_under_partial_band_noise_has_the_error_rate_of_the_jammed_hops() {
    const CHANNELS: usize = 50;
    const SYMBOLS: usize = 400_000;
    let mut card = Card::load("proakis2008_fh_partial_band").unwrap();
    let mut rng = Rng::new(1975);
    for (m, ebnj_db, jammed) in [(2usize, 10.0, 50usize), (2, 10.0, 10), (2, 20.0, 1), (4, 10.0, 10), (4, 16.0, 1)] {
        let link = Fhss::new(CHANNELS, m, 1, Fhss::random_pattern(CHANNELS, SYMBOLS, &mut rng)).unwrap();
        let n = link.symbol_len();
        let bits_per_symbol = (m as f64).log2();
        let symbols: Vec<usize> = (0..SYMBOLS).map(|_| rng.below(m)).collect();
        let mut r = link.modulate(&symbols).unwrap();
        // Eb/NJ = N / (J log2 M) for a unit envelope
        let power = n as f64 / (db2lin(ebnj_db) * bits_per_symbol);
        let jammer = partial_band_noise(r.len(), n, 0, jammed * m, power, &mut rng).unwrap();
        r.iter_mut().zip(&jammer).for_each(|(v, j)| *v += j);
        let decided = link.demodulate(&r).unwrap();
        // the bit errors of the natural labelling of the tones
        let bit_errors: u32 = decided.iter().zip(&symbols).map(|(a, b)| (a ^ b).count_ones()).sum();
        let ber = f64::from(bit_errors) / (SYMBOLS as f64 * bits_per_symbol);
        let rho = jammed as f64 / CHANNELS as f64;
        let expected = theory::ber_partial_band(m, ebnj_db, rho, None).unwrap();
        println!("M = {m}, Eb/NJ = {ebnj_db} dB, rho = {rho}: BER {ber:.4e} (theory {expected:.4e})");
        card.check(&json!({"m": m, "ebnj_db": ebnj_db, "rho": rho}), ber / expected);
    }
    // the worst case of binary FSK at 20 dB is one channel of 50
    let (rho, worst) = theory::worst_case_partial_band(2, 20.0, None).unwrap();
    card.check(&json!({"quantity": "worst fraction of the band at 20 dB"}), rho);
    card.check(&json!({"quantity": "worst error rate at 20 dB"}), worst);
    card.finish().unwrap();
}
