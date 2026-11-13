//! The error rates over flat fading by the MGF approach (`comnum::mapping::theory::{SnrFading,
//! ber_binary_fading, ser_fading}`) against M. K. Simon and M.-S. Alouini, "A unified
//! approach to the performance analysis of digital communication over generalized fading
//! channels," *Proc. IEEE* 86(9), 1998: eq. (22), the average of the conditional error
//! rate over the pdf of the SNR (Table 2: Rice and Nakagami-m), which eqs. (23) and (46)
//! write as one finite integral of the MGF (Table 3).
//!
//! The test simulates one symbol at a time through a flat fading channel known at the
//! receiver: BPSK at a mean $E_{b}/N_{0}$ of 10 dB and 16-QAM at a mean $E_{s}/N_{0}$
//! of 20 dB, the gain of each symbol drawn by `comnum::channel::fading::flat_gains` from
//! a Rice distribution of factor $K$ or a Nakagami-m distribution, complex white
//! Gaussian noise, the division
//! by the amplitude, the minimum-distance decision. It measures the bit error rate of
//! BPSK and the symbol error rate of 16-QAM, and prints the MGF average next to them.
//! The expected values, the numerical averages of eq. (22), and their tolerances are in
//! the card `validation/cards/simon1998_mgf_fading.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::fading::{FlatFading, flat_gains};
use comnum::core::rng::Rng;
use comnum::mapping::constellation::{Constellation, ConstellationKind};
use comnum::mapping::source::symbols;
use comnum::mapping::theory::{SnrFading, ber_binary_fading, ser_fading};
use comnum::math::db2lin;
use comnum_validation::card::Card;
use serde_json::json;

const SYMBOLS: usize = 400_000;

/// The error rate of the constellation `c` at the mean SNR per symbol `esn0_db` over
/// the fading: the fraction of symbols decided wrongly.
fn simulate(c: &Constellation, esn0_db: f64, fading: FlatFading, rng: &mut Rng) -> f64 {
    let sigma = (1.0 / db2lin(esn0_db)).sqrt();
    let (labels, points) = symbols(c, SYMBOLS, None, rng).unwrap();
    let gains = flat_gains(fading, SYMBOLS, 1, rng).unwrap();
    let errors = labels
        .iter()
        .zip(&points)
        .zip(&gains)
        .filter(|((label, x), h)| c.nearest((*h * *x + rng.complex_gaussian() * sigma) / *h) != **label)
        .count();
    errors as f64 / SYMBOLS as f64
}

#[test]
fn the_simulated_error_rates_over_rice_and_nakagami_fading_are_their_mgf_averages() {
    let mut card = Card::load("simon1998_mgf_fading").unwrap();
    let mut rng = Rng::new(1998);
    let bpsk = Constellation::psk(2).unwrap().normalized();
    let qam16 = Constellation::qam(16).unwrap().normalized();
    for (name, parameter) in [("rice", 3.0), ("rice", 10.0), ("nakagami", 0.5), ("nakagami", 2.0), ("nakagami", 4.0)] {
        // the same channel for the simulation and for the theory
        let fading = FlatFading::from_name(name, Some(parameter)).unwrap();
        let theory = SnrFading::from(fading);
        // BPSK: Es = Eb, the symbol errors are the bit errors
        let ber = simulate(&bpsk, 10.0, fading, &mut rng);
        let mgf = ber_binary_fading(1.0, theory, 10.0, 1).unwrap();
        println!("BPSK at 10 dB over {name} {parameter}: BER {ber:.3e} (MGF average {mgf:.3e})");
        card.check(&json!({"signal": "bpsk", "fading": name, "parameter": parameter}), ber);
        let ser = simulate(&qam16, 20.0, fading, &mut rng);
        let mgf = ser_fading(ConstellationKind::Qam, 16, theory, 20.0).unwrap();
        println!("16-QAM at 20 dB over {name} {parameter}: SER {ser:.3e} (MGF average {mgf:.3e})");
        card.check(&json!({"signal": "qam16", "fading": name, "parameter": parameter}), ser);
    }
    card.finish().unwrap();
}
