//! The error rates over flat fading by the MGF approach (`comnum::mapping::theory::{SnrFading,
//! ber_binary_fading, ser_fading}`) against M. K. Simon and M.-S. Alouini, "A unified
//! approach to the performance analysis of digital communication over generalized fading
//! channels," *Proc. IEEE* 86(9), 1998: eq. (22), the average of the conditional error
//! rate over the pdf of the SNR (Table 2: Rice and Nakagami-m), which eqs. (23) and (46)
//! write as one finite integral of the MGF (Table 3).
//!
//! The test simulates one symbol at a time through a flat fading channel known at the
//! receiver: BPSK at a mean $E_{b}/N_{0}$ of 10 dB and 16-QAM at a mean $E_{s}/N_{0}$
//! of 20 dB, the amplitude of each symbol drawn from a Rice distribution of factor $K$
//! (a line of sight $\sqrt{K/(1 + K)}$ plus a circular Gaussian term of power
//! $1/(1 + K)$) or from a Nakagami-m distribution (the power a gamma variable of shape
//! $m$ scaled to a unit mean, `Rng::gamma`), complex white Gaussian noise, the division
//! by the amplitude, the minimum-distance decision. It measures the bit error rate of
//! BPSK and the symbol error rate of 16-QAM, and prints the MGF average next to them.
//! The expected values, the numerical averages of eq. (22), and their tolerances are in
//! the card `validation/cards/simon1998_mgf_fading.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::core::rng::Rng;
use comnum::mapping::constellation::{Constellation, ConstellationKind};
use comnum::mapping::source::symbols;
use comnum::mapping::theory::{SnrFading, ber_binary_fading, ser_fading};
use comnum::math::db2lin;
use comnum_validation::card::Card;
use serde_json::json;

const SYMBOLS: usize = 400_000;

/// The amplitude of one symbol, of unit mean power, drawn from the fading.
fn amplitude(fading: SnrFading, rng: &mut Rng) -> C64 {
    match fading {
        SnrFading::Rice { k_factor } => {
            C64::new((k_factor / (1.0 + k_factor)).sqrt(), 0.0)
                + rng.complex_gaussian() * (1.0 / (1.0 + k_factor)).sqrt()
        }
        SnrFading::Nakagami { m } => C64::new((rng.gamma(m) / m).sqrt(), 0.0),
        other => unreachable!("{other:?}"),
    }
}

/// The error rate of the constellation `c` at the mean SNR per symbol `esn0_db` over
/// the fading: the fraction of symbols decided wrongly.
fn simulate(c: &Constellation, esn0_db: f64, fading: SnrFading, rng: &mut Rng) -> f64 {
    let sigma = (1.0 / db2lin(esn0_db)).sqrt();
    let (labels, points) = symbols(c, SYMBOLS, None, rng).unwrap();
    let mut errors = 0usize;
    for (&label, &x) in labels.iter().zip(&points) {
        let h = amplitude(fading, rng);
        let y = (h * x + rng.complex_gaussian() * sigma) / h;
        if c.nearest(y) != label {
            errors += 1;
        }
    }
    errors as f64 / SYMBOLS as f64
}

#[test]
fn the_simulated_error_rates_over_rice_and_nakagami_fading_are_their_mgf_averages() {
    let mut card = Card::load("simon1998_mgf_fading").unwrap();
    let mut rng = Rng::new(1998);
    let bpsk = Constellation::psk(2).unwrap().normalized();
    let qam16 = Constellation::qam(16).unwrap().normalized();
    let fadings = [
        SnrFading::Rice { k_factor: 3.0 },
        SnrFading::Rice { k_factor: 10.0 },
        SnrFading::Nakagami { m: 0.5 },
        SnrFading::Nakagami { m: 2.0 },
        SnrFading::Nakagami { m: 4.0 },
    ];
    for fading in fadings {
        let (name, parameter) = match fading {
            SnrFading::Rice { k_factor } => ("rice", k_factor),
            SnrFading::Nakagami { m } => ("nakagami", m),
            other => unreachable!("{other:?}"),
        };
        // BPSK: Es = Eb, the symbol errors are the bit errors
        let ber = simulate(&bpsk, 10.0, fading, &mut rng);
        let mgf = ber_binary_fading(1.0, fading, 10.0, 1).unwrap();
        println!("BPSK at 10 dB over {name} {parameter}: BER {ber:.3e} (MGF average {mgf:.3e})");
        card.check(&json!({"signal": "bpsk", "fading": name, "parameter": parameter}), ber);
        let ser = simulate(&qam16, 20.0, fading, &mut rng);
        let mgf = ser_fading(ConstellationKind::Qam, 16, fading, 20.0).unwrap();
        println!("16-QAM at 20 dB over {name} {parameter}: SER {ser:.3e} (MGF average {mgf:.3e})");
        card.check(&json!({"signal": "qam16", "fading": name, "parameter": parameter}), ser);
    }
    card.finish().unwrap();
}
