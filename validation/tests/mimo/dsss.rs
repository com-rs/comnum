//! Simulated direct-sequence links (`comnum::waveform::dsss::Dsss`) against their exact
//! error rates:
//!
//! - synchronous DS-CDMA, eight users on Gold codes of length 31 with unequal
//!   amplitudes, BPSK: the bit error rate of the first user with the conventional
//!   receiver and with the decorrelator against the exact expressions of R. Lupas and
//!   S. Verdú, "Linear multiuser detectors for synchronous code-division
//!   multiple-access channels," *IEEE Trans. Inf. Theory* 35(1), 123–136, 1989
//!   (`theory::ber_matched_filter`, the mean over the 128 patterns of the interfering
//!   bits, and `theory::ber_decorrelator`), card `validation/cards/lupas1989_ber.yaml`;
//! - one user on an m-sequence of 127 chips over three Rayleigh paths of equal mean
//!   power, delayed by 0, 2 and 5 chips and constant over a burst of 20 symbols, a rake
//!   receiver of one, two and three fingers that knows the gains: the bit error rate
//!   against the maximal-ratio combining of as many Rayleigh branches (Proakis and
//!   Salehi, *Digital Communications*, 5th ed., eq. (13.4-15),
//!   `comnum::mapping::theory::ber_rayleigh_mrc`), card
//!   `validation/cards/proakis2008_rake.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::core::rng::Rng;
use comnum::mapping::constellation::Constellation;
use comnum::mapping::theory::ber_rayleigh_mrc;
use comnum::math::sequences::{gold_codes, m_sequence};
use comnum::math::{C64, db2lin};
use comnum::mimo::detection::Linear;
use comnum::waveform::dsss::{Dsss, theory};
use comnum_validation::card::Card;
use serde_json::json;

/// `n` random BPSK symbols per user and their labels.
fn random_bpsk(c: &Constellation, users: usize, n: usize, rng: &mut Rng) -> (Vec<Vec<usize>>, Vec<Vec<C64>>) {
    let labels: Vec<Vec<usize>> = (0..users).map(|_| (0..n).map(|_| usize::from(rng.bit())).collect()).collect();
    let symbols = labels.iter().map(|l| l.iter().map(|&i| c.points()[i]).collect()).collect();
    (labels, symbols)
}

#[test]
fn lupas_verdu_1989_the_simulated_error_rates_of_the_linear_receivers_are_the_exact_ones() {
    const SYMBOLS: usize = 1_000_000;
    let mut card = Card::load("lupas1989_ber").unwrap();
    let mut rng = Rng::new(1989);
    let bpsk = Constellation::psk(2).unwrap();
    // the decimated sequence and seven sums: correlations of -9/31, 7/31 and -1/31
    let cdma = Dsss::new(&gold_codes(5).unwrap()[1..9]).unwrap();
    let cdma = cdma.with_amplitudes(&[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 1.0]).unwrap();
    for ebn0_db in [4.0, 7.0] {
        let noise_var = 1.0 / db2lin(ebn0_db);
        let (labels, symbols) = random_bpsk(&bpsk, 8, SYMBOLS, &mut rng);
        let mut chips = cdma.modulate(&symbols).unwrap();
        let noise = rng.complex_noise(chips.len(), noise_var);
        chips.iter_mut().zip(&noise).for_each(|(c, w)| *c += w);
        for (name, linear) in [("mr", Linear::mr(bpsk.clone())), ("zf", Linear::zf(bpsk.clone()))] {
            let decided = cdma.detect(&linear, &chips, noise_var).unwrap();
            let ber = decided[0].iter().zip(&labels[0]).filter(|(a, b)| a != b).count() as f64 / SYMBOLS as f64;
            let exact = match name {
                "mr" => theory::ber_matched_filter(&cdma.signatures(), 0, noise_var).unwrap(),
                _ => theory::ber_decorrelator(&cdma.signatures(), 0, noise_var).unwrap(),
            };
            println!("{name} at {ebn0_db} dB: BER {ber:.4e} (exact {exact:.4e})");
            card.check(&json!({"receiver": name, "ebn0_db": ebn0_db}), ber / exact);
        }
    }
    card.finish().unwrap();
}

#[test]
fn a_rake_receiver_over_rayleigh_paths_has_the_error_rate_of_maximal_ratio_combining() {
    const BURSTS: usize = 20_000;
    const SYMBOLS: usize = 20;
    const DELAYS: [usize; 3] = [0, 2, 5];
    let mut card = Card::load("proakis2008_rake").unwrap();
    let mut rng = Rng::new(1958);
    let bpsk = Constellation::psk(2).unwrap();
    let link = Dsss::new(&[m_sequence(7).unwrap()]).unwrap();
    for ebn0_db in [5.0, 10.0] {
        // the mean Eb/N0 of every path: a third of the total
        let per_path = db2lin(ebn0_db) / 3.0;
        let noise_var = 1.0 / db2lin(ebn0_db);
        let mut errors = [0usize; 3];
        for _ in 0..BURSTS {
            let (labels, symbols) = random_bpsk(&bpsk, 1, SYMBOLS, &mut rng);
            let x = link.modulate(&symbols).unwrap();
            let paths: Vec<(usize, C64)> =
                DELAYS.iter().map(|&d| (d, rng.complex_gaussian() * (1.0f64 / 3.0).sqrt())).collect();
            let mut r = rng.complex_noise(x.len() + DELAYS[2], noise_var);
            for &(d, g) in &paths {
                x.iter().enumerate().for_each(|(i, xi)| r[i + d] += g * xi);
            }
            for (fingers, count) in errors.iter_mut().enumerate() {
                let z = link.rake(&r, 0, &paths[..=fingers]).unwrap();
                *count += z.iter().zip(&labels[0]).filter(|(z, l)| usize::from(z.re < 0.0) != **l).count();
            }
        }
        for (fingers, count) in errors.iter().enumerate() {
            let ber = *count as f64 / (BURSTS * SYMBOLS) as f64;
            let mrc = ber_rayleigh_mrc(per_path, fingers + 1).unwrap();
            println!("{} finger(s) at {ebn0_db} dB: BER {ber:.4e} (MRC {mrc:.4e})", fingers + 1);
            card.check(&json!({"fingers": fingers + 1, "ebn0_db": ebn0_db}), ber / mrc);
        }
    }
    card.finish().unwrap();
}
