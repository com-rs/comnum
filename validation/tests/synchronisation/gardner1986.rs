//! Gardner's timing-error detector (`comnum::dsp::sync::timing`) against F. M. Gardner,
//! "A BPSK/QPSK timing-error detector for sampled receivers," *IEEE Trans. Commun.*
//! 34(5), 423–429, 1986.
//!
//! The paper prints no numerical curve: it derives the detector (eqs. (1), (11)), its
//! S-curve as a sum over the pulse (eq. (15)) and, for a pulse band-limited to $1/T$, the
//! sinusoidal closed form (29) with the overlap integral $G_{R}$, and shows the
//! independence from the carrier phase (eq. (33)). The test measures:
//!
//! - the gain $8\pi G_{R}/T^{2}$ of [`gardner_gain`] for raised cosines of roll-off 0.2
//!   to 1 (and its limit 8/3 at roll-off 1);
//! - the mean of [`gardner_ted`] by Monte Carlo on a real raised-cosine BPSK signal of
//!   $10^{5}$ symbols ±1 (the pulse evaluated at the exact instants, 64 symbols on each
//!   side), sampled at $\tau + qT$ and $\tau + (q - \tfrac12)T$, against the S-curve;
//! - the detector on the signal turned by a carrier phase of 1 rad (eq. (33)).
//!
//! The expected values, evaluated from eqs. (15) and (29) outside comnum, and their
//! tolerance are in `validation/cards/gardner1986_scurve.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::dsp::filter::rc_taps;
use comnum::dsp::sync::timing::{gardner_gain, gardner_ted};
use comnum::prelude::*;
use comnum_validation::card::Card;
use serde_json::json;

/// Symbols on each side of an instant that the pulse sum keeps.
const SPAN: usize = 64;

/// The real raised-cosine signal of `a` at the time `t` (in symbols).
fn signal(a: &[f64], rolloff: f64, t: f64) -> f64 {
    let k0 = t.floor() as i64;
    let ks: Vec<i64> = (k0 - SPAN as i64..=k0 + SPAN as i64).filter(|&k| k >= 0 && (k as usize) < a.len()).collect();
    let times: Vec<f64> = ks.iter().map(|&k| t - k as f64).collect();
    let g = rc_taps(&times, rolloff, 1.0);
    ks.iter().zip(&g).map(|(&k, g)| a[k as usize] * g).sum()
}

/// The mean of the detector over the symbols, at the timing offset `tau` and a carrier
/// phase `phase`.
fn mean_detector(a: &[f64], rolloff: f64, tau: f64, phase: f64) -> f64 {
    let rot = C64::from_polar(1.0, phase);
    let range = SPAN + 1..a.len() - SPAN - 1;
    let n = range.len() as f64;
    range
        .map(|r| {
            let r = r as f64;
            let x = [r - 1.0, r - 0.5, r].map(|t| C64::new(signal(a, rolloff, t + tau), 0.0) * rot);
            gardner_ted(&x)
        })
        .sum::<f64>()
        / n
}

#[test]
#[ignore = "statistical validation: run with --ignored (about 2 s)"]
fn gardner_1986_s_curve_gain_and_phase_independence() {
    let mut card = Card::load("gardner1986_scurve").unwrap();
    for b in [0.2, 0.35, 0.5, 0.75, 1.0] {
        card.check(&json!({"quantity": "gain", "rolloff": b}), gardner_gain(b));
    }
    let mut rng = Rng::new(1986);
    let a: Vec<f64> = (0..100_000).map(|_| if rng.uniform() < 0.5 { 1.0 } else { -1.0 }).collect();
    for b in [0.35, 0.5, 1.0] {
        for tau in [0.1, 0.25] {
            let m = mean_detector(&a, b, tau, 0.0);
            println!("rolloff {b}, tau {tau}: mean {m:.5}");
            card.check(&json!({"quantity": "mean", "rolloff": b, "tau": tau}), m);
        }
    }
    // eq. (33): the same mean with the carrier turned by 1 rad
    let d = (mean_detector(&a, 0.5, 0.1, 1.0) - mean_detector(&a, 0.5, 0.1, 0.0)).abs();
    card.check(&json!({"claim": "carrier_phase_independence"}), d);
    card.finish().unwrap();
}
