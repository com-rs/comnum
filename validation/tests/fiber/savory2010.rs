//! The truncated impulse response of the chromatic-dispersion compensation
//! (`comnum::dsp::compensation::cd::{cd_savory_length, cd_savory_taps}`) against S. J.
//! Savory, "Digital coherent optical receivers: algorithms and subsystems," *IEEE J. Sel.
//! Topics Quantum Electron.* 16(5), 1164–1179, 2010, eq. (11):
//!
//! $$
//! h_{\mathrm{cd}}[k] = \frac{1}{\sqrt{\rho}} \exp\left(-j \frac{\pi}{\rho}
//! \left(k - \frac{N - 1}{2}\right)^{2}\right), \qquad N = \lfloor |\rho| \rfloor, \qquad
//! \rho = \frac{2\pi\beta_{2}L}{T_{\mathrm{ADC}}^{2}},
//! $$
//!
//! for $\beta_{2} = -21$ ps²/km (the value of Sec. VI-A): the number of taps, the modulus
//! of the taps and their phase relative to the centre tap. The paper takes the electrical
//! engineering convention of the field, $e^{j(\omega t - \beta z)}$ (its footnote 11), and
//! comnum that of physics, so the phases of the library are those of eq. (11) conjugated;
//! the test conjugates them. The values of eq. (11), the tolerances and the differences
//! are in the card `validation/cards/savory2010_eq11.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::dsp::compensation::cd::{cd_savory_length, cd_savory_taps};
use comnum::dsp::compensation::iq::{gram_schmidt, iq_correlation, lowdin};
use comnum::prelude::*;
use comnum_validation::card::Card;
use serde_json::json;

/// $\beta_{2}$ (s²/km) of Sec. VI-A.
const BETA2: f64 = -21e-24;

#[test]
fn savory_2010_eq_11() {
    let mut card = Card::load("savory2010_eq11").unwrap();
    for (length_km, fs) in [(100.0, 56e9), (500.0, 56e9), (1000.0, 56e9), (80.0, 64e9)] {
        let n = cd_savory_length(BETA2, length_km, fs);
        card.check(&json!({"quantity": "taps", "length_km": length_km, "fs": fs}), n as f64);
    }
    // the taps of 1000 km at 56 GSa/s, about the centre tap c
    let (length_km, fs) = (1000.0, 56e9);
    let h = cd_savory_taps(BETA2, length_km, fs, cd_savory_length(BETA2, length_km, fs)).unwrap();
    let c = h.len() / 2;
    for m in [0, 1, 5, c] {
        let inputs = |q: &str| json!({"quantity": q, "length_km": length_km, "fs": fs, "offset": m});
        card.check(&inputs("modulus"), h[c + m].norm());
        if m > 0 {
            card.check(&inputs("phase"), (h[c + m] / h[c]).conj().arg());
        }
    }
    card.finish().unwrap();
}

/// Eqs. (6)-(8) and Fig. 3: the arms of a front end with a phase error $2\theta$,
/// $r = M[I, Q]^{T}$, $M = [[\cos\theta, \sin\theta], [\sin\theta, \cos\theta]]$, on
/// $10^{5}$ 16-QAM symbols: the correlation of the parts ($\sin 2\theta$, eq. (6)), the
/// phase of the Gram–Schmidt output against the symbols ($-\theta$: eq. (7) keeps $r_{1}$, the
/// signal turned by $\theta$), and the error of the Löwdin output, which turns both parts
/// back (eq. (8), Fig. 3). Values and tolerances in `validation/cards/savory2010_eqs6_8.yaml`.
#[test]
fn savory_2010_eqs_6_to_8_orthonormalisation() {
    let mut card = Card::load("savory2010_eqs6_8").unwrap();
    let mut rng = Rng::new(2010);
    let levels = [-3.0, -1.0, 1.0, 3.0];
    let x: Vec<C64> = (0..100_000)
        .map(|_| {
            C64::new(levels[(rng.uniform() * 4.0) as usize % 4], levels[(rng.uniform() * 4.0) as usize % 4])
                / 10f64.sqrt()
        })
        .collect();
    for theta in [0.05f64, 0.1, 0.2, 0.3] {
        let r: Vec<C64> = x
            .iter()
            .map(|z| C64::new(theta.cos() * z.re + theta.sin() * z.im, theta.sin() * z.re + theta.cos() * z.im))
            .collect();
        card.check(&json!({"quantity": "correlation", "theta": theta}), iq_correlation(&r));
        let phase = |y: &[C64]| y.iter().zip(&x).map(|(a, b)| a * b.conj()).sum::<C64>().arg();
        let g = gram_schmidt(&r).unwrap();
        card.check(&json!({"quantity": "gram_schmidt_phase", "theta": theta}), phase(&g));
        let l = lowdin(&r).unwrap();
        card.check(&json!({"quantity": "lowdin_phase", "theta": theta}), phase(&l));
        let err = (l.iter().zip(&x).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / x.len() as f64).sqrt();
        card.check(&json!({"quantity": "lowdin_rms_error", "theta": theta}), err);
    }
    card.finish().unwrap();
}
