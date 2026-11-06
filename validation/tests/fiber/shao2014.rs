//! The two nonlinear steps of the symmetric split-step method against J. Shao, X. Liang
//! and S. Kumar, "Comparison of split-step Fourier schemes for simulating fiber optic
//! communication systems," *IEEE Photonics J.* 6(4), 7200515, 2014: Table 2, the error of
//! a single step of length $h$ when the loss is combined with the dispersion (their scheme
//! I, eq. (11): the phase $\gamma |A|^{2} h$ of the power at the middle of the step after
//! half a step of dispersion and loss, [`NonlinearStep::Midpoint`], comnum's default and
//! OptiCommPy's `ssfm`) and when it is combined with the nonlinearity (their scheme II,
//! eq. (21): the phase $\gamma |A|^{2} h_{\mathrm{eff}}$ of the power after a lossless half
//! step of dispersion, $h_{\mathrm{eff}} = (1 - e^{-\alpha h})/\alpha$ of eq. (19), the loss
//! of the whole step applied after the phase, [`NonlinearStep::EffectiveLength`], the step
//! of Optilux's `fiber`).
//!
//! The system of their Sec. 3: 8192 random 32-QAM symbols shaped by a raised cosine of
//! roll-off 0.8, at 10 and 25 Gbaud, launched at 0 dBm (the power of Table 2 is not
//! stated: 0 dBm is the paper's first launch power) into a fibre of $\alpha = 0.046$
//! km⁻¹, $|\beta_{2}| = 5$ ps²/km and $\gamma = 2.2$ W⁻¹km⁻¹ (the sign of $\beta_{2}$ is not
//! stated: both are run, the card takes the anomalous one). The error of a single step is
//! the global error of their eq. (43), $\lVert A_{n} - A_{\mathrm{ref}} \rVert^{2} /
//! \lVert A_{\mathrm{ref}} \rVert^{2}$, of the field after one step of $h$ against the
//! reference, the same scheme with a nonlinear phase of 10⁻⁵ rad per step (steps of
//! 1.25 m here, $\gamma P_{\mathrm{peak}} h \approx 10^{-5}$), propagated over the same $h$.
//! The reference is computed once per symbol rate over 60 km and read at every $h$ of
//! the table; the sampling (8 samples per symbol) and the random symbols are the test's
//! own: the absolute errors depend on them, the ratio of the two schemes at equal $h$
//! and the order of the error in $h$ far less (the card
//! `validation/cards/shao2014_table2.yaml` records the published table and compares
//! both).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::core::rng::Rng;
use comnum::dsp::filter::{PulseShape, fir_filter, pulse_shape};
use comnum::dsp::resample::upsample;
use comnum::mapping::constellation::Constellation;
use comnum::mapping::source::symbols;
use comnum::math::fft::{fft, ifft};
use comnum::math::nlse::{ManakovSplitStep, Medium, NonlinearStep, PropagationOptions, StepRule};
use comnum_validation::card::Card;
use serde_json::json;

const SYMBOLS: usize = 8192;
const SPS: usize = 8;
const ROLLOFF: f64 = 0.8;
const POWER_W: f64 = 1e-3;
const ALPHA_NP_KM: f64 = 0.046;
const BETA2_PS2_KM: f64 = 5.0;
const GAMMA_W_KM: f64 = 2.2;
/// The steps of Table 2 (km), increasing: the reference is read at each of them.
const STEPS_KM: [f64; 11] = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0];
/// The step of the reference (km): 1.25 m, a nonlinear phase of about 1e-5 rad per step
/// at the peak power of the launched field, and a divisor of every increment of `STEPS_KM`.
const REFERENCE_STEP_KM: f64 = 1.25e-3;

/// The cross 32-QAM constellation: the 6 × 6 grid of odd levels without its corners,
/// labelled in the order of the grid (the labelling plays no part in a propagation).
fn qam32() -> Constellation {
    let levels = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
    let points: Vec<C64> = levels
        .iter()
        .flat_map(|&i| levels.iter().map(move |&q| C64::new(i, q)))
        .filter(|z| !(z.re.abs() == 5.0 && z.im.abs() == 5.0))
        .collect();
    assert_eq!(points.len(), 32);
    Constellation::from_points(points).unwrap().normalized()
}

/// The launched field at `symbol_rate`: 8192 random 32-QAM symbols (seed 1), raised
/// cosine of roll-off 0.8 at `SPS` samples per symbol, a mean power of 0 dBm; and the
/// sampling rate.
fn launch(symbol_rate: f64) -> (Vec<C64>, f64) {
    let c = qam32();
    let mut rng = Rng::new(1);
    let (_, s) = symbols(&c, SYMBOLS, None, &mut rng).unwrap();
    let h = pulse_shape(PulseShape::Rc { rolloff: ROLLOFF }, SPS, 16 * SPS + 1).unwrap();
    let mut x = fir_filter(&h, &upsample(&s, SPS));
    let power: f64 = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
    let scale = (POWER_W / power).sqrt();
    for v in &mut x {
        *v *= scale;
    }
    (x, symbol_rate * SPS as f64)
}

fn medium(beta2_sign: f64) -> Medium {
    Medium {
        alpha_np_km: ALPHA_NP_KM,
        beta2_s2_km: beta2_sign * BETA2_PS2_KM * 1e-24,
        gamma_w_km: GAMMA_W_KM,
        beta3_s3_km: 0.0,
    }
}

/// `x` after `length_km` of `medium` in steps of `step_km` with the nonlinear step
/// `nonlinear`.
fn propagate(medium: &Medium, nonlinear: NonlinearStep, x: &[C64], length_km: f64, step_km: f64, fs: f64) -> Vec<C64> {
    let engine = ManakovSplitStep { rule: StepRule::Fixed(step_km), nonlinear, ..ManakovSplitStep::new(*medium) };
    let mut e = x.to_vec();
    let options = PropagationOptions { gain: None, birefringence: None };
    engine.run_fields(&mut [e.as_mut_slice()], length_km, fs, &options).unwrap();
    e
}

/// A hypothesis for the paper's scheme II, which the table of the paper shows worse
/// than eq. (21) as comnum applies it: the phase of the power at the *start* of the
/// step, $|A(z)|^{2}$ before any dispersion, over $h_{\mathrm{eff}}$, between two lossless
/// half steps of dispersion, the loss of the step after the phase (eq. (21) with
/// $A(z)$ in place of $A_{l2}(z + h/2)$).
fn scheme_ii_with_the_power_at_the_start(medium: &Medium, x: &[C64], h: f64, fs: f64) -> Vec<C64> {
    let n = x.len();
    let omega: Vec<f64> = (0..n)
        .map(|k| {
            let f = if k < n / 2 { k as f64 } else { k as f64 - n as f64 } * fs / n as f64;
            2.0 * std::f64::consts::PI * f
        })
        .collect();
    let lossless = Medium { alpha_np_km: 0.0, gamma_w_km: 0.0, ..*medium };
    let half = lossless.linear_response(&omega, h / 2.0);
    let h_eff = (1.0 - (-medium.alpha_np_km * h).exp()) / medium.alpha_np_km;
    let loss = (-medium.alpha_np_km * h / 2.0).exp();
    let mut e = fft(x);
    for (v, l) in e.iter_mut().zip(&half) {
        *v *= *l;
    }
    let mut e = ifft(&e);
    for (v, x0) in e.iter_mut().zip(x) {
        *v *= C64::from_polar(loss, medium.gamma_w_km * x0.norm_sqr() * h_eff);
    }
    let mut e = fft(&e);
    for (v, l) in e.iter_mut().zip(&half) {
        *v *= *l;
    }
    ifft(&e)
}

/// The global error of eq. (43).
fn global_error(u: &[C64], reference: &[C64]) -> f64 {
    let err: f64 = u.iter().zip(reference).map(|(a, b)| (a - b).norm_sqr()).sum();
    err / reference.iter().map(C64::norm_sqr).sum::<f64>()
}

/// The single-step errors of both schemes at every step of Table 2, for one symbol rate
/// and one sign of $\beta_{2}$: `(h, e_I, e_II)`.
fn single_step_errors(symbol_rate: f64, beta2_sign: f64) -> Vec<(f64, f64, f64)> {
    let (x, fs) = launch(symbol_rate);
    let m = medium(beta2_sign);
    let peak = x.iter().map(C64::norm_sqr).fold(0.0, f64::max);
    println!(
        "{:.0} Gbaud, beta2 {:+.0} ps2/km: {} samples at {:.0} GS/s, peak {:.2} mW, reference phase {:.2e} rad per step",
        symbol_rate / 1e9,
        beta2_sign * BETA2_PS2_KM,
        x.len(),
        fs / 1e9,
        peak * 1e3,
        GAMMA_W_KM * peak * REFERENCE_STEP_KM
    );
    // the reference, read at every step of the table (propagated by the increments)
    let mut reference = x.clone();
    let mut covered = 0.0;
    let mut out = Vec::new();
    for &h in &STEPS_KM {
        let increment = h - covered;
        let n = (increment / REFERENCE_STEP_KM).round();
        assert!((n * REFERENCE_STEP_KM - increment).abs() < 1e-9, "the reference step divides the increments");
        reference = propagate(&m, NonlinearStep::Midpoint, &reference, increment, REFERENCE_STEP_KM, fs);
        covered = h;
        let one = propagate(&m, NonlinearStep::Midpoint, &x, h, h, fs);
        let two = propagate(&m, NonlinearStep::EffectiveLength, &x, h, h, fs);
        let (e1, e2) = (global_error(&one, &reference), global_error(&two, &reference));
        let start = global_error(&scheme_ii_with_the_power_at_the_start(&m, &x, h, fs), &reference);
        println!(
            "  h = {h:>4} km: scheme I {e1:.3e}, scheme II {e2:.3e}, ratio I/II {:.1}; II with the power at the start {start:.3e} (I over it {:.1})",
            e1 / e2,
            e1 / start
        );
        out.push((h, e1, e2));
    }
    out
}

/// The order of the error in $h$ between the first two steps of the table,
/// $\log(e_{2}/e_{1}) / \log(h_{2}/h_{1})$.
fn order(rows: &[(f64, f64, f64)], scheme: usize) -> f64 {
    let pick = |r: &(f64, f64, f64)| if scheme == 1 { r.1 } else { r.2 };
    (pick(&rows[1]) / pick(&rows[0])).ln() / (rows[1].0 / rows[0].0).ln()
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored (about 3 minutes)"]
fn shao_2014_table_2_single_step_errors() {
    let mut card = Card::load("shao2014_table2").unwrap();
    let t = std::time::Instant::now();
    for (rate, name) in [(10e9, "10 Gbaud"), (25e9, "25 Gbaud")] {
        // the other sign of beta2, for the record
        let normal = single_step_errors(rate, 1.0);
        let anomalous = single_step_errors(rate, -1.0);
        for ((h, n1, n2), (_, a1, a2)) in normal.iter().zip(&anomalous) {
            println!("{name}, h = {h} km: anomalous/normal scheme I {:.2}, scheme II {:.2}", a1 / n1, a2 / n2);
        }
        for (h, e1, e2) in &anomalous {
            card.check(&json!({"rate": name, "h_km": h, "scheme": "I"}), *e1);
            card.check(&json!({"rate": name, "h_km": h, "scheme": "II"}), *e2);
            card.check(&json!({"rate": name, "h_km": h, "ratio": "I/II"}), e1 / e2);
        }
        card.check(&json!({"rate": name, "order": "I"}), order(&anomalous, 1));
        card.check(&json!({"rate": name, "order": "II"}), order(&anomalous, 2));
    }
    println!("{:.0} s", t.elapsed().as_secs_f64());
    card.finish().unwrap();
}
