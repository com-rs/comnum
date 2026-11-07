//! The convergence of the split-step solver of `comnum::channel::optical::fiber::Ssfm`
//! towards its own solution with very fine steps, on the link of Fig. 8 of C. Häger and
//! H. D. Pfister, "Physics-based deep learning for fiber-optic communication systems,"
//! *IEEE J. Sel. Areas Commun.* 39(1), 280–294, 2021 (arXiv:2010.14258): one channel of
//! Gaussian symbols at 32 GBd, root-raised-cosine pulses of roll-off 0.1 at 6 samples per
//! symbol, spans of 100 km of standard fibre ($\beta_{2} = -21.683$ ps²/km at 194.6 THz,
//! $\gamma = 1.3$ W⁻¹km⁻¹, 0.2 dB/km), ideal noiseless amplifiers after every span. The
//! paper propagates with 500 logarithmic steps per span (its Table I); its effective SNR
//! curves are reproduced by `specs/validation/hager2021_fig8.yaml`.
//!
//! When the step $h$ tends to zero the symmetric split step converges to the solution of
//! the nonlinear Schrödinger equation on the sampling grid, with a global error in
//! $h^{2}$ (Sinkin et al., *J. Lightw. Technol.* 21(1), 2003, Sec. III-E). A solution with
//! very fine steps is therefore a reference for the error of the step, and of the step
//! alone: it does not check the equation (signs, units, the definition of $\gamma$),
//! which the exact solitons (`sinkin2003_exact`) and the recorded fields of Optilux and
//! OptiCommPy do, nor the sampling, which stays that of the link. On the full link the
//! gallery example `plot_08_split_step_techniques` takes 500 uniform steps per span as
//! the reference and measures the effective SNR every technique loses against it; here
//! the link is cut to two spans and 256 symbols, 4 dBm, so that CI runs it in about a
//! second, and the reference has 4000 uniform steps per span (25 m).
//!
//! The test measures the normalised squared error of eq. (12) of Sinkin et al. (squared),
//! $\lVert u_{K} - u_{\mathrm{ref}} \rVert^{2} / \lVert u_{\mathrm{ref}} \rVert^{2}$, of every
//! technique of the solver with $K$ steps per span: uniform steps with the phase at the
//! middle of the step (the default), with the effective length of Optilux and with the
//! trapezoidal mean of OptiCommPy, the logarithmic steps of Bosco et al. (the paper's),
//! and the single-precision path. The card `validation/cards/hager2021_convergence.yaml`
//! holds the claims: the reference is converged, every scheme is of order 2 and tends to
//! the same field, and the paper's 500 logarithmic steps leave an error far below the
//! noise of the amplifiers of the link (the ASE in the band of the channel).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::optical::edfa::ase_density;
use comnum::channel::optical::fiber::{FiberParams, Precision, Ssfm};
use comnum::core::rng::Rng;
use comnum::dsp::filter::rrc_spectrum;
use comnum::math::fft::{fft, ifft};
use comnum::math::nlse::{NonlinearStep, StepRule};
use comnum::math::{db2lin, dbm2w, lin2db};
use comnum_validation::card::Card;
use serde_json::json;

const RATE: f64 = 32e9;
const SPS: usize = 6;
const SYMBOLS: usize = 256;
const SPANS: usize = 2;
const POWER_DBM: f64 = 4.0;
/// The steps per span of the reference (25 m).
const REFERENCE_STEPS: usize = 4000;

/// The link of Table I of the paper, cut to [`SPANS`] spans.
fn fiber() -> FiberParams {
    FiberParams {
        n_spans: SPANS,
        span_km: 100.0,
        alpha_db_km: 0.2,
        gamma_w_km: 1.3,
        d_ps_nm_km: 17.20934,
        fc_hz: 194.6e12,
        nf_db: 5.0,
        slope_ps_nm2_km: 0.0,
    }
}

/// Gaussian symbols shaped by a root-raised cosine on the periodic block, at
/// [`POWER_DBM`].
fn launch() -> Vec<C64> {
    let mut rng = Rng::new(8);
    let a: Vec<C64> = (0..SYMBOLS).map(|_| rng.complex_gaussian()).collect();
    let spectrum = fft(&a);
    let shape = rrc_spectrum(SYMBOLS, SPS, 0.1);
    let up: Vec<C64> = (0..SYMBOLS * SPS).map(|k| spectrum[k % SYMBOLS] * shape[k]).collect();
    let mut x = ifft(&up);
    let p = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
    let scale = (dbm2w(POWER_DBM) / p).sqrt();
    for v in &mut x {
        *v *= scale;
    }
    x
}

/// $\lVert a - b \rVert^{2} / \lVert b \rVert^{2}$ in dB.
fn error_db(a: &[C64], b: &[C64]) -> f64 {
    lin2db(a.iter().zip(b).map(|(u, v)| (u - v).norm_sqr()).sum::<f64>() / b.iter().map(C64::norm_sqr).sum::<f64>())
}

/// The techniques compared, by their name in the card.
const SCHEMES: [&str; 5] = ["midpoint", "effective_length", "trapezoidal", "logarithmic", "f32"];

/// The field after the link with `k` steps per span of the technique `scheme`.
fn propagate(x0: &[C64], scheme: &str, k: usize) -> Vec<C64> {
    let base = Ssfm::new(fiber(), 100.0 / k as f64);
    let (solver, precision) = match scheme {
        "midpoint" => (base, Precision::F64),
        "effective_length" => (Ssfm { nonlinear: NonlinearStep::EffectiveLength, ..base }, Precision::F64),
        "trapezoidal" => (Ssfm { nonlinear: NonlinearStep::Trapezoidal, ..base }, Precision::F64),
        "logarithmic" => (Ssfm { rule: Some(StepRule::Logarithmic(k)), ..base }, Precision::F64),
        "f32" => (base, Precision::F32),
        _ => unreachable!("unknown scheme {scheme}"),
    };
    let mut x = x0.to_vec();
    solver.propagate_with(&mut x, RATE * SPS as f64, None, precision).unwrap();
    x
}

#[test]
fn the_split_step_converges_to_its_fine_step_solution_on_the_link_of_fig_8() {
    let x0 = launch();
    let reference = propagate(&x0, "midpoint", REFERENCE_STEPS);
    let mut card = Card::load("hager2021_convergence").unwrap();

    // the reference against itself with twice the step: its own error is 1/16 of that
    let half = propagate(&x0, "midpoint", REFERENCE_STEPS / 2);
    let self_error = error_db(&half, &reference);
    println!("reference: {REFERENCE_STEPS} against {} steps per span: {self_error:.1} dB", REFERENCE_STEPS / 2);
    card.check(&json!({"claim": "reference_converged"}), self_error);

    // the noise of the amplifiers in the band of the channel, relative to the signal
    let f = fiber();
    let ase = SPANS as f64 * ase_density(db2lin(f.span_gain_db()), db2lin(f.nf_db), f.fc_hz) * RATE;
    let noise_db = lin2db(ase / dbm2w(POWER_DBM));
    println!("ASE in the band of the channel: {noise_db:.1} dB of the signal");

    for scheme in SCHEMES {
        let errors: Vec<(usize, f64)> = [10, 20, 50, 100, 200, 500, 1000]
            .iter()
            .map(|&k| (k, error_db(&propagate(&x0, scheme, k), &reference)))
            .collect();
        let row: Vec<String> = errors.iter().map(|(k, e)| format!("{k}: {e:.1}")).collect();
        println!("{scheme:>16}: {}", row.join(", "));
        let at = |k: usize| errors.iter().find(|(n, _)| *n == k).unwrap().1;
        if scheme == "f32" {
            // the rounding of single precision accumulates over the steps: a floor
            let floor = errors.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min);
            card.check(&json!({"scheme": scheme, "claim": "floor_below_noise"}), floor - noise_db);
            continue;
        }
        // the order on the field: the squared error falls by 2p × 3.01 dB per halving
        // (the logarithmic steps reach it only past 500 steps per span: their longest
        // steps, at the end of the span, decide the error before)
        let (k1, k2) = if scheme == "logarithmic" { (500, 1000) } else { (50, 100) };
        let order = (at(k1) - at(k2)) / (2.0 * lin2db(2.0));
        card.check(&json!({"scheme": scheme, "claim": "order"}), order);
        card.check(&json!({"scheme": scheme, "claim": "converged"}), at(1000));
        if scheme == "logarithmic" {
            card.check(&json!({"claim": "paper_steps_below_noise"}), at(500) - noise_db);
            let uniform = error_db(&propagate(&x0, "midpoint", 100), &reference);
            card.check(&json!({"claim": "logarithmic_over_midpoint"}), at(100) - uniform);
        }
        if scheme == "effective_length" {
            card.check(
                &json!({"claim": "effective_length_over_midpoint"}),
                at(100) - error_db(&propagate(&x0, "midpoint", 100), &reference),
            );
        }
    }
    card.finish().unwrap();
}
