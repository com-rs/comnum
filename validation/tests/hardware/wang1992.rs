//! The small-signal response of a dispersive IM-DD link (`comnum::hardware::optical`:
//! a `DirectlyModulatedLaser`, `linear_fiber_channel`, square-law detection) against the
//! closed form of J. Wang and K. Petermann, "Small signal analysis for dispersive optical
//! fiber communication systems," *J. Lightw. Technol.* 10(1), 96–100, 1992, in the form of
//! F. Devaux, Y. Sorel and J. F. Kerdiles, "Simple measurement of fiber dispersion and of
//! chirp parameter of intensity modulated light emitter," *J. Lightw. Technol.* 11(12),
//! 1937–1940, 1993 (`direct_detection::dispersion_fading_response`,
//! `direct_detection::fading_frequency_hz`): for the power $P_{0}(1 + m \cos \Omega t)$
//! of a laser of linewidth enhancement $\beta_{c}$ and adiabatic coefficient $\kappa$,
//! the detected modulation is $m |H(\Omega)|$ with
//!
//! $$
//! H(\Omega) = \cos\theta + \beta_{c}\sin\theta - j\,\frac{\beta_{c}\kappa P_{0}}{\Omega}\sin\theta,
//! \qquad \theta = \frac{\beta_{2}\Omega^{2}L}{2},
//! $$
//!
//! which vanishes (without adiabatic chirp) at $f_{u}^{2} = [1 + 2u - \frac{2}{\pi}
//! \arctan\beta_{c}] / (4\pi|\beta_{2}|L)$. The sign of $\beta_{c}$ decides whether the
//! first notch comes closer or moves away: the test checks the convention of the chirp of
//! the laser against that of the fibre.
//!
//! The link: 40 km of standard fibre ($D$ = 17 ps/(nm km) at 193.1 THz, $\beta_{2}$ =
//! −21.7 ps²/km, no slope, no loss), $P_{0}$ = 1 mW, $m$ = 0.02, 2^14 samples at 64 GHz;
//! the laser without bandwidth limit, $\beta_{c}$ = 0, 3 and −3 (transient chirp), and
//! $\beta_{c}$ = 3 with $\kappa$ = 10^13 Hz/W. The modulation is read on the spectrum of
//! the detected power, $2|X(\Omega)|/(m X(0))$, at frequencies on the grid; the first
//! notch is the frequency of the minimum of that reading between $0.7 f_{0}$ and $1.3
//! f_{0}$ in steps of 50 MHz (the floor of the second order hides the first notch of
//! $\beta_{c} = 3$ behind the second one on a wider search), divided by $f_{0}$.
//! The values and tolerances are in the card `validation/cards/wang1992_small_signal.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::optical::fiber::{FiberParams, linear_fiber_channel};
use comnum::hardware::optical::DirectlyModulatedLaser;
use comnum::hardware::optical::direct_detection::{dispersion_fading_response, fading_frequency_hz};
use comnum::math::fft::fft;
use comnum_validation::card::Card;
use serde_json::json;
use std::f64::consts::PI;
use std::fmt::Write as _;

const N: usize = 1 << 14;
const FS: f64 = 64e9;
const P0: f64 = 1e-3;
const M: f64 = 0.02;
const LENGTH_KM: f64 = 40.0;

fn fiber() -> FiberParams {
    FiberParams { n_spans: 1, span_km: LENGTH_KM, slope_ps_nm2_km: 0.0, alpha_db_km: 0.0, ..FiberParams::default() }
}

/// The frequency of the bin nearest to `f`.
fn on_grid(f: f64) -> f64 {
    (f * N as f64 / FS).round() * FS / N as f64
}

/// The detected modulation at `f` (on the grid), relative to `M`.
fn measured(dml: &DirectlyModulatedLaser, f: f64) -> f64 {
    let p: Vec<f64> = (0..N).map(|n| P0 * (1.0 + M * (2.0 * PI * f * n as f64 / FS).cos())).collect();
    let field = dml.field(&p, FS).unwrap();
    let out = linear_fiber_channel(&field, &fiber(), LENGTH_KM, FS);
    let power: Vec<_> = out.iter().map(|e| comnum::C64::new(e.norm_sqr(), 0.0)).collect();
    let s = fft(&power);
    let bin = (f * N as f64 / FS).round() as usize;
    2.0 * s[bin].norm() / (M * s[0].norm())
}

fn theory(dml: &DirectlyModulatedLaser, f: f64) -> f64 {
    dispersion_fading_response(
        f,
        dml.linewidth_enhancement,
        dml.adiabatic_chirp_hz_w,
        P0,
        fiber().beta2_s2_km(),
        LENGTH_KM,
    )
    .unwrap()
    .norm()
}

const FREQUENCIES_GHZ: [f64; 6] = [1.0, 2.0, 3.0, 6.0, 8.0, 12.0];

#[test]
fn the_imdd_link_follows_the_small_signal_response_of_wang_1992() {
    let mut card = Card::load("wang1992_small_signal").unwrap();
    for (chirp, kappa) in [(0.0, 0.0), (3.0, 0.0), (-3.0, 0.0), (3.0, 1e13)] {
        let dml = DirectlyModulatedLaser {
            linewidth_enhancement: chirp,
            adiabatic_chirp_hz_w: kappa,
            bandwidth_hz: 0.0,
            ..Default::default()
        };
        let name = if kappa > 0.0 { "adiabatic".to_string() } else { format!("{chirp}") };
        let mut row = format!("beta_c {chirp}, kappa {kappa:.0e}:");
        for f_ghz in FREQUENCIES_GHZ {
            let f = on_grid(f_ghz * 1e9);
            let (meas, th) = (measured(&dml, f), theory(&dml, f));
            write!(row, " {f_ghz} GHz {meas:.4}/{th:.4}").unwrap();
            card.check(&json!({"beta_c": name, "f_ghz": f_ghz}), meas / th);
        }
        if kappa == 0.0 {
            // the first notch: the minimum of the reading over the grid, against f_0
            let f0 = fading_frequency_hz(0, chirp, fiber().beta2_s2_km(), LENGTH_KM).unwrap();
            let (lo, hi) = ((0.7 * f0 / 50e6) as usize, (1.3 * f0 / 50e6) as usize);
            let (f_min, _) = (lo..=hi)
                .map(|k| on_grid(k as f64 * 50e6))
                .map(|f| (f, measured(&dml, f)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            write!(row, "; notch {:.3} GHz (theory {:.3})", f_min / 1e9, f0 / 1e9).unwrap();
            card.check(&json!({"beta_c": name, "notch": 0}), f_min / f0);
        }
        println!("{row}");
    }
    card.finish().unwrap();
}
