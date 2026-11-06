//! The Raman module (`comnum::channel::optical::raman`) against M. N. Islam, "Raman
//! amplifiers for telecommunications," *IEEE J. Sel. Topics Quantum Electron.* 8(3),
//! 548–559, 2002: the review that gives, with every fibre parameter, the double Rayleigh
//! backscattering of a Raman-pumped span and a depleted single-pump amplifier.
//!
//! - **Fig. 6(b)** (`islam2002_fig6b`): the effective Rayleigh reflection coefficient $R^{2}$
//!   (dB) against the on-off gain, for SMF-28 of 80 and 100 km (0.25 dB/km at the signal,
//!   0.29 at the pump, 1.62 dB/(W·km), Rayleigh coefficient −42.3 dB/km) and the
//!   dispersion-compensating fibres DK-40 (6.5 km) and DK-80 (13 km) (0.49 and 0.63 dB/km,
//!   13.5 dB/(W·km), −33.8 dB/km), pumped at 1451 nm, signal at 1550 nm. The paper does
//!   not state the pump direction: the pump is counter-propagating (the distributed
//!   amplifier of its Sec. III); for a small signal the coefficient is the same in both
//!   directions (the double integral of the module is invariant under $z \to L - z$;
//!   within 3·10⁻³ dB on the grid of 100 m). The
//!   pump power of every on-off gain is found by bisection; the signal is 1 µW (no
//!   depletion). The Rayleigh coefficient of the paper is read as $10 \log_{10} \kappa$
//!   with $\kappa$ in 1/km: without pump it gives the −57 dB (SMF-28, DK-40) and −52 dB
//!   (DK-80) of the figure.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::raman::{Direction, Raman, RamanGain, RamanPump};
use comnum::math::{bisect, db2np, lin2db, nm2hz};
use comnum_validation::card::Card;
use serde_json::json;

const SIGNAL_NM: f64 = 1550.0;
const PUMP_NM: f64 = 1451.0;

/// A fibre of Fig. 6(b): length (km), loss at the signal and at the pump (dB/km), the
/// Raman gain coefficient (dB/(W·km)) and the Rayleigh scattering coefficient (dB/km).
struct Fibre {
    length_km: f64,
    alpha_s: f64,
    alpha_p: f64,
    gain_db_w_km: f64,
    rayleigh_db_km: f64,
}

fn fibre(name: &str) -> Fibre {
    match name {
        "smf28_80km" => {
            Fibre { length_km: 80.0, alpha_s: 0.25, alpha_p: 0.29, gain_db_w_km: 1.62, rayleigh_db_km: -42.3 }
        }
        "smf28_100km" => {
            Fibre { length_km: 100.0, alpha_s: 0.25, alpha_p: 0.29, gain_db_w_km: 1.62, rayleigh_db_km: -42.3 }
        }
        "dk40" => Fibre { length_km: 6.5, alpha_s: 0.49, alpha_p: 0.63, gain_db_w_km: 13.5, rayleigh_db_km: -33.8 },
        "dk80" => Fibre { length_km: 13.0, alpha_s: 0.49, alpha_p: 0.63, gain_db_w_km: 13.5, rayleigh_db_km: -33.8 },
        other => panic!("no fibre {other}"),
    }
}

/// The Raman configuration of `f` with one pump of `power_w` in `direction`.
fn raman(f: &Fibre, power_w: f64, direction: Direction) -> Raman {
    let pump = RamanPump { alpha_db_km: Some(f.alpha_p), ..RamanPump::at_wavelength(PUMP_NM, power_w, direction) };
    Raman {
        pumps: vec![pump],
        gain: RamanGain::flat(db2np(f.gain_db_w_km), 80.0, nm2hz(PUMP_NM)).unwrap(),
        aeff_um2: 80.0,
        isrs: false,
        rayleigh_db_km: Some(f.rayleigh_db_km),
        ..Raman::default()
    }
}

/// $R^{2}$ (dB) of the small-signal span of `f` pumped to `on_off_db` of on-off gain.
fn reflection_db(f: &Fibre, on_off_db: f64, direction: Direction) -> f64 {
    let fiber = FiberParams { span_km: f.length_km, alpha_db_km: f.alpha_s, ..FiberParams::default() };
    let signal = [(nm2hz(SIGNAL_NM), 1e-6)];
    let solve = |w: f64| raman(f, w, direction).solve(&fiber, &signal).unwrap();
    // the bracket from the small-signal closed form (Bromage 2004, eq. (5))
    let a_p = db2np(f.alpha_p);
    let hi = 1.5 * db2np(on_off_db) / (db2np(f.gain_db_w_km) * (1.0 - (-a_p * f.length_km).exp()) / a_p);
    let power = if on_off_db <= 0.0 { 0.0 } else { bisect(|w| solve(w).on_off_gain_db(0) < on_off_db, 0.0, hi, 60) };
    let p = solve(power);
    assert!((p.on_off_gain_db(0) - on_off_db).abs() < 1e-6);
    lin2db(p.double_rayleigh[0])
}

#[test]
fn islam_2002_fig_6b_double_rayleigh_against_the_on_off_gain() {
    let mut card = Card::load("islam2002_fig6b").unwrap();
    for inputs in card.inputs_matching(&json!({})) {
        let f = fibre(inputs["fibre"].as_str().unwrap());
        let gain = inputs["on_off_db"].as_f64().unwrap();
        let r = reflection_db(&f, gain, Direction::Counter);
        // the same with a co-propagating pump, to 10⁻⁶ dB (the integral is symmetric)
        // the same with a co-propagating pump (the integral is symmetric; the trapezoids
        // of the two mirrored profiles differ by their discretisation, 3·10⁻³ dB at 30 dB)
        assert!((r - reflection_db(&f, gain, Direction::Co)).abs() < 1e-2);
        card.check(&inputs, r);
    }
    card.finish().unwrap();
}
