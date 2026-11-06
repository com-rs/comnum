//! The SIR of OFDM and FBMC over doubly-selective channels against R. Nissel, S. Schwarz
//! and M. Rupp, "Filter bank multicarrier modulation schemes for future mobile
//! communications," *IEEE J. Sel. Areas Commun.* 35(8), 1768–1782, 2017
//! (`references/multicarrier/2017_nissel_fbmc-schemes-future-mobile_jsac.pdf`):
//!
//! - Figs. 5–8: the SIR of each scheme at its best subcarrier spacing $F$ (a golden-section
//!   search on $\log F$ above the lower bound of the figure, 5 kHz at 2.5 GHz and 100 kHz
//!   at 60 GHz) at 0, 100, 200 and 500 km/h over Pedestrian A and Vehicular A (ITU-R
//!   M.1225, 2.5 GHz), TDL-A of 30 ns and TDL-B of 900 ns (TR 38.901, 60 GHz), Jakes'
//!   Doppler. CP-OFDM with the prefix on the delay spread is the closed form of their eq.
//!   (42) (`ofdm::budget::jakes_ici_sir`, $F = \kappa/\tau_{\max}$, $\kappa = 1/14$, or 1
//!   for $TF = 2$ in Fig. 8); OFDM without prefix, FBMC-OQAM (Hermite with $K = 8$,
//!   PHYDYAS with $K = 4$) and FBMC-QAM at $TF = 2$ (Fig. 8: every other subcarrier and
//!   half-symbol of the OQAM lattice) are the matrix SIR of `comnum::waveform::sir`,
//!   eqs. (35) and (40), on a grid of at least 256 samples per $T_{0}$ and at most an
//!   eighth of the RMS delay spread (the delays are rounded to it): OFDM and FBMC-OQAM on
//!   their complete lattices in the Parseval form (`sir_complete`, every interferer
//!   counted), FBMC-QAM as the sum over 9 subcarriers and $2K + 1$ symbols. The paper gives
//!   neither $L$, $K$ nor its sampling rate.
//!
//! The values read on the figures (to about 0.7 dB), the labels of the optimal spacings
//! and the tolerances are in `validation/cards/nissel2017jsac_eq42.yaml` (CP-OFDM, CI) and
//! `validation/cards/nissel2017jsac_figs5_8.yaml` (the matrix SIRs, a campaign).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::multipath::{Profile, max_doppler_hz};
use comnum::math::{golden_max, lin2db};
use comnum::waveform::fbmc::{Fbmc, Prototype};
use comnum::waveform::ofdm::Ofdm;
use comnum::waveform::ofdm::budget::jakes_ici_sir;
use comnum::waveform::sir::{DoublySelective, Pulse};
use comnum_validation::card::Card;
use serde_json::json;

/// The fewest samples per $T_{0}$.
const M_MIN: usize = 256;
/// Subcarriers on each side of the wanted one of FBMC-QAM (the prototypes span two).
const SIDE: usize = 4;

/// Samples per $T_{0}$ at the spacing `f_hz` over `profile`: at least [`M_MIN`], and
/// enough for a sampling interval of an eighth of the RMS delay spread (the delays are
/// rounded to the grid).
fn grid(f_hz: f64, profile: &Profile) -> usize {
    let tau_rms = comnum::channel::multipath::rms_delay_spread(&profile.taps());
    M_MIN.max((8.0 / (f_hz * tau_rms)).ceil() as usize).next_multiple_of(2)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scheme {
    Ofdm,
    Oqam(Prototype),
    Qam(Prototype),
}

fn channel(profile: &Profile, sample_rate: f64, doppler_hz: f64) -> DoublySelective {
    DoublySelective::new(&profile.taps(), sample_rate, doppler_hz).unwrap()
}

/// The SIR (dB) of `scheme` at the subcarrier spacing `f_hz` over `profile` at `doppler_hz`.
fn sir_db(scheme: Scheme, f_hz: f64, profile: &Profile, doppler_hz: f64) -> f64 {
    let m = grid(f_hz, profile);
    let sir = match scheme {
        Scheme::Ofdm => {
            // every subcarrier of every symbol: a complete basis, the Parseval form
            let ofdm = Ofdm { oversampling: 1, n_cp: 0, ..Ofdm::new(m, m).unwrap() };
            let ch = channel(profile, f_hz * m as f64, doppler_hz);
            ch.sir_complete(&ofdm.window(m / 2, 1), &ofdm.pulse(m / 2, 1))
        }
        Scheme::Oqam(p) => {
            // the whole OQAM lattice: a complete real frame, the Parseval form
            let k = if p == Prototype::Hermite { 8 } else { 4 };
            let fbmc = Fbmc::with_prototype(m, k, p).unwrap();
            // T0 = M samples and F = 1/T0
            let ch = channel(profile, f_hz * m as f64, doppler_hz);
            let g = fbmc.pulse(m / 2, 2 * k);
            ch.sir_complete(&g, &g)
        }
        Scheme::Qam(p) => {
            let k = if p == Prototype::Hermite { 8 } else { 4 };
            // F = 2/T0: the spacing f_hz is two subcarriers of the lattice
            let m = grid(f_hz / 2.0, profile);
            let fbmc = Fbmc::with_prototype(m, k, p).unwrap();
            let ch = channel(profile, f_hz * m as f64 / 2.0, doppler_hz);
            let (m0, n0) = (m / 2, 2 * k);
            let tx: Vec<Pulse> = (0..=2 * k)
                .flat_map(|n| (0..=2 * SIDE).map(move |m| (m0 - 2 * SIDE + 2 * m, 2 * n)))
                .map(|(m, n)| fbmc.pulse(m, n))
                .collect();
            ch.sir_qam(&fbmc.pulse(m0, n0), &tx, k * (2 * SIDE + 1) + SIDE)
        }
    };
    lin2db(sir)
}

/// The best SIR (dB) over the subcarrier spacing in `[f_min, f_max]` (Hz), and that spacing.
fn best(scheme: Scheme, profile: &Profile, doppler_hz: f64, f_min: f64, f_max: f64) -> (f64, f64) {
    let log_f = golden_max(|x| sir_db(scheme, 10f64.powf(x), profile, doppler_hz), f_min.log10(), f_max.log10(), 14);
    let f = 10f64.powf(log_f);
    (sir_db(scheme, f, profile, doppler_hz), f)
}

fn scheme(name: &str) -> Scheme {
    match name {
        "ofdm" => Scheme::Ofdm,
        "oqam_hermite" => Scheme::Oqam(Prototype::Hermite),
        "oqam_phydyas" => Scheme::Oqam(Prototype::Phydyas),
        "qam_hermite" => Scheme::Qam(Prototype::Hermite),
        "qam_phydyas" => Scheme::Qam(Prototype::Phydyas),
        other => panic!("unknown scheme {other}"),
    }
}

/// The channel, carrier and lower bound on the spacing of a figure.
fn setting(figure: u64) -> (Profile, f64, f64) {
    match figure {
        5 => (Profile::PedestrianA, 2.5e9, 5e3),
        6 => (Profile::VehicularA, 2.5e9, 5e3),
        7 => (Profile::TdlA { delay_spread_s: 30e-9 }, 60e9, 100e3),
        8 => (Profile::TdlB { delay_spread_s: 900e-9 }, 60e9, 100e3),
        _ => unreachable!(),
    }
}

#[test]
fn nissel_2017_jsac_eq_42_cp_ofdm_in_closed_form() {
    let mut card = Card::load("nissel2017jsac_eq42").unwrap();
    for inputs in card.inputs_matching(&json!({"quantity": "sir_db"})) {
        let (figure, kmh) = (inputs["figure"].as_u64().unwrap(), inputs["kmh"].as_f64().unwrap());
        let (profile, fc, _) = setting(figure);
        let kappa = if figure == 8 { 1.0 } else { 1.0 / 14.0 };
        let tau_max = profile.taps().iter().map(|t| t.0).fold(0.0, f64::max);
        let fd = max_doppler_hz(kmh / 3.6, fc);
        let f = kappa / tau_max;
        let sir = lin2db(jakes_ici_sir(fd / f));
        println!("Fig. {figure}, {kmh} km/h: CP-OFDM F = {:.1} kHz, SIR {sir:.2} dB", f / 1e3);
        card.check(&inputs, sir);
        let label = json!({"figure": figure, "kmh": kmh, "quantity": "spacing_khz"});
        if !card.inputs_matching(&label).is_empty() {
            card.check(&label, f / 1e3);
        }
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "closed forms on matrices: a few minutes"]
fn nissel_2017_jsac_figs_5_to_8_optimal_sir() {
    let mut card = Card::load("nissel2017jsac_figs5_8").unwrap();
    for inputs in card.inputs_matching(&json!({"quantity": "sir_db"})) {
        let name = inputs["scheme"].as_str().unwrap();
        let (figure, kmh) = (inputs["figure"].as_u64().unwrap(), inputs["kmh"].as_f64().unwrap());
        let (profile, fc, f_min) = setting(figure);
        let fd = max_doppler_hz(kmh / 3.6, fc);
        let t = std::time::Instant::now();
        let (sir, f) = best(scheme(name), &profile, fd, f_min, 50.0 * f_min);
        println!(
            "Fig. {figure}, {kmh} km/h, {name}: SIR {sir:.2} dB at F = {:.1} kHz ({:.0} s)",
            f / 1e3,
            t.elapsed().as_secs_f64()
        );
        card.check(&inputs, sir);
        let label = json!({"figure": figure, "kmh": kmh, "scheme": name, "quantity": "spacing_khz"});
        if !card.inputs_matching(&label).is_empty() {
            card.check(&label, f / 1e3);
        }
    }
    card.finish().unwrap();
}
