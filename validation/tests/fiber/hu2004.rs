//! The pump-power optimiser (`Raman::design_pumps`) against J. Hu, B. S. Marks and
//! C. R. Menyuk, "Flat-gain fiber Raman amplifiers using equally spaced pumps," *J.
//! Lightw. Technol.* 22(6), 1519–1522, 2004, Table I, case I: the signal gain ripple of
//! 2, 4 and 8 pumps equally spaced in wavelength around 1450 nm over signal bands of 20,
//! 40 and 80 nm, at the best spacing.
//!
//! The link of Sec. III: 50 km, backward-pumped, CW signals of −3 dBm every 0.8 nm over
//! 1540–1560, 1530–1570 or 1525–1605 nm; the gain spectrum of Fig. 1(a) (silica at a
//! pump of 1 µm, the same curve as Perlin & Winful 2002 Fig. 1, read from the rasters
//! at 0.5 THz: [`silica_gain_1um`]) scaled with the pump frequency and halved for
//! depolarised pumps ($g_{jk} = g_{j}/(2 A_{\mathrm{eff}})$, Sec. II; $A_{\mathrm{eff}}$
//! is not given: 50 µm², Perlin & Winful's), the loss profile of Fig. 1(b) read at 10 nm
//! ([`hu_loss_db_km`]). The signals' loss varies over the band, which the solver does not
//! take (one loss for the signals): the target on-off gain of every signal is its own
//! loss over the span, the flat net gain of the paper, and the solver's signal loss is
//! the band's average. For each number of pumps and band, the spacing is scanned over the
//! range of Fig. 2 by the linear fit of the design alone (the minimax fit of the
//! integrated pump powers, seconds), then the full design (powers and profile) is run at
//! the best spacing and its ripple, the spread of the net gain over the signals, is the
//! value of the card `hu2004_table1`. The paper optimises by a genetic algorithm and a
//! shooting method; the profiles are readings of raster figures.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::raman::{DesignObjective, Direction, Raman, RamanGain, RamanPump};
use comnum::math::linalg::nnls_minimax;
use comnum::math::{SPEED_OF_LIGHT, nm2hz};
use comnum_validation::card::Card;
use ndarray::Array2;
use serde_json::json;

/// The Raman gain spectrum of silica at a pump of 1 µm (Perlin & Winful 2002, Fig. 1;
/// Hu et al. 2004, Fig. 1(a)), $g_{R}$ in units of 10⁻¹³ m/W at 0, 0.5, …, 40 THz (read
/// from the rasters at 300 dpi, ±0.015).
pub(crate) const SILICA_GAIN_1UM: [f64; 81] = [
    0.010, 0.044, 0.087, 0.139, 0.183, 0.210, 0.235, 0.260, 0.283, 0.304, 0.322, 0.344, 0.367, 0.396, 0.425, 0.470,
    0.524, 0.583, 0.651, 0.730, 0.812, 0.884, 0.932, 0.965, 0.978, 0.980, 0.974, 0.949, 0.903, 0.962, 0.870, 0.646,
    0.423, 0.242, 0.181, 0.187, 0.224, 0.204, 0.187, 0.125, 0.104, 0.083, 0.075, 0.075, 0.077, 0.087, 0.104, 0.156,
    0.187, 0.173, 0.137, 0.100, 0.067, 0.056, 0.048, 0.042, 0.038, 0.037, 0.035, 0.035, 0.038, 0.046, 0.054, 0.058,
    0.058, 0.052, 0.044, 0.037, 0.033, 0.029, 0.029, 0.031, 0.035, 0.035, 0.031, 0.027, 0.023, 0.016, 0.014, 0.012,
    0.010,
];

/// The gain spectrum of the two papers as a [`RamanGain`], $g_{R}/K_{\mathrm{eff}}$ (m/W)
/// for the polarisation factor `polarisation_factor` (2 for depolarised pumps, Hu et
/// al.'s $1/(2 A_{\mathrm{eff}})$), the pump at 1 µm.
pub(crate) fn silica_gain_1um(polarisation_factor: f64) -> RamanGain {
    let offsets: Vec<f64> = (0..81).map(|k| f64::from(k) * 0.5e12).collect();
    let gain: Vec<f64> = SILICA_GAIN_1UM.iter().map(|g| g * 1e-13 / polarisation_factor).collect();
    RamanGain::new(offsets, gain, SPEED_OF_LIGHT / 1e-6).unwrap()
}

/// The loss profile of Hu et al. Fig. 1(b) (dB/km) at 1400, 1410, …, 1700 nm (read on
/// the raster, ±0.01).
const HU_LOSS: [f64; 31] = [
    0.89, 0.80, 0.69, 0.62, 0.54, 0.48, 0.42, 0.37, 0.32, 0.29, 0.26, 0.24, 0.22, 0.20, 0.19, 0.18, 0.176, 0.176,
    0.176, 0.18, 0.19, 0.21, 0.22, 0.25, 0.27, 0.31, 0.35, 0.40, 0.46, 0.55, 0.62,
];

/// A loss profile tabulated at 1400, 1410, …, 1700 nm, linear between the points.
pub(crate) fn loss_db_km(table: &[f64; 31], wavelength_nm: f64) -> f64 {
    let x = ((wavelength_nm - 1400.0) / 10.0).clamp(0.0, 30.0);
    let i = (x.floor() as usize).min(29);
    table[i] + (x - i as f64) * (table[i + 1] - table[i])
}

/// The loss of Hu et al. Fig. 1(b) at `wavelength_nm`.
pub(crate) fn hu_loss_db_km(wavelength_nm: f64) -> f64 {
    loss_db_km(&HU_LOSS, wavelength_nm)
}

const SPAN_KM: f64 = 50.0;
const AEFF_UM2: f64 = 50.0;

/// The signals of a band of `band_nm` around 1550 nm (Sec. III) every 0.8 nm at −3 dBm:
/// their frequencies and the targets (the loss of each over the span, dB).
fn band(band_nm: f64) -> (Vec<(f64, f64)>, Vec<f64>) {
    let (lo, hi): (f64, f64) = match band_nm as i32 {
        20 => (1540.0, 1560.0),
        40 => (1530.0, 1570.0),
        _ => (1525.0, 1605.0),
    };
    let n = ((hi - lo) / 0.8).round() as usize + 1;
    let wavelengths: Vec<f64> = (0..n).map(|k| lo + k as f64 * 0.8).collect();
    let signals = wavelengths.iter().map(|w| (nm2hz(*w), 5e-4)).collect();
    let targets = wavelengths.iter().map(|w| hu_loss_db_km(*w) * SPAN_KM).collect();
    (signals, targets)
}

/// The Raman configuration of `n_pumps` counter-pumps spaced by `spacing_nm` around
/// 1450 nm, each with its loss of Fig. 1(b).
fn pumps(n_pumps: usize, spacing_nm: f64) -> Raman {
    let pumps = (0..n_pumps)
        .map(|j| {
            let w = 1450.0 + (j as f64 - (n_pumps as f64 - 1.0) / 2.0) * spacing_nm;
            RamanPump { alpha_db_km: Some(hu_loss_db_km(w)), ..RamanPump::at_wavelength(w, 0.0, Direction::Counter) }
        })
        .collect();
    Raman { pumps, gain: silica_gain_1um(2.0), aeff_um2: AEFF_UM2, ..Raman::default() }
}

/// The ripple (dB) of the linear fit alone: the minimax fit of the integrated pump
/// powers to the targets (no profile solved; the tilt of the signals left out).
fn fit_ripple(raman: &Raman, signals: &[(f64, f64)], targets_db: &[f64]) -> f64 {
    let (m, n) = (signals.len(), raman.pumps.len());
    let mut a = Array2::<f64>::zeros((m, n));
    for k in 0..m {
        for j in 0..n {
            a[[k, j]] = raman.coupling(signals[k].0, raman.pumps[j].frequency_hz) * 10.0 * std::f64::consts::LOG10_E;
        }
    }
    let x = nnls_minimax(&a.view(), targets_db, 200).unwrap();
    let errors: Vec<f64> = (0..m).map(|k| (0..n).map(|j| a[[k, j]] * x[j]).sum::<f64>() - targets_db[k]).collect();
    errors.iter().copied().fold(f64::MIN, f64::max) - errors.iter().copied().fold(f64::MAX, f64::min)
}

/// The ripple of the full design at the best spacing: the five best spacings of the scan
/// of the linear fit (which leaves the Raman tilt of the signals out) are designed in
/// full, and the best kept: (spacing, ripple).
fn best_design(n_pumps: usize, band_nm: f64) -> (f64, f64) {
    let (signals, targets) = band(band_nm);
    let (max_spacing, step) = match n_pumps {
        2 => (150.0, 1.0),
        4 => (50.0, 0.5),
        _ => (20.0, 0.25),
    };
    let mut scan: Vec<(f64, f64)> = Vec::new();
    let mut spacing = step;
    while spacing <= max_spacing {
        scan.push((spacing, fit_ripple(&pumps(n_pumps, spacing), &signals, &targets)));
        spacing += step;
    }
    // the local minima of the scan, the five best
    let mut minima: Vec<(f64, f64)> = (0..scan.len())
        .filter(|&i| (i == 0 || scan[i].1 <= scan[i - 1].1) && (i + 1 == scan.len() || scan[i].1 <= scan[i + 1].1))
        .map(|i| scan[i])
        .collect();
    minima.sort_by(|a, b| a.1.total_cmp(&b.1));
    let fiber = FiberParams {
        span_km: SPAN_KM,
        alpha_db_km: targets.iter().sum::<f64>() / targets.len() as f64 / SPAN_KM,
        ..FiberParams::default()
    };
    let mut best = (0.0, f64::INFINITY);
    for (spacing, fit) in minima.iter().take(5) {
        let raman = Raman { step_km: 0.5, ..pumps(n_pumps, *spacing) };
        let d = raman.design_pumps(&fiber, &signals, &targets, DesignObjective::Minimax).unwrap();
        println!(
            "{n_pumps} pumps, {band_nm} nm: spacing {spacing} nm, fit ripple {fit:.3} dB, design ripple {:.3} dB, powers {:?} mW",
            d.ripple_db,
            d.pumps.iter().map(|p| (p.power_w * 1e3).round()).collect::<Vec<_>>()
        );
        if d.ripple_db < best.1 {
            best = (*spacing, d.ripple_db);
        }
    }
    best
}

#[test]
#[ignore = "70 s: `just card hu2004_table1`"]
fn hu_2004_table_i_case_i_ripple_of_equally_spaced_pumps() {
    let mut card = Card::load("hu2004_table1").unwrap();
    for inputs in card.inputs_matching(&json!({"case": "I"})) {
        let n = inputs["pumps"].as_u64().unwrap() as usize;
        let b = inputs["band_nm"].as_f64().unwrap();
        card.check(&inputs, best_design(n, b).1);
    }
    card.finish().unwrap();
}
