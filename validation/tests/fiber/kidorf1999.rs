//! The multi-pump Raman solver (`comnum::channel::optical::raman`, `Raman::solve`)
//! against H. Kidorf, K. Rottwitt, M. Nissov, M. Ma and E. Rabarijaona, "Pump
//! interactions in a 100-nm bandwidth Raman amplifier," *IEEE Photon. Technol. Lett.*
//! 11(5), 530–532, 1999: eight counter-propagating pumps from 1416 to 1502 nm in 45 km
//! of pure silica core fibre, 100 signals from 1520 to 1619 nm, 1 nm apart.
//!
//! - **Fig. 3, 4 and 5** (`kidorf1999_fig5`, `kidorf1999_fig4`): the optimised design.
//!   The launch powers of the pumps are the bars of Fig. 3, the signals −3.0 dBm each
//!   (Fig. 3; the 50 mW of the text over 100 channels). Fig. 5 gives the power of every
//!   pump at the far end of the span (z = 0) and the maximum of the 1502 nm pump, which
//!   gains from the shorter pumps; Fig. 4 and Sec. IV give the net gain of the fibre
//!   (peak-to-peak ripple 1.1 dB, text) and the optical SNR in 100 GHz (34.6 dB at
//!   1520 nm, 39.1 dB at 1619 nm, text).
//! - **Fig. 2** (`kidorf1999_fig2`): the first design, eight pumps of 120 mW evenly
//!   spaced from 1432 to 1516 nm, whose gain varies by 10.5 dB (text).
//!
//! The gain spectrum is Fig. 1, the measured $\frac{1}{2} g_{R}/A_{\mathrm{eff}}$ of the
//! fibre for an unpolarised pump at 1452 nm, read every 2 nm from 1462 to 1600 nm
//! ([`FIG1`]; the axis is labelled "1/2 Gr/Aeff *1000" without unit, read as
//! 1/(W·km): its peak, 0.43, is the depolarised efficiency of standard fibre, 0.41 for
//! the spectrum of GNPy). The figure stops at 19.1 THz of offset; the pumps reach the
//! signals up to 26.5 THz away: beyond the figure the spectrum is the shape of
//! [`RamanGain::ssmf`] scaled to the last reading (an assumption). The loss is given as
//! a minimum, 0.170 dB/km at 1563 nm (Sec. III): the signals take 0.170 dB/km, the
//! pumps the Rayleigh scaling $0.170 (1563/\lambda)^{4}$ dB/km (0.25 at 1416 nm, 0.20 at
//! 1502 nm; an assumption). Fig. 2 does not state the pump direction: counter, like the
//! design of Fig. 5. The paper's model also has the Rayleigh backscattering of every
//! wave, which does not change the gain (Hu et al. 2004, Sec. II).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use std::sync::OnceLock;

use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::raman::{Direction, Raman, RamanGain, RamanProfile, RamanPump};
use comnum::math::{SPEED_OF_LIGHT, dbm2w, lin2db, nm2hz};
use comnum_validation::card::Card;
use serde_json::json;

/// Fig. 1: $\frac{1}{2} g_{R}/A_{\mathrm{eff}}$ (1/(W·km)) for a pump at 1452 nm, at
/// 1462, 1464, …, 1600 nm (read from the raster at 600 dpi, column by column, ±0.005).
const FIG1: [f64; 70] = [
    0.064, 0.085, 0.097, 0.105, 0.111, 0.117, 0.122, 0.126, 0.128, 0.129, 0.129, 0.131, 0.132, 0.134, 0.136, 0.138,
    0.141, 0.144, 0.149, 0.154, 0.161, 0.169, 0.175, 0.184, 0.194, 0.203, 0.213, 0.224, 0.235, 0.245, 0.254, 0.264,
    0.273, 0.284, 0.292, 0.314, 0.326, 0.336, 0.353, 0.365, 0.375, 0.386, 0.398, 0.407, 0.417, 0.423, 0.425, 0.418,
    0.410, 0.394, 0.400, 0.431, 0.429, 0.371, 0.290, 0.219, 0.173, 0.147, 0.129, 0.115, 0.106, 0.100, 0.099, 0.106,
    0.118, 0.140, 0.140, 0.120, 0.095, 0.077,
];
const PUMP_NM_FIG1: f64 = 1452.0;
const AEFF_UM2: f64 = 80.0;
const SPAN_KM: f64 = 45.0;
const SIGNAL_LOSS_DB_KM: f64 = 0.170;
const SIGNAL_DBM: f64 = -3.0;
const OSNR_BANDWIDTH_HZ: f64 = 100e9;

/// The launch powers of the optimised design (dBm), read on the bars of Fig. 3 (±0.15).
const FIG3_PUMPS: [(f64, f64); 8] = [
    (1416.0, 21.33),
    (1421.0, 21.29),
    (1425.0, 21.29),
    (1430.0, 21.29),
    (1439.0, 20.95),
    (1456.0, 19.71),
    (1473.0, 20.13),
    (1502.0, 19.44),
];

/// The gain spectrum of Fig. 1 as a [`RamanGain`] ($g_{R}$ such that $g_{R}/A_{\mathrm{eff}}$
/// is the efficiency of the figure), extended beyond the figure by the shape of standard
/// fibre.
fn fig1_gain() -> RamanGain {
    let fp = nm2hz(PUMP_NM_FIG1);
    let aeff = AEFF_UM2 * 1e-12;
    let mut offsets = vec![0.0];
    let mut gain = vec![0.0];
    for (i, v) in FIG1.iter().enumerate() {
        offsets.push(fp - nm2hz(1462.0 + 2.0 * i as f64));
        gain.push(v * 1e-3 * aeff);
    }
    let (last_offset, last_gain) = (offsets[offsets.len() - 1], gain[gain.len() - 1]);
    let ssmf = RamanGain::ssmf();
    let scale = last_gain / ssmf.at(last_offset);
    let mut df = (last_offset / 0.5e12).floor() * 0.5e12 + 0.5e12;
    while df <= 40e12 {
        offsets.push(df);
        gain.push(scale * ssmf.at(df));
        df += 0.5e12;
    }
    RamanGain::new(offsets, gain, fp).unwrap()
}

/// The loss of a pump (dB/km): the Rayleigh scaling of the minimum loss of Sec. III.
fn pump_loss_db_km(wavelength_nm: f64) -> f64 {
    SIGNAL_LOSS_DB_KM * (1563.0 / wavelength_nm).powi(4)
}

/// The signals of the paper: 1520, 1521, …, 1619 nm at −3 dBm.
fn signals() -> Vec<(f64, f64)> {
    (0..100).map(|k| (nm2hz(1520.0 + f64::from(k)), dbm2w(SIGNAL_DBM))).collect()
}

/// The span with counter-propagating `pumps` (wavelength nm, launch power W).
fn solve(pumps: &[(f64, f64)]) -> RamanProfile {
    let raman = Raman {
        pumps: pumps
            .iter()
            .map(|&(nm, w)| RamanPump {
                alpha_db_km: Some(pump_loss_db_km(nm)),
                ..RamanPump::at_wavelength(nm, w, Direction::Counter)
            })
            .collect(),
        gain: fig1_gain(),
        aeff_um2: AEFF_UM2,
        step_km: 0.25,
        ..Raman::default()
    };
    let fiber = FiberParams {
        span_km: SPAN_KM,
        alpha_db_km: SIGNAL_LOSS_DB_KM,
        fc_hz: SPEED_OF_LIGHT / 1570e-9,
        ..FiberParams::default()
    };
    raman.solve(&fiber, &signals()).unwrap()
}

/// The optimised design of Fig. 3, solved once for the two cards.
fn optimised() -> &'static RamanProfile {
    static PROFILE: OnceLock<RamanProfile> = OnceLock::new();
    PROFILE.get_or_init(|| solve(&FIG3_PUMPS.map(|(nm, dbm)| (nm, dbm2w(dbm)))))
}

/// The peak-to-peak spread and the mean of the net gain of the signals (dB).
fn ripple_and_mean(p: &RamanProfile) -> (f64, f64) {
    let g: Vec<f64> = (0..p.n_signals).map(|k| p.net_gain_db(k)).collect();
    let max = g.iter().copied().fold(f64::MIN, f64::max);
    let min = g.iter().copied().fold(f64::MAX, f64::min);
    (max - min, g.iter().sum::<f64>() / g.len() as f64)
}

#[test]
fn kidorf_1999_fig_5_power_of_the_backward_pumps() {
    let p = optimised();
    let mut card = Card::load("kidorf1999_fig5").unwrap();
    let n = p.n_signals;
    let at_far_end = |j: usize| p.power_dbm(n + j)[0];
    for (j, (nm, launch)) in FIG3_PUMPS.iter().enumerate() {
        println!("pump {nm} nm: launched {launch:.2} dBm, {:.2} dBm at z = 0", at_far_end(j));
        card.check(&json!({"quantity": "power_at_z0_dbm", "pump_nm": nm}), at_far_end(j));
    }
    card.check(&json!({"quantity": "spread_1502_1416_db"}), at_far_end(7) - at_far_end(0));
    let profile = p.power_dbm(n + 7);
    let (i_max, max) = profile.iter().enumerate().fold((0, f64::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    println!("pump 1502 nm: maximum {max:.2} dBm at {:.1} km", p.z_km[i_max]);
    card.check(&json!({"quantity": "max_1502_dbm"}), max);
    card.check(&json!({"quantity": "max_1502_position_km"}), p.z_km[i_max]);
    card.finish().unwrap();
}

#[test]
fn kidorf_1999_fig_4_gain_and_noise_of_the_optimised_design() {
    let p = optimised();
    let mut card = Card::load("kidorf1999_fig4").unwrap();
    let (ripple, mean) = ripple_and_mean(p);
    println!("net gain: ripple {ripple:.2} dB, mean {mean:.2} dB");
    card.check(&json!({"quantity": "ripple_db"}), ripple);
    card.check(&json!({"quantity": "mean_gain_db"}), mean);
    for (k, nm) in [(0, 1520), (99, 1619)] {
        let osnr = SIGNAL_DBM + p.net_gain_db(k) - lin2db(p.ase_psd[k] * OSNR_BANDWIDTH_HZ * 1e3);
        println!("{nm} nm: net gain {:.2} dB, OSNR {osnr:.2} dB in 100 GHz", p.net_gain_db(k));
        card.check(&json!({"quantity": "osnr_db", "signal_nm": nm}), osnr);
    }
    card.finish().unwrap();
}

#[test]
fn kidorf_1999_fig_2_evenly_spaced_pumps() {
    let p = solve(&(0..8).map(|j| (1432.0 + 12.0 * f64::from(j), 0.120)).collect::<Vec<_>>());
    let mut card = Card::load("kidorf1999_fig2").unwrap();
    let (ripple, mean) = ripple_and_mean(&p);
    println!(
        "net gain: ripple {ripple:.2} dB, mean {mean:.2} dB, {:.2} dB at 1520 nm, {:.2} dB at 1619 nm",
        p.net_gain_db(0),
        p.net_gain_db(99)
    );
    card.check(&json!({"quantity": "ripple_db"}), ripple);
    card.check(&json!({"quantity": "mean_gain_db"}), mean);
    card.finish().unwrap();
}
