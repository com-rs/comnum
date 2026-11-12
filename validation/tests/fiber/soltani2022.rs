//! The bidirectional Raman solver (`Raman::solve`) against the power maps of M. Soltani,
//! F. Da Ros, A. Carena and D. Zibar, "Spectral and spatial power evolution design with
//! machine learning-enabled Raman amplification," *J. Lightw. Technol.* 40(12),
//! 3546–3556, 2022, Table III and Fig. 5.
//!
//! The link of Sec. III: one span of 80 km of single-mode fibre, 40 channels at 192.05 +
//! 0.1 k THz launched at 0 dBm, four co-pumps and four counter-pumps at 1366, 1425, 1455
//! and 1475 nm (the 1366 nm pumps of second order) with the powers of the four designs of
//! Table III; the attenuation of the signals 0.2 dB/km, of the first-order pumps 0.25
//! dB/km, of the second-order ones 0.32 dB/km; a constant effective area of 80 µm²; the
//! gain spectrum of standard silica with the peak of the paper, 0.4125 1/(W·km). The
//! paper computes its maps with GNPy (its boundary-value solver with residual control, at
//! 500 m), whose versions up to 2.4 take the efficiency of a pump from its offset alone:
//! the gain is not scaled with the frequency of the pump here
//! (`RamanGain::with_pump_scaling(false)`). The gain table, the version of GNPy and the
//! losses of the pump couplers are not given; the card `soltani2022_fig5` holds the
//! values read from the vector paths of Fig. 5 and the agreement obtained.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::raman::{Direction, Raman, RamanGain, RamanPump};
use comnum_validation::card::Card;
use serde_json::json;

const STEP_KM: f64 = 0.05;

/// The pump powers of Table III (mW), p1 to p8: co-pumps at 1366, 1425, 1455, 1475 nm,
/// then counter-pumps at the same wavelengths.
const DESIGNS: [(&str, [f64; 8]); 4] = [
    ("a", [330.0, 33.0, 145.0, 12.0, 1030.0, 12.0, 19.0, 43.0]),
    ("b", [430.0, 45.0, 98.0, 12.0, 1150.0, 8.0, 12.0, 24.0]),
    ("c", [440.0, 49.0, 90.0, 13.0, 1060.0, 6.0, 18.0, 63.0]),
    ("d", [450.0, 47.0, 76.0, 14.0, 990.0, 6.0, 21.0, 63.0]),
];
const WAVELENGTHS_NM: [f64; 4] = [1366.0, 1425.0, 1455.0, 1475.0];

#[test]
fn soltani_2022_fig_5_power_maps_of_table_iii() {
    let mut card = Card::load("soltani2022_fig5").unwrap();
    let fiber = FiberParams { span_km: 80.0, alpha_db_km: 0.2, ..FiberParams::default() };
    let signals: Vec<(f64, f64)> = (0..40).map(|k| (192.05e12 + 0.1e12 * f64::from(k), 1e-3)).collect();
    // standard silica with the peak of the paper, 0.4125 1/(W km) at 80 um2
    let ssmf = RamanGain::ssmf();
    let peak = ssmf.peak().1 / 80e-12 * 1e3;
    let gain = ssmf.scaled(0.4125 / peak).unwrap().with_pump_scaling(false);
    for (design, powers) in DESIGNS {
        let pumps = powers
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let nm = WAVELENGTHS_NM[i % 4];
                let direction = if i < 4 { Direction::Co } else { Direction::Counter };
                RamanPump {
                    alpha_db_km: Some(if nm < 1400.0 { 0.32 } else { 0.25 }),
                    ..RamanPump::at_wavelength(nm, p * 1e-3, direction)
                }
            })
            .collect();
        let raman =
            Raman { pumps, gain: gain.clone(), aeff_um2: 80.0, isrs: true, step_km: STEP_KM, ..Raman::default() };
        let profile = raman.solve(&fiber, &signals).unwrap();
        for (k, _) in signals.iter().enumerate() {
            let p = profile.power_dbm(k);
            // the frequency as the card writes it (exact decimal keys)
            let f_thz = (19205 + 10 * k) as f64 / 100.0;
            // every channel at the end of the span, five of them every 10 km before it
            let along: &[u32] =
                if [0, 10, 20, 30, 39].contains(&k) { &[10, 20, 30, 40, 50, 60, 70, 80] } else { &[80] };
            for &z_km in along {
                let i = profile.z_km.iter().position(|z| (z - f64::from(z_km)).abs() < 1e-9).unwrap();
                card.check(&json!({"design": design, "f_thz": f_thz, "z_km": z_km}), p[i]);
            }
        }
    }
    card.finish().unwrap();
}
