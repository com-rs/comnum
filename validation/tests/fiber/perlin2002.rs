//! The pump-power optimiser (`Raman::design_pumps`) against V. E. Perlin and H. G.
//! Winful, "Optimal design of flat-gain wide-band fiber Raman amplifiers," *J. Lightw.
//! Technol.* 20(2), 250–254, 2002, Fig. 2: the designs for 4 and 8 pumps of the 10 THz
//! example of Sec. III, at the pump wavelengths read on the figure (the paper optimises
//! them too; the library takes them as given).
//!
//! The link: 50 km of standard fibre ($A_{\mathrm{eff}}$ = 50 µm²), 100 channels at
//! 100 GHz from 186.7 to 196.6 THz (1525 to 1605 nm) at −3 dBm, backward pumps, net
//! gain 0 dB (the target on-off gain of every channel is its loss over the span, the
//! loss profile of the inset of Fig. 1 read at 10 nm), the Raman tilt of the signals
//! neglected as in the paper (`isrs: false`), the gain spectrum of Fig. 1 (read at
//! 0.5 THz, `hu2004::silica_gain_1um`, halved for depolarised pumps: the paper's
//! $K_{\mathrm{eff}}$ is not given). The card `perlin2002_fig2` holds the net-gain ripple
//! printed in panels (g) and (h) (twice the ± value), the launch powers of panels (d)
//! and (e) and the total pump power of the text.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use super::hu2004::{loss_db_km, silica_gain_1um};
use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::raman::{DesignObjective, Direction, PumpDesign, Raman, RamanPump};
use comnum::math::w2dbm;
use comnum_validation::card::Card;
use serde_json::json;

/// The loss of the inset of Fig. 1 (dB/km) at 1400, 1410, …, 1700 nm (read on the
/// raster, ±0.02).
const PERLIN_LOSS: [f64; 31] = [
    1.05, 0.94, 0.83, 0.74, 0.65, 0.57, 0.50, 0.44, 0.39, 0.35, 0.31, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21, 0.21, 0.21,
    0.22, 0.23, 0.24, 0.26, 0.29, 0.32, 0.36, 0.41, 0.47, 0.54, 0.62, 0.70,
];

const SPAN_KM: f64 = 50.0;
const AEFF_UM2: f64 = 50.0;
/// The polarisation factor $K_{\mathrm{eff}}$ of eq. (1), not given by the paper: 1
/// reproduces the total pump power of the text (with 2, the depolarised value of Hu et
/// al. 2004, every power is 2.6 dB above the figure and the total 940 mW against 530).
const POLARISATION_FACTOR: f64 = 1.0;

/// The pump wavelengths read on Fig. 2(a) (n = 4) and 2(b) (n = 8), ±1 nm.
fn wavelengths(n: usize) -> Vec<f64> {
    match n {
        4 => vec![1419.0, 1437.0, 1454.0, 1500.0],
        _ => vec![1417.0, 1429.0, 1439.0, 1446.0, 1458.0, 1486.0, 1498.0, 1514.0],
    }
}

/// The design of `n` pumps at the read wavelengths.
fn design(n: usize) -> PumpDesign {
    let pumps = wavelengths(n)
        .iter()
        .map(|w| RamanPump {
            alpha_db_km: Some(loss_db_km(&PERLIN_LOSS, *w)),
            ..RamanPump::at_wavelength(*w, 0.0, Direction::Counter)
        })
        .collect();
    let raman = Raman {
        pumps,
        gain: silica_gain_1um(POLARISATION_FACTOR),
        aeff_um2: AEFF_UM2,
        isrs: false,
        step_km: 0.5,
        ..Raman::default()
    };
    let signals: Vec<(f64, f64)> = (0..100).map(|k| (186.7e12 + k as f64 * 100e9, 5e-4)).collect();
    let targets: Vec<f64> = signals
        .iter()
        .map(|(f, _)| loss_db_km(&PERLIN_LOSS, 1e9 * comnum::math::SPEED_OF_LIGHT / f) * SPAN_KM)
        .collect();
    let fiber = FiberParams {
        span_km: SPAN_KM,
        alpha_db_km: targets.iter().sum::<f64>() / 100.0 / SPAN_KM,
        ..FiberParams::default()
    };
    raman.design_pumps(&fiber, &signals, &targets, DesignObjective::Minimax).unwrap()
}

#[test]
fn perlin_2002_fig_2_designs_at_the_read_wavelengths() {
    let mut card = Card::load("perlin2002_fig2").unwrap();
    for n in [4usize, 8] {
        let d = design(n);
        let total: f64 = d.pumps.iter().map(|p| p.power_w).sum();
        println!(
            "{n} pumps: ripple {:.3} dB, powers {:?} dBm, total {:.0} mW, {} iterations",
            d.ripple_db,
            d.pumps.iter().map(|p| (w2dbm(p.power_w) * 10.0).round() / 10.0).collect::<Vec<_>>(),
            total * 1e3,
            d.iterations
        );
        card.check(&json!({"pumps": n, "quantity": "ripple_db"}), d.ripple_db);
        card.check(&json!({"pumps": n, "quantity": "total_mw"}), total * 1e3);
        for (j, p) in d.pumps.iter().enumerate() {
            card.check(&json!({"pumps": n, "quantity": "power_dbm", "pump": j}), w2dbm(p.power_w));
        }
    }
    card.finish().unwrap();
}
