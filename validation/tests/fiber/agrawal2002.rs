//! The dispersion of the linear fibre (`linear_fiber_channel`, the response
//! $e^{j(\beta_{2}\omega^{2}/2 + \beta_{3}\omega^{3}/6)L}$ of `comnum_math::nlse`) against
//! G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002, eq. (2.4.22):
//! the r.m.s. width of a chirp-free Gaussian pulse of half-width $T_{0}$ grows by
//!
//! $$
//! \frac{\sigma}{\sigma_{0}} = \sqrt{\left(1 + \frac{\beta_{2}L}{T_{0}^{2}}\right)^{2} +
//! \left(\frac{\beta_{3}L}{2T_{0}^{3}}\right)^{2}}, \qquad \sigma_{0} = \frac{T_{0}}{\sqrt{2}},
//! $$
//!
//! (`direct_detection::gaussian_broadening_factor` with $C = 0$, where the first term
//! is $1 + 0$ for $C = 0$ and the $\beta_{2}$ term adds in quadrature). The width is
//! measured on the propagated power of a 5 ps pulse sampled at 4 THz, for the
//! third-order dispersion alone ($D$ = 0, $S$ = 0.1 ps/(nm² km): $\beta_{3}$ = 0.164
//! ps³/km, 40 and 200 km), for standard fibre with its slope ($D$ = 17, $S$ = 0.057:
//! $\beta_{3}$ = 0.129 ps³/km, 40 km) and for normal dispersion without slope. The sign
//! of $\beta_{3}$ is checked on the asymmetry of the pulse: the skewness of the power is
//! positive for $\beta_{3} > 0$ and $\beta_{2} = 0$ (the oscillatory tail on the trailing
//! edge, Agrawal's Fig. 3.6, in the convention of `comnum_math::nlse`).
//! The values and tolerances are in the card `validation/cards/agrawal2002_eq2_4_22.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::optical::fiber::{FiberParams, linear_fiber_channel};
use comnum::hardware::optical::direct_detection::gaussian_broadening_factor;
use comnum_validation::card::Card;
use serde_json::json;

const FS: f64 = 4e12;
const N: usize = 1 << 14;
const T0: f64 = 5e-12;

fn time(k: usize) -> f64 {
    (k as f64 - N as f64 / 2.0) / FS
}

/// The r.m.s. width and the skewness of a power waveform.
fn moments(p: &[f64]) -> (f64, f64) {
    let total: f64 = p.iter().sum();
    let mean = p.iter().enumerate().map(|(k, v)| v * time(k)).sum::<f64>() / total;
    let m2 = p.iter().enumerate().map(|(k, v)| v * (time(k) - mean).powi(2)).sum::<f64>() / total;
    let m3 = p.iter().enumerate().map(|(k, v)| v * (time(k) - mean).powi(3)).sum::<f64>() / total;
    (m2.sqrt(), m3 / m2.powf(1.5))
}

#[test]
fn agrawal_2002_eq_2_4_22_the_third_order_dispersion_of_the_linear_fibre() {
    let mut card = Card::load("agrawal2002_eq2_4_22").unwrap();
    let pulse: Vec<C64> = (0..N).map(|k| C64::new((-time(k).powi(2) / (2.0 * T0 * T0)).exp(), 0.0)).collect();
    let (sigma0, _) = moments(&pulse.iter().map(C64::norm_sqr).collect::<Vec<_>>());
    for (d, slope, length_km) in [(0.0, 0.1, 40.0), (0.0, 0.1, 200.0), (17.0, 0.057, 40.0), (-17.0, 0.0, 40.0)] {
        let fiber = FiberParams { d_ps_nm_km: d, slope_ps_nm2_km: slope, alpha_db_km: 0.0, ..FiberParams::default() };
        let out = linear_fiber_channel(&pulse, &fiber, length_km, FS);
        let (sigma, skew) = moments(&out.iter().map(C64::norm_sqr).collect::<Vec<_>>());
        let closed = gaussian_broadening_factor(T0, 0.0, fiber.beta2_s2_km(), fiber.beta3_s3_km(), length_km).unwrap();
        println!(
            "D {d} S {slope} L {length_km} km: beta3 {:.3} ps3/km, sigma/sigma0 {:.5} (eq. (2.4.22): {closed:.5}), skewness {skew:+.3}",
            fiber.beta3_s3_km() * 1e36,
            sigma / sigma0
        );
        card.check(
            &json!({"d_ps_nm_km": d, "slope_ps_nm2_km": slope, "length_km": length_km}),
            sigma / sigma0 / closed,
        );
        if d == 0.0 {
            card.check(&json!({"skewness": "sign", "length_km": length_km}), skew.signum());
        }
    }
    card.finish().unwrap();
}
