//! The sum-of-sinusoids Rayleigh fading of `comnum::channel::multipath::Tdl` against Y. R.
//! Zheng and C. Xiao, "Simulation models with correct statistical properties for Rayleigh
//! fading channels," *IEEE Trans. Commun.* 51(6), 920–928, 2003.
//!
//! `Tdl` draws every tap with the angles of eq. (15), $\alpha_{n} = (2\pi n - \pi + \theta)/(4M)$,
//! a new $\theta$ per tap and link (eq. (21)) and independent phases on the in-phase and
//! quadrature sums, $M = 8$ sinusoids (the paper's figures). One tap of unit power and
//! maximum Doppler shift $f_{D}$:
//!
//! - Theorem 1, eqs. (16a)-(16e): the ensemble correlations $E[X_{c}(t)X_{c}(t+\tau)]/(p/2)$
//!   and $E[X(t)X^{*}(t+\tau)]/p$ equal $J_{0}(2\pi f_{D}\tau)$ for every $M$, the
//!   cross-correlation $E[X_{c}X_{s}]$ is zero: 20 000 realisations at seven lags;
//! - Theorem 3, eqs. (19a)-(19b): the level-crossing rate and the average fade duration of
//!   the envelope (`rayleigh_level_crossing_rate`, `rayleigh_average_fade_duration`),
//!   measured on 400 realisations of 200 Doppler periods sampled at $f_{D}T_{s} = 0.002$
//!   (the paper: $0.025$, which misses the short deep fades it reports at -30 dB), at five
//!   levels from -20 to +5 dB.
//!
//! The expected values (the closed forms of the paper) and the tolerances are in
//! `validation/cards/zheng2003_statistics.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::multipath::{Tdl, TdlParams, rayleigh_average_fade_duration, rayleigh_level_crossing_rate};
use comnum::prelude::*;
use comnum_validation::card::Card;
use serde_json::json;

/// One tap of unit power, maximum Doppler shift `fd` (Hz) at the sampling rate 1 Hz, with
/// `m` sinusoids.
fn channel_m(fd: f64, m: usize) -> Tdl {
    let p = TdlParams { taps: vec![(0.0, 0.0)], doppler_hz: fd, sample_rate: 1.0, ..TdlParams::default() };
    let mut ch = Tdl::new(&p).unwrap();
    ch.sinusoids = m;
    ch
}

/// [`channel_m`] with the 8 sinusoids of the paper's figures.
fn channel(fd: f64) -> Tdl {
    channel_m(fd, 8)
}

/// Upward crossings of `level` and samples below it in `records` realisations.
fn crossings(ch: &Tdl, len: usize, records: usize, level: f64, rng: &mut Rng) -> (usize, usize) {
    let (mut ups, mut below) = (0, 0);
    for _ in 0..records {
        let taps = ch.draw(len, rng);
        let env: Vec<f64> = (0..len).map(|i| taps.coefficient(0, i, 0, 0).norm()).collect();
        ups += env.windows(2).filter(|w| w[0] < level && w[1] >= level).count();
        below += env.iter().filter(|&&e| e < level).count();
    }
    (ups, below)
}

#[test]
#[ignore = "statistical validation: run with --ignored (about 10 s)"]
fn zheng_xiao_2003_correlations_level_crossings_and_fades() {
    let mut card = Card::load("zheng2003_statistics").unwrap();
    let mut rng = Rng::new(2003);
    // Theorem 1: ensemble correlations at the lags f_D tau
    let ch = channel(1.0);
    let lags = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
    let trials = 20_000;
    let (mut rcc, mut rxx, mut rcs) = (vec![0.0; lags.len()], vec![0.0; lags.len()], vec![0.0; lags.len()]);
    for _ in 0..trials {
        // the origin at a random time: the statistics do not depend on t (WSS)
        let t0 = 100.0 * rng.uniform();
        let times: Vec<f64> = std::iter::once(t0).chain(lags.iter().map(|l| t0 + l)).collect();
        let taps = ch.draw_at(&times, &mut rng);
        let x0 = taps.coefficient(0, 0, 0, 0);
        for (k, _) in lags.iter().enumerate() {
            let x = taps.coefficient(0, k + 1, 0, 0);
            rcc[k] += x0.re * x.re;
            rcs[k] += x0.re * x.im;
            rxx[k] += (x0 * x.conj()).re;
        }
    }
    let n = trials as f64;
    for (k, &lag) in lags.iter().enumerate() {
        card.check(&json!({"quantity": "rcc", "fd_tau": lag}), rcc[k] / n / 0.5);
        card.check(&json!({"quantity": "rxx", "fd_tau": lag}), rxx[k] / n);
        card.check(&json!({"quantity": "rcs", "fd_tau": lag}), rcs[k] / n / 0.5);
    }
    // Theorem 3: level crossings and fades of the envelope, in time over the records
    let fd = 0.002;
    let ch = channel(fd);
    let len = (200.0 / fd) as usize;
    let levels_db = [-20.0, -10.0, -3.0, 0.0, 5.0];
    let (mut ups, mut below) = (vec![0usize; levels_db.len()], vec![0usize; levels_db.len()]);
    let records = 400;
    for _ in 0..records {
        let taps = ch.draw(len, &mut rng);
        let env: Vec<f64> = (0..len).map(|i| taps.coefficient(0, i, 0, 0).norm()).collect();
        for (j, db) in levels_db.iter().enumerate() {
            let level = 10f64.powf(db / 20.0);
            ups[j] += env.windows(2).filter(|w| w[0] < level && w[1] >= level).count();
            below[j] += env.iter().filter(|&&e| e < level).count();
        }
    }
    let duration = (records * len) as f64; // samples of 1 s
    for (j, db) in levels_db.iter().enumerate() {
        let rho = 10f64.powf(db / 20.0);
        let lcr = ups[j] as f64 / duration / fd;
        let afd = below[j] as f64 / ups[j] as f64 * fd;
        println!(
            "{db} dB: LCR/fd {lcr:.4} ({:.4}), fd AFD {afd:.4} ({:.4})",
            rayleigh_level_crossing_rate(rho, 1.0),
            rayleigh_average_fade_duration(rho, 1.0)
        );
        card.check(&json!({"quantity": "lcr", "rho_db": db}), lcr);
        card.check(&json!({"quantity": "afd", "rho_db": db}), afd);
        // the closed forms of comnum against the card's values of eq. (19)
        card.check(&json!({"quantity": "lcr_closed_form", "rho_db": db}), rayleigh_level_crossing_rate(rho, 1.0));
        card.check(&json!({"quantity": "afd_closed_form", "rho_db": db}), rayleigh_average_fade_duration(rho, 1.0));
    }
    // the same at +5 dB with 64 sinusoids: the shortfall of 8 is that of finite M
    let (u, _) = crossings(&channel_m(fd, 64), len, records, 10f64.powf(0.25), &mut rng);
    let lcr64 = u as f64 / duration / fd;
    println!("5 dB, 64 sinusoids: LCR/fd {lcr64:.4}");
    card.check(&json!({"quantity": "lcr", "rho_db": 5.0, "sinusoids": 64}), lcr64);
    card.finish().unwrap();
}
