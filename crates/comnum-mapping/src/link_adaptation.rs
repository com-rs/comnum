//! Variable-rate variable-power MQAM over Rayleigh fading (Goldsmith & Chua 1997): the
//! Shannon capacity with power adapted in time, the spectral efficiency of MQAM whose
//! size and power follow the channel gain at a target bit error rate, with a continuum or
//! a finite set of constellations, and the average time spent in every fading region.
//!
//! The bit error rate of MQAM at the SNR $\gamma$ is bounded by eq. (17),
//! $\mathrm{BER} \le 0.2\, e^{-1.5\gamma/(M-1)}$ ([`mqam_ber_bound`]), which fixes the
//! constellation of a target BER at $M(\gamma) = 1 + K\gamma S(\gamma)/\bar S$, with the
//! power gap $K = -1.5/\ln(5\,\mathrm{BER})$ ([`mqam_power_gap`], eqs. (20), (24)).
//! Over Rayleigh fading of mean SNR $\bar\gamma$, $p(\gamma) = e^{-\gamma/\bar\gamma}/\bar\gamma$,
//! every integral is an exponential integral $E_{1}$:
//!
//! - capacity with water-filling in time, eqs. (5)–(7): the cutoff $\gamma_{0} = x\bar\gamma$
//!   solves $e^{-x}/x - E_{1}(x) = \bar\gamma$ and $C/B = E_{1}(x)/\ln 2$
//!   ([`rayleigh_water_filling_cutoff`], [`rayleigh_capacity_water_filling`]);
//! - continuous-rate adaptive MQAM, eqs. (23)–(25): the same water-filling at the mean
//!   SNR $K\bar\gamma$ ([`rayleigh_adaptive_mqam_efficiency`]);
//! - discrete rates, eqs. (28)–(32): the constellations $M_{1} < \dots < M_{N}$ (none in
//!   region 0) are used for $M_{j} \le \gamma/\gamma_{K}^{*} < M_{j+1}$ with the power
//!   $(M_{j} - 1)/(\gamma K)$; the power constraint (32) fixes $\gamma_{K}^{*}$,
//!   $$\sum_{j=1}^{N} \frac{M_{j} - 1}{K\bar\gamma}\left[E_{1}\!\left(\frac{\gamma_{K}^{*}M_{j}}{\bar\gamma}\right) - E_{1}\!\left(\frac{\gamma_{K}^{*}M_{j+1}}{\bar\gamma}\right)\right] = 1,$$
//!   and $R/B = \sum_{j} \log_{2}M_{j}\, p(M_{j} \le \gamma/\gamma_{K}^{*} < M_{j+1})$
//!   ([`rayleigh_discrete_mqam`]);
//! - the average fade region duration, eqs. (34)–(35): $\bar\tau_{j} = \pi_{j}/(N_{j+1} + N_{j})$
//!   with $\pi_{j}$ the probability of region $j$ and $N_{j}$ the level-crossing rate of
//!   its lower edge ([`rayleigh_region_durations`]);
//! - a constant error $\epsilon = \hat\gamma/\gamma$ of the SNR estimate raises the BER of
//!   the target $\mathrm{BER}_{0}$ to $0.2\,(5\,\mathrm{BER}_{0})^{1/\epsilon}$, eq. (42)
//!   ([`estimation_error_ber`]).
//!
//! # Examples
//! ```
//! use comnum_mapping::link_adaptation::{rayleigh_adaptive_mqam_efficiency, rayleigh_capacity_water_filling};
//! // 20 dB of mean SNR: 5.90 bit/s/Hz of capacity, 4.22 for adaptive MQAM at 1e-3 (Fig. 3)
//! let c = rayleigh_capacity_water_filling(100.0)?;
//! let r = rayleigh_adaptive_mqam_efficiency(100.0, 1e-3)?;
//! assert!((c - 5.90).abs() < 0.01 && (r - 4.22).abs() < 0.01, "{c} {r}");
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. A. J. Goldsmith, S.-G. Chua, "Variable-rate variable-power MQAM for fading
//!    channels," *IEEE Trans. Commun.* 45(10), 1218–1230, 1997: eqs. (5)–(7), (17),
//!    (20)–(25), (28)–(35), (42), Figs. 3, 5, 10, Table II.

use comnum_core::{Error, Result};
use comnum_math::{bisect, expint_e1};
use std::f64::consts::{LN_2, PI};

/// The bound (17) of the bit error rate of MQAM of size `m` at the SNR `snr` (linear):
/// $0.2\, e^{-1.5\,\gamma/(M-1)}$, within 1 dB for $M \ge 4$ and SNRs of 0 to 30 dB.
pub fn mqam_ber_bound(snr: f64, m: f64) -> f64 {
    0.2 * (-1.5 * snr / (m - 1.0)).exp()
}

/// The power gap $K = -1.5/\ln(5\,\mathrm{BER})$ of MQAM at the target bit error rate
/// `ber` (eq. (24)): $-5.48$ dB at $10^{-3}$.
///
/// # Errors
/// [`Error::Config`] unless $0 <$ `ber` $< 0.2$.
pub fn mqam_power_gap(ber: f64) -> Result<f64> {
    if !(ber > 0.0 && ber < 0.2) {
        return Err(Error::Config(format!("the target bit error rate must be in (0, 0.2), got {ber}")));
    }
    Ok(-1.5 / (5.0 * ber).ln())
}

fn check_snr(mean_snr: f64) -> Result<()> {
    if mean_snr > 0.0 && mean_snr.is_finite() {
        Ok(())
    } else {
        Err(Error::Config(format!("the mean SNR must be positive and finite, got {mean_snr}")))
    }
}

/// The cutoff $\gamma_{0}$ (linear) of water-filling in time over Rayleigh fading of mean
/// SNR `mean_snr` (linear), eq. (6): $x = \gamma_{0}/\bar\gamma$ solves
/// $e^{-x}/x - E_{1}(x) = \bar\gamma$ (bisection on $\ln x$ to $10^{-15}$).
///
/// # Errors
/// [`Error::Config`] for a mean SNR that is not positive and finite.
pub fn rayleigh_water_filling_cutoff(mean_snr: f64) -> Result<f64> {
    check_snr(mean_snr)?;
    // f(x) = e^{-x}/x - E1(x) falls from infinity to 0: the predicate holds below the root
    let ln_x = bisect(|l| (-l.exp()).exp() / l.exp() - expint_e1(l.exp()) > mean_snr, -50.0, 5.0, 200);
    Ok(ln_x.exp() * mean_snr)
}

/// The capacity (bit/s/Hz) of Rayleigh fading of mean SNR `mean_snr` (linear) with the
/// power adapted by water-filling in time, eq. (7): $E_{1}(\gamma_{0}/\bar\gamma)/\ln 2$.
///
/// # Errors
/// As [`rayleigh_water_filling_cutoff`].
pub fn rayleigh_capacity_water_filling(mean_snr: f64) -> Result<f64> {
    let g0 = rayleigh_water_filling_cutoff(mean_snr)?;
    Ok(expint_e1(g0 / mean_snr) / LN_2)
}

/// The spectral efficiency (bit/s/Hz) of MQAM with rate and power adapted continuously
/// to Rayleigh fading of mean SNR `mean_snr` (linear) at the target bit error rate `ber`,
/// eq. (25): the water-filling capacity at the mean SNR $K\bar\gamma$.
///
/// # Errors
/// As [`mqam_power_gap`] and [`rayleigh_water_filling_cutoff`].
pub fn rayleigh_adaptive_mqam_efficiency(mean_snr: f64, ber: f64) -> Result<f64> {
    rayleigh_capacity_water_filling(mqam_power_gap(ber)? * mean_snr)
}

/// MQAM with a finite set of constellations adapted to Rayleigh fading (eqs. (28)–(32)).
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteRate {
    /// The threshold $\gamma_{K}^{*}$ (linear): constellation $M_{j}$ is used for
    /// $M_{j} \le \gamma/\gamma_{K}^{*} < M_{j+1}$.
    pub gamma_k: f64,
    /// The spectral efficiency (bit/s/Hz), eq. (31).
    pub efficiency: f64,
    /// The lower edges $A_{j} = \gamma_{K}^{*}M_{j}$ (linear SNR) of the regions $1..N$.
    pub edges: Vec<f64>,
}

/// The discrete-rate adaptive MQAM of the constellation sizes `sizes` ($M_{1} < \dots <
/// M_{N}$, no transmission below $M_{1}$) over Rayleigh fading of mean SNR `mean_snr`
/// (linear) at the target BER `ber`: $\gamma_{K}^{*}$ from the power constraint (32), by
/// bisection on its logarithm to $10^{-15}$, and the efficiency (31).
///
/// # Errors
/// [`Error::Config`] for an empty or unsorted set, a size below 2, or as
/// [`mqam_power_gap`] and [`rayleigh_water_filling_cutoff`].
pub fn rayleigh_discrete_mqam(mean_snr: f64, ber: f64, sizes: &[usize]) -> Result<DiscreteRate> {
    check_snr(mean_snr)?;
    let k = mqam_power_gap(ber)?;
    if sizes.is_empty() || sizes[0] < 2 || sizes.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Config(format!("the constellation sizes must be increasing and at least 2, got {sizes:?}")));
    }
    let m: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
    let e1 = |a: f64| if a.is_infinite() { 0.0 } else { expint_e1(a / mean_snr) };
    // the average power of eq. (32) for the threshold g: it decreases as g grows
    let power = |g: f64| -> f64 {
        (0..m.len())
            .map(|j| {
                let next = m.get(j + 1).map_or(f64::INFINITY, |&n| g * n);
                (m[j] - 1.0) / (k * mean_snr) * (e1(g * m[j]) - e1(next))
            })
            .sum()
    };
    let ln_g = bisect(|l| power(l.exp()) > 1.0, -60.0, 20.0, 200);
    let g = ln_g.exp();
    let edges: Vec<f64> = m.iter().map(|&mj| g * mj).collect();
    let surv = |a: f64| (-a / mean_snr).exp();
    let efficiency = (0..m.len())
        .map(|j| {
            let next = edges.get(j + 1).map_or(0.0, |&a| surv(a));
            m[j].log2() * (surv(edges[j]) - next)
        })
        .sum();
    Ok(DiscreteRate { gamma_k: g, efficiency, edges })
}

/// The average time (s) spent in every region $0..N$ of a [`DiscreteRate`] policy over
/// Rayleigh fading of mean SNR `mean_snr` (linear) and maximum Doppler shift `doppler_hz`
/// (eqs. (34)–(35)): region 0 below $A_{1}$ (no transmission), region $j$ on
/// $[A_{j}, A_{j+1})$; $\bar\tau_{j} = \pi_{j}/(N_{j} + N_{j+1})$ with
/// $N_{j} = \sqrt{2\pi A_{j}/\bar\gamma}\, f_{D}\, e^{-A_{j}/\bar\gamma}$ ($N_{0} = 0$,
/// $N_{N+1} = 0$).
///
/// # Errors
/// [`Error::Config`] for a mean SNR or a Doppler shift that is not positive and finite.
pub fn rayleigh_region_durations(policy: &DiscreteRate, mean_snr: f64, doppler_hz: f64) -> Result<Vec<f64>> {
    check_snr(mean_snr)?;
    if !(doppler_hz > 0.0 && doppler_hz.is_finite()) {
        return Err(Error::Config(format!("the Doppler shift must be positive and finite, got {doppler_hz}")));
    }
    let a: Vec<f64> = std::iter::once(0.0).chain(policy.edges.iter().copied()).collect();
    let rate = |a: f64| (2.0 * PI * a / mean_snr).sqrt() * doppler_hz * (-a / mean_snr).exp();
    let surv = |a: f64| (-a / mean_snr).exp();
    Ok((0..a.len())
        .map(|j| {
            let (lo, hi) = (a[j], a.get(j + 1).copied());
            let prob = surv(lo) - hi.map_or(0.0, surv);
            let crossings = rate(lo) + hi.map_or(0.0, rate);
            prob / crossings
        })
        .collect())
}

/// The bit error rate of adaptive MQAM designed for `ber0` when the SNR is estimated as
/// $\epsilon$ = `epsilon` times its value (perfect AGC), eq. (42):
/// $0.2\,(5\,\mathrm{BER}_{0})^{1/\epsilon}$.
pub fn estimation_error_ber(ber0: f64, epsilon: f64) -> f64 {
    0.2 * (5.0 * ber0).powf(1.0 / epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cutoff solves eq. (6) and lies below 1 (Sec. III); the adaptive MQAM loses the
    /// power gap K against the capacity; the bound (17) gives 0.2 at zero SNR.
    #[test]
    fn the_cutoff_solves_eq_6_and_mqam_pays_the_gap() {
        for snr_db in [0.0, 10.0, 20.0, 30.0] {
            let s: f64 = 10f64.powf(snr_db / 10.0);
            let g0 = rayleigh_water_filling_cutoff(s).unwrap();
            let x = g0 / s;
            assert!(((-x).exp() / g0 - expint_e1(x) / s - 1.0).abs() < 1e-9, "{snr_db} dB");
            assert!(g0 < 1.0);
            let c = rayleigh_capacity_water_filling(s).unwrap();
            let r = rayleigh_adaptive_mqam_efficiency(s, 1e-3).unwrap();
            assert!(r < c);
        }
        assert!((mqam_ber_bound(0.0, 4.0) - 0.2).abs() < 1e-15);
        assert!((10.0 * mqam_power_gap(1e-3).unwrap().log10() + 5.48).abs() < 0.01);
        assert!(mqam_power_gap(0.3).is_err() && rayleigh_capacity_water_filling(-1.0).is_err());
    }

    /// With many fine constellations the discrete rate approaches the continuous one, and
    /// a single constellation of size 2 never exceeds 1 bit/s/Hz.
    #[test]
    fn discrete_rates_approach_the_continuous_one_and_saturate() {
        let s = 100.0;
        let sizes: Vec<usize> = (1..=14).map(|b| 1usize << b).collect();
        let d = rayleigh_discrete_mqam(s, 1e-3, &sizes).unwrap();
        let c = rayleigh_adaptive_mqam_efficiency(s, 1e-3).unwrap();
        assert!(d.efficiency < c && c - d.efficiency < 0.6, "{} {c}", d.efficiency);
        let one = rayleigh_discrete_mqam(s, 1e-3, &[2]).unwrap();
        assert!(one.efficiency < 1.0 && one.efficiency > 0.9);
        assert!(rayleigh_discrete_mqam(s, 1e-3, &[4, 2]).is_err());
        let t = rayleigh_region_durations(&d, s, 100.0).unwrap();
        assert_eq!(t.len(), sizes.len() + 1);
        assert!(t.iter().all(|&x| x > 0.0 && x.is_finite()));
    }
}
