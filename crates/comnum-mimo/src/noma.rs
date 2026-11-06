//! Downlink non-orthogonal multiple access (NOMA) with users dropped at random in a disc
//! (Ding, Yang, Fan & Poor 2014): superposition coding at the base station, successive
//! interference cancellation (SIC) at the users, outage probabilities in closed form.
//!
//! $M$ users lie uniformly in a disc of radius $\mathcal{R}_{D}$ around the base station;
//! the channel of a user at the distance $d$ is $h = g/\sqrt{1 + d^{\alpha}}$ with
//! $g \sim \mathcal{CN}(0, 1)$. The unordered gain $|h|^{2}$ has the CDF (eq. (3))
//! $$F(y) = \frac{2}{\mathcal{R}_{D}^{2}}\int_{0}^{\mathcal{R}_{D}}\left(1 - e^{-(1 + z^{\alpha})y}\right) z\, dz$$
//! ([`RandomDisc::gain_cdf`], by Gauss–Legendre with 256 nodes), approximated in the
//! paper by Gauss–Chebyshev with $N$ nodes (eq. (4), [`RandomDisc::gain_cdf_chebyshev`]).
//! With the gains ordered $|h_{1}|^{2} \le \dots \le |h_{M}|^{2}$, the power coefficients
//! $a_{1} \ge \dots \ge a_{M}$ and the target rates $\tilde R_{m}$, $\phi_{j} = 2^{\tilde R_{j}} - 1$,
//! user $m$ decodes the messages $1..m$ when $|h_{m}|^{2} \ge \psi_{m}^{*} = \max_{j \le m}\psi_{j}$,
//! $\psi_{j} = \phi_{j}/(\rho(a_{j} - \phi_{j}\sum_{i > j} a_{i}))$ and $\psi_{M} = \phi_{M}/(\rho a_{M})$
//! (eqs. (7)–(8); no outage threshold when $a_{j} \le \phi_{j}\sum_{i > j}a_{i}$: outage 1).
//! The outage of user $m$ is that of the $m$-th order statistic (eq. (9)),
//! $P_{m} = I_{F(\psi_{m}^{*})}(m, M - m + 1)$ ([`NomaDownlink::outage`]), and at high SNR
//! (eq. (11)) $P_{m} \approx \frac{\tau_{m}}{m}\eta^{m}(\psi_{m}^{*})^{m}$, $\tau_{m} = \frac{M!}{(m-1)!(M-m)!}$,
//! $\eta = \frac{1}{\mathcal{R}_{D}}\sum_{n}\beta_{n}$ ([`NomaDownlink::outage_high_snr`]):
//! user $m$ has the diversity order $m$.
//!
//! # Examples
//! ```
//! use comnum_mimo::noma::{NomaDownlink, RandomDisc};
//! // Fig. 3 of Ding et al., case 1: 80 m, alpha = 3, rates 0.1 and 0.5, rho = 80 dB
//! let disc = RandomDisc { radius: 80.0, alpha: 3.0 };
//! let noma = NomaDownlink::new(vec![0.8, 0.2], vec![0.1, 0.5])?;
//! let p = noma.outage_high_snr(&disc, 1e8, 10)?;
//! assert!((p[0] / 3.781e-4 - 1.0).abs() < 1e-3 && (p[1] / 0.1837e-4 - 1.0).abs() < 1e-3, "{p:?}");
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. Z. Ding, Z. Yang, P. Fan, H. V. Poor, "On the performance of non-orthogonal
//!    multiple access in 5G systems with randomly deployed users," *IEEE Signal Process.
//!    Lett.* 21(12), 1501–1505, 2014 (arXiv:1406.1516): eqs. (3)–(11), Figs. 1, 3.

use comnum_core::{Error, Result};
use comnum_math::{beta_inc, gauss_chebyshev, gauss_legendre};

/// Users dropped uniformly in a disc around the base station, path loss $1 + d^{\alpha}$,
/// Rayleigh fading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomDisc {
    /// Radius $\mathcal{R}_{D}$ of the disc (m).
    pub radius: f64,
    /// Path-loss exponent $\alpha$.
    pub alpha: f64,
}

impl RandomDisc {
    fn check(&self) -> Result<()> {
        if self.radius > 0.0 && self.alpha > 0.0 && self.radius.is_finite() && self.alpha.is_finite() {
            Ok(())
        } else {
            Err(Error::Config(format!("a disc needs a positive radius and exponent, got {self:?}")))
        }
    }

    /// The CDF of the gain $|h|^{2}$ of a user dropped at random, eq. (3) (Gauss–Legendre,
    /// 256 nodes on $[0, \mathcal{R}_{D}]$).
    ///
    /// # Errors
    /// [`Error::Config`] for a radius or an exponent that is not positive and finite.
    pub fn gain_cdf(&self, y: f64) -> Result<f64> {
        self.check()?;
        let (t, w) = gauss_legendre(256);
        let r = self.radius;
        let s: f64 = t
            .iter()
            .zip(&w)
            .map(|(&t, &w)| {
                let z = r / 2.0 * (t + 1.0);
                w * (-(-(1.0 + z.powf(self.alpha)) * y).exp_m1()) * z
            })
            .sum();
        Ok(s * r / 2.0 * 2.0 / (r * r))
    }

    /// The Gauss–Chebyshev nodes $c_{n}$ and coefficients $\beta_{n}$ of eqs. (4)–(5).
    fn chebyshev(&self, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (theta, w) = gauss_chebyshev(n);
        let r = self.radius;
        let z: Vec<f64> = theta.iter().map(|&t| r / 2.0 * t + r / 2.0).collect();
        let c: Vec<f64> = z.iter().map(|&z| 1.0 + z.powf(self.alpha)).collect();
        // the factor of 1 - e^{-c_n y} in eq. (4) and beta_n of eq. (5)
        let a: Vec<f64> = theta.iter().zip(&w).zip(&z).map(|((&t, &w), &z)| w * (1.0 - t * t).sqrt() * z).collect();
        let beta = a.iter().zip(&c).map(|(a, c)| a * c).collect();
        (c, a, beta)
    }

    /// The Gauss–Chebyshev approximation (4) of [`RandomDisc::gain_cdf`] with `n` nodes.
    ///
    /// # Errors
    /// As [`RandomDisc::gain_cdf`], and [`Error::Config`] for `n` = 0.
    pub fn gain_cdf_chebyshev(&self, y: f64, n: usize) -> Result<f64> {
        self.check()?;
        if n == 0 {
            return Err(Error::Config("Gauss-Chebyshev needs at least one node".into()));
        }
        let (c, a, _) = self.chebyshev(n);
        Ok(c.iter().zip(&a).map(|(c, a)| a * (1.0 - (-c * y).exp())).sum::<f64>() / self.radius)
    }

    /// The slope $\eta = \frac{1}{\mathcal{R}_{D}}\sum_{n}\beta_{n}$ of the approximate CDF at
    /// zero (eq. (10)) with `n` Gauss–Chebyshev nodes.
    ///
    /// # Errors
    /// As [`RandomDisc::gain_cdf_chebyshev`].
    pub fn gain_slope(&self, n: usize) -> Result<f64> {
        self.check()?;
        if n == 0 {
            return Err(Error::Config("Gauss-Chebyshev needs at least one node".into()));
        }
        Ok(self.chebyshev(n).2.iter().sum::<f64>() / self.radius)
    }
}

/// A downlink NOMA policy: the power coefficients $a_{1} \ge \dots \ge a_{M}$ of the
/// users ordered by increasing gain and their target rates $\tilde R_{m}$ (bit per
/// channel use).
#[derive(Clone, Debug, PartialEq)]
pub struct NomaDownlink {
    /// Power coefficients $a_{m}$ (summing to 1).
    pub powers: Vec<f64>,
    /// Target rates $\tilde R_{m}$ (bit per channel use).
    pub rates: Vec<f64>,
}

impl NomaDownlink {
    /// The policy of the power coefficients `powers` and the target rates `rates`.
    ///
    /// # Errors
    /// [`Error::Config`] for no user, lengths that differ, negative values or powers that
    /// do not sum to 1.
    pub fn new(powers: Vec<f64>, rates: Vec<f64>) -> Result<NomaDownlink> {
        if powers.is_empty() || powers.len() != rates.len() {
            return Err(Error::Shape(format!("{} power coefficients for {} rates", powers.len(), rates.len())));
        }
        if powers.iter().chain(&rates).any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(Error::Config("power coefficients and rates must be non-negative".into()));
        }
        if (powers.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
            return Err(Error::Config(format!("the power coefficients must sum to 1, got {powers:?}")));
        }
        Ok(NomaDownlink { powers, rates })
    }

    /// The thresholds $\psi_{m}^{*}$ of the users at the transmit SNR `snr` (linear): user
    /// $m$ decodes its message when $|h_{m}|^{2} \ge \psi_{m}^{*}$ (infinite when condition
    /// (8) fails for one of the messages it decodes).
    pub fn thresholds(&self, snr: f64) -> Vec<f64> {
        let m = self.powers.len();
        let psi: Vec<f64> = (0..m)
            .map(|j| {
                let phi = 2f64.powf(self.rates[j]) - 1.0;
                let rest: f64 = self.powers[j + 1..].iter().sum();
                let margin = self.powers[j] - phi * rest;
                if margin > 0.0 { phi / (snr * margin) } else { f64::INFINITY }
            })
            .collect();
        psi.iter()
            .scan(0.0f64, |acc, &p| {
                *acc = acc.max(p);
                Some(*acc)
            })
            .collect()
    }

    /// The outage probability of every user at the transmit SNR `snr` (linear), eq. (9):
    /// $I_{F(\psi_{m}^{*})}(m, M - m + 1)$, $F$ the exact CDF (3).
    ///
    /// # Errors
    /// As [`RandomDisc::gain_cdf`].
    pub fn outage(&self, disc: &RandomDisc, snr: f64) -> Result<Vec<f64>> {
        let m = self.powers.len();
        self.thresholds(snr)
            .iter()
            .enumerate()
            .map(|(i, &psi)| {
                if psi.is_infinite() {
                    return Ok(1.0);
                }
                let f = disc.gain_cdf(psi)?;
                Ok(beta_inc((i + 1) as f64, (m - i) as f64, f))
            })
            .collect()
    }

    /// The high-SNR approximation (11) of the outage of every user with `n`
    /// Gauss–Chebyshev nodes: $\frac{\tau_{m}}{m}\eta^{m}(\psi_{m}^{*})^{m}$, clipped at 1.
    ///
    /// # Errors
    /// As [`RandomDisc::gain_slope`].
    pub fn outage_high_snr(&self, disc: &RandomDisc, snr: f64, n: usize) -> Result<Vec<f64>> {
        let eta = disc.gain_slope(n)?;
        let m = self.powers.len();
        Ok(self
            .thresholds(snr)
            .iter()
            .enumerate()
            .map(|(i, &psi)| {
                let k = i + 1;
                // tau_m / m = M! / (m! (M - m)!), the binomial coefficient
                let binom = (0..k).fold(1.0, |c, j| c * (m - j) as f64 / (j + 1) as f64);
                (binom * (eta * psi).powi(k as i32)).min(1.0)
            })
            .collect())
    }
}

/// The outage probability of a user scheduled at random on an orthogonal resource at the
/// transmit SNR `snr` (linear) and the rate `rate` (bit per channel use): $F((2^{R} - 1)/\rho)$
/// (the "conventional MA" of Ding et al. 2014, Figs. 1 and 3).
///
/// # Errors
/// As [`RandomDisc::gain_cdf`].
pub fn orthogonal_outage(disc: &RandomDisc, snr: f64, rate: f64) -> Result<f64> {
    disc.gain_cdf((2f64.powf(rate) - 1.0) / snr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact CDF is 0 at 0 and tends to 1; for alpha = 2 it has the closed form
    /// $1 - e^{-y}(1 - e^{-R^{2}y})/(R^{2}y)$; Gauss–Chebyshev approaches it.
    #[test]
    fn the_gain_cdf_has_its_closed_form_for_alpha_2() {
        let d = RandomDisc { radius: 3.0, alpha: 2.0 };
        for y in [1e-3f64, 0.05, 0.5, 3.0] {
            let r2 = 9.0;
            let exact = 1.0 - (-y).exp() * (1.0 - (-r2 * y).exp()) / (r2 * y);
            assert!((d.gain_cdf(y).unwrap() - exact).abs() < 1e-12, "{y}");
            assert!((d.gain_cdf_chebyshev(y, 200).unwrap() - exact).abs() < 1e-4, "{y}");
        }
        assert!(d.gain_cdf(0.0).unwrap().abs() < 1e-15);
        assert!(RandomDisc { radius: -1.0, alpha: 2.0 }.gain_cdf(1.0).is_err());
    }

    /// Condition (8) fails for $a_{1} = 0.6$ at the rate 2 ($\phi = 3 > 0.6/0.4$): every user is
    /// in outage; the outage falls with the SNR, faster for the stronger user.
    #[test]
    fn the_outage_follows_condition_8_and_the_diversity_order() {
        let d = RandomDisc { radius: 5.0, alpha: 3.0 };
        let bad = NomaDownlink::new(vec![0.6, 0.4], vec![2.0, 1.0]).unwrap();
        assert_eq!(bad.outage(&d, 1e6).unwrap(), vec![1.0, 1.0]);
        let noma = NomaDownlink::new(vec![0.8, 0.2], vec![0.1, 0.5]).unwrap();
        let (p3, p4) = (noma.outage(&d, 1e3).unwrap(), noma.outage(&d, 1e4).unwrap());
        // one decade per 10 dB for user 1, two for user 2
        assert!(((p3[0] / p4[0]).log10() - 1.0).abs() < 0.1 && ((p3[1] / p4[1]).log10() - 2.0).abs() < 0.15);
        assert!(NomaDownlink::new(vec![0.5, 0.4], vec![0.1, 0.1]).is_err());
        assert!(NomaDownlink::new(vec![1.0], vec![0.1, 0.1]).is_err());
    }
}
