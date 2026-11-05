//! Training-based estimation of a flat MIMO channel: the least-squares, scaled
//! least-squares, LMMSE and relaxed MMSE estimators, their mean square errors, the
//! optimal training matrices and the combination of several training periods
//! (Biguesh & Gershman 2006).
//!
//! # Model
//!
//! An $r \times t$ channel $H$ is probed by $N \ge t$ training vectors, the columns of
//! the $t \times N$ matrix $P$ of power $\lVert P \rVert_{F}^{2} = \mathcal{P}$:
//!
//! $$
//! S = H P + V,
//! $$
//!
//! $V$ white of variance $\sigma_{n}^{2}$ per entry (eq. (2)). The prior of $H$ is its
//! correlation $R_{H} = E\\{H^{H} H\\}$ ($t \times t$, footnote 2: not that of
//! $\mathrm{vec}(H)$), the rows of $H$ uncorrelated. Every estimator is linear,
//! $\hat{H} = S A$ with an $N \times t$ matrix $A$ ([`ChannelEstimator`]):
//!
//! | Estimator | $A$ | Needs | Eq. |
//! |---|---|---|---|
//! | LS | $P^{\dagger} = P^{H} (P P^{H})^{-1}$ | — | (4) |
//! | scaled LS | $\gamma_{o} P^{\dagger}$, $\gamma_{o} = \frac{\mathrm{tr}\\{R_{H}\\}}{J_{\mathrm{LS}} + \mathrm{tr}\\{R_{H}\\}}$ | $\mathrm{tr}\\{R_{H}\\}$, $\sigma_{n}^{2}$ | (14), (16) |
//! | LMMSE | $(P^{H} R_{H} P + \sigma_{n}^{2} r I)^{-1} P^{H} R_{H}$ | $R_{H}$, $\sigma_{n}^{2}$ | (25), (26) |
//! | relaxed MMSE | the LMMSE of $R_{H} = \frac{\mathrm{tr}\\{R_{H}\\}}{t} I$ | $\mathrm{tr}\\{R_{H}\\}$, $\sigma_{n}^{2}$ | (47), (48) |
//!
//! Scaled LS and relaxed MMSE take $\mathrm{tr}\\{R_{H}\\}$ either known or estimated
//! by $\mathrm{tr}\\{\hat{H}_{\mathrm{LS}}^{H} \hat{H}_{\mathrm{LS}}\\}$ (eq. (17), the
//! LS-SLS and LS-RMMSE estimators; [`TracePrior`]).
//!
//! # Mean square error
//!
//! With $B = I - P A$, the error $H - \hat{H} = H B - V A$ has
//!
//! $$
//! J = E\\{\lVert H - \hat{H} \rVert_{F}^{2}\\} = \mathrm{tr}\\{B^{H} R_{H} B\\} +
//!   \sigma_{n}^{2} r \\, \mathrm{tr}\\{A^{H} A\\},
//! $$
//!
//! the closed form [`Training::mse`] for every estimator whose prior is known, also
//! when the prior of the estimator is not the true $R_{H}$ (a mismatch). It gives
//! $J_{\mathrm{LS}} = \sigma_{n}^{2} r \\, \mathrm{tr}\\{(P P^{H})^{-1}\\}$ (7),
//! $J_{\mathrm{SLS}} = \frac{J_{\mathrm{LS}} \mathrm{tr}\\{R_{H}\\}}{J_{\mathrm{LS}} +
//! \mathrm{tr}\\{R_{H}\\}}$ (15) and
//! $J_{\mathrm{MMSE}} = \mathrm{tr}\\{(R_{H}^{-1} + \frac{1}{\sigma_{n}^{2} r} P P^{H})^{-1}\\}$
//! (28). LS-SLS and LS-RMMSE have no closed form (a Monte Carlo).
//!
//! # Training
//!
//! Orthogonal rows, $P P^{H} = \frac{\mathcal{P}}{t} I$ (9), minimise the MSE of LS,
//! scaled LS and relaxed MMSE; [`Training::orthogonal`] is the DFT matrix of eq. (10),
//! of entries of equal magnitude. For the LMMSE, the optimal training
//! [`Training::mmse_optimal`] water-fills the power on the eigenvectors of
//! $R_{H} = Q \Lambda Q^{H}$: $P = \sqrt{\sigma_{n}^{2} r}\\, Q \\,[\mathrm{diag}(p_{i})^{1/2}, 0]$,
//! $p_{i} = (\mu_{o} - 1/\lambda_{i})^{+}$, $\sum_{i} p_{i} = \mathcal{P}/(\sigma_{n}^{2} r)$
//! (eqs. (41)–(44), the unitary $U$ of (44) the identity).
//!
//! # Several training periods
//!
//! The LS estimates $\hat{H}_{k}$ of $K$ periods of training $P_{k}$, the channel
//! unchanged, are combined by the best linear unbiased estimator [`blue_combine`] of
//! weights [`blue_weights`] (Lemma 4, eq. (54)):
//!
//! $$
//! \alpha_{k} = \frac{1 / \kappa_{k}}{\sum_{l} 1 / \kappa_{l}}, \qquad
//! \kappa_{k} = \mathrm{tr}\\{(P_{k} P_{k}^{H})^{-1}\\},
//! $$
//!
//! $\alpha_{k} = \mathcal{P}_{k}/\mathcal{P}_{\mathrm{tot}}$ for orthogonal training
//! (58), of MSE [`blue_mse`] $\sigma_{n}^{2} r / \sum_{l} 1/\kappa_{l}$ (eqs. (59), (61)).
//!
//! # Examples
//! ```
//! use comnum_mimo::estimation::{ChannelEstimator, Training};
//! use comnum_math::linalg::{identity, matmul};
//! let (t, r, power, noise) = (4, 2, 10.0, 0.5);
//! let p = Training::orthogonal(t, t, power)?;
//! let r_h = identity(t) * comnum_math::C64::from(r as f64);
//! // LS: sigma^2 t^2 r / P, eq. (12)
//! let j = p.mse(&ChannelEstimator::Ls, &r_h.view(), noise, r)?;
//! assert!((j - noise * (t * t * r) as f64 / power).abs() < 1e-12);
//! // a noiseless observation is estimated exactly by LS
//! let h = comnum_math::linalg::CMat::from_elem((r, t), comnum_math::C64::new(0.3, -1.0));
//! let s = matmul(&h.view(), &p.symbols().view());
//! let h_ls = p.estimate(&s.view(), &ChannelEstimator::Ls, noise)?;
//! assert!((&h_ls - &h).iter().all(|e| e.norm() < 1e-12));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. M. Biguesh and A. B. Gershman, "Training-based MIMO channel estimation: a study
//!    of estimator tradeoffs and optimal training signals," *IEEE Trans. Signal
//!    Process.*, 54(3), 884–893, 2006.

use crate::precoding::water_filling_powers;
use comnum_core::{Error, Result, ensure, ensure_shape};
use comnum_math::C64;
use comnum_math::linalg::{CMat, hermitian, identity, inverse_hpd, matmul, solve_hpd, svd};
use ndarray::ArrayView2;
use std::f64::consts::PI;

/// The trace $\mathrm{tr}\\{R_{H}\\}$ used by the scaled LS and relaxed MMSE estimators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TracePrior {
    /// Known (SLS, eq. (16); RMMSE, eq. (48)).
    Known(f64),
    /// Estimated from the LS estimate, $\mathrm{tr}\\{\hat{H}_{\mathrm{LS}}^{H}
    /// \hat{H}_{\mathrm{LS}}\\}$ (eq. (17): LS-SLS; with (48): LS-RMMSE, eq. (51)).
    FromLs,
}

/// A linear estimator $\hat{H} = S A$ of the channel (see the module).
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelEstimator {
    /// Least squares, $A = P^{\dagger}$ (eq. (4)).
    Ls,
    /// Scaled least squares, $A = \gamma_{o} P^{\dagger}$ (eqs. (14), (16)).
    ScaledLs(TracePrior),
    /// LMMSE of the correlation $R_{H} = E\\{H^{H} H\\}$ ($t \times t$, eq. (26)).
    Mmse(CMat),
    /// The LMMSE of $R_{H} = \frac{\mathrm{tr}\\{R_{H}\\}}{t} I$ (eq. (48)).
    RelaxedMmse(TracePrior),
}

/// A training matrix $P$ ($t \times N$, see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Training {
    p: CMat,
    /// $(P P^{H})^{-1}$ when $P$ has rank $t$ (needed by LS).
    gram_inv: Option<CMat>,
}

impl Training {
    /// The training of the $t \times N$ matrix `p`, $N \ge t$.
    ///
    /// # Errors
    /// [`Error::Config`] for an empty matrix, fewer columns than rows or a non-finite
    /// entry. A matrix of rank below $t$ is accepted (the LMMSE estimator and the
    /// water-filled training of a singular correlation); LS and scaled LS refuse it.
    pub fn new(p: CMat) -> Result<Training> {
        let (t, n) = p.dim();
        ensure(t >= 1 && n >= t, || format!("the training must be t x N with N >= t >= 1, got {t} x {n}"))?;
        ensure(p.iter().all(|z| z.re.is_finite() && z.im.is_finite()), || "the training must be finite".into())?;
        let gram = matmul(&p.view(), &hermitian(&p.view()).view());
        let gram_inv = inverse_hpd(&gram.view()).ok();
        Ok(Training { p, gram_inv })
    }

    /// The DFT training of eq. (10): $\[P\]_{k,n} = \sqrt{\mathcal{P}/(N t)}\\,
    /// e^{j 2\pi k n / N}$, of orthogonal rows, $P P^{H} = \frac{\mathcal{P}}{t} I$ (9).
    ///
    /// # Errors
    /// [`Error::Config`] for `n < n_tx`, no antenna or a power that is not positive.
    pub fn orthogonal(n_tx: usize, n: usize, total_power: f64) -> Result<Training> {
        ensure(n_tx >= 1 && n >= n_tx, || format!("the training needs N >= t >= 1, got t = {n_tx}, N = {n}"))?;
        ensure(total_power > 0.0 && total_power.is_finite(), || {
            format!("the training power must be positive, got {total_power}")
        })?;
        let a = (total_power / (n * n_tx) as f64).sqrt();
        let p = CMat::from_shape_fn((n_tx, n), |(k, m)| C64::from_polar(a, 2.0 * PI * ((k * m) % n) as f64 / n as f64));
        Training::new(p)
    }

    /// The training that minimises the MSE of the LMMSE estimator of the correlation
    /// `r_h` (eqs. (41)–(44)): the power water-filled on the eigenvectors of $R_{H}$, the
    /// last $N - t$ columns zero.
    ///
    /// # Errors
    /// [`Error::Config`] for a correlation that is not $t \times t$ with $t \ge 1$,
    /// `n < t`, no receive antenna, a noise variance or a power that is not positive.
    pub fn mmse_optimal(
        r_h: &ArrayView2<'_, C64>,
        noise_var: f64,
        n_rx: usize,
        total_power: f64,
        n: usize,
    ) -> Result<Training> {
        let t = check_correlation(r_h)?;
        ensure(n >= t, || format!("the training needs N >= t, got t = {t}, N = {n}"))?;
        ensure(n_rx >= 1, || "at least one receive antenna".into())?;
        ensure(noise_var > 0.0 && noise_var.is_finite(), || {
            format!("the noise variance must be positive, got {noise_var}")
        })?;
        let scale = noise_var * n_rx as f64;
        let (q, lambda, _) = svd(r_h);
        let powers = water_filling_powers(&lambda, total_power / scale)?;
        let mut p = CMat::zeros((t, n));
        for (i, pw) in powers.iter().enumerate() {
            let g = (scale * pw).sqrt();
            for k in 0..t {
                p[(k, i)] = q[(k, i)] * g;
            }
        }
        Training::new(p)
    }

    /// The training symbols $P$ ($t \times N$).
    pub fn symbols(&self) -> &CMat {
        &self.p
    }

    /// Transmit antennas $t$.
    pub fn n_tx(&self) -> usize {
        self.p.nrows()
    }

    /// Training symbols $N$.
    pub fn n_symbols(&self) -> usize {
        self.p.ncols()
    }

    /// The power $\lVert P \rVert_{F}^{2} = \mathcal{P}$ (eq. (5)).
    pub fn total_power(&self) -> f64 {
        self.p.iter().map(C64::norm_sqr).sum()
    }

    /// The estimate $\hat{H} = S A$ ($r \times t$) of the channel from the received
    /// training `s` ($r \times N$), the noise variance per entry `noise_var`
    /// ($\sigma_{n}^{2}$, unused by LS).
    ///
    /// # Errors
    /// [`Error::Shape`] when `s` does not have $N$ columns or the correlation of
    /// [`ChannelEstimator::Mmse`] is not $t \times t$; [`Error::Config`] for a negative
    /// noise variance, a trace that is not positive, LS or scaled LS on a training of
    /// rank below $t$; [`Error::Numerical`] for a singular LMMSE system (no noise and a
    /// rank-deficient $P^{H} R_{H} P$).
    pub fn estimate(&self, s: &ArrayView2<'_, C64>, estimator: &ChannelEstimator, noise_var: f64) -> Result<CMat> {
        ensure_shape(s.ncols() == self.n_symbols(), || {
            format!("the received training has {} columns, the training {}", s.ncols(), self.n_symbols())
        })?;
        ensure(s.nrows() >= 1, || "at least one receive antenna".into())?;
        let from_ls = || -> Result<f64> {
            let h_ls = matmul(s, &self.ls_filter()?.view());
            Ok(h_ls.iter().map(C64::norm_sqr).sum())
        };
        let a = self.filter(estimator, noise_var, s.nrows(), from_ls)?;
        Ok(matmul(s, &a.view()))
    }

    /// The MSE $E\\{\lVert H - \hat{H} \rVert_{F}^{2}\\}$ of `estimator` on a channel of
    /// true correlation `r_h` ($E\\{H^{H} H\\}$, $t \times t$) with `n_rx` receive
    /// antennas and the noise variance `noise_var` (see the module): eqs. (7), (13),
    /// (28) and (46) for any training, (12), (20), (32) and (49) for the optimal ones.
    ///
    /// # Errors
    /// [`Error::Config`] for LS-SLS and LS-RMMSE ([`TracePrior::FromLs`]: no closed
    /// form) and the errors of [`Training::estimate`]; [`Error::Shape`] for a
    /// correlation that is not $t \times t$.
    pub fn mse(
        &self,
        estimator: &ChannelEstimator,
        r_h: &ArrayView2<'_, C64>,
        noise_var: f64,
        n_rx: usize,
    ) -> Result<f64> {
        ensure_shape(r_h.dim() == (self.n_tx(), self.n_tx()), || {
            format!("the correlation is {:?}, the training has {} rows", r_h.dim(), self.n_tx())
        })?;
        ensure(n_rx >= 1, || "at least one receive antenna".into())?;
        let a = self.filter(estimator, noise_var, n_rx, || {
            Err(Error::Config("LS-SLS and LS-RMMSE have no closed-form MSE (estimate it by Monte Carlo)".into()))
        })?;
        let b = identity(self.n_tx()) - matmul(&self.p.view(), &a.view());
        let bias = matmul(&hermitian(&b.view()).view(), &matmul(r_h, &b.view()).view());
        let bias: f64 = (0..self.n_tx()).map(|i| bias[(i, i)].re).sum();
        let noise: f64 = a.iter().map(C64::norm_sqr).sum();
        Ok(bias + noise_var * n_rx as f64 * noise)
    }

    /// $P^{\dagger} = P^{H} (P P^{H})^{-1}$.
    fn ls_filter(&self) -> Result<CMat> {
        let g = self.gram_inv.as_ref().ok_or_else(|| {
            Error::Config(format!("LS needs a training of rank t = {} (P P^H invertible)", self.n_tx()))
        })?;
        Ok(matmul(&hermitian(&self.p.view()).view(), &g.view()))
    }

    /// The LMMSE filter $(P^{H} R P + \sigma_{n}^{2} r I)^{-1} P^{H} R$ (eq. (25)).
    fn lmmse_filter(&self, r: &ArrayView2<'_, C64>, noise_var: f64, n_rx: usize) -> Result<CMat> {
        let ph_r = matmul(&hermitian(&self.p.view()).view(), r);
        let m = matmul(&ph_r.view(), &self.p.view()) + identity(self.n_symbols()) * C64::from(noise_var * n_rx as f64);
        solve_hpd(&m.view(), &ph_r.view())
    }

    /// The matrix $A$ of `estimator`; `from_ls` gives the estimated trace of
    /// [`TracePrior::FromLs`].
    fn filter(
        &self,
        estimator: &ChannelEstimator,
        noise_var: f64,
        n_rx: usize,
        from_ls: impl FnOnce() -> Result<f64>,
    ) -> Result<CMat> {
        ensure(noise_var >= 0.0 && noise_var.is_finite(), || {
            format!("the noise variance must be non-negative, got {noise_var}")
        })?;
        let trace = |prior: &TracePrior| -> Result<f64> {
            let tr = match prior {
                TracePrior::Known(tr) => *tr,
                TracePrior::FromLs => from_ls()?,
            };
            ensure(tr > 0.0 && tr.is_finite(), || format!("the trace of R_H must be positive, got {tr}"))?;
            Ok(tr)
        };
        match estimator {
            ChannelEstimator::Ls => self.ls_filter(),
            ChannelEstimator::ScaledLs(prior) => {
                let a = self.ls_filter()?;
                let g = self.gram_inv.as_ref().map_or(0.0, |g| (0..self.n_tx()).map(|i| g[(i, i)].re).sum());
                let tr = trace(prior)?;
                let j_ls = noise_var * n_rx as f64 * g;
                Ok(a * C64::from(tr / (j_ls + tr)))
            }
            ChannelEstimator::Mmse(r) => {
                check_correlation(&r.view())?;
                ensure_shape(r.nrows() == self.n_tx(), || {
                    format!("the correlation is {:?}, the training has {} rows", r.dim(), self.n_tx())
                })?;
                self.lmmse_filter(&r.view(), noise_var, n_rx)
            }
            ChannelEstimator::RelaxedMmse(prior) => {
                let alpha = trace(prior)? / self.n_tx() as f64;
                self.lmmse_filter(&(identity(self.n_tx()) * C64::from(alpha)).view(), noise_var, n_rx)
            }
        }
    }
}

/// The size $t$ of a correlation, square and finite.
fn check_correlation(r: &ArrayView2<'_, C64>) -> Result<usize> {
    let (t, c) = r.dim();
    ensure(t >= 1 && t == c, || format!("the correlation must be square and non-empty, got {t} x {c}"))?;
    ensure(r.iter().all(|z| z.re.is_finite() && z.im.is_finite()), || "the correlation must be finite".into())?;
    Ok(t)
}

/// $1/\kappa_{k} = 1/\mathrm{tr}\\{(P_{k} P_{k}^{H})^{-1}\\}$ of every period.
fn inverse_kappas(trainings: &[Training]) -> Result<Vec<f64>> {
    ensure(!trainings.is_empty(), || "at least one training period".into())?;
    let t = trainings[0].n_tx();
    trainings
        .iter()
        .map(|tr| {
            ensure_shape(tr.n_tx() == t, || "every period must train the same transmit antennas".into())?;
            let g = tr.gram_inv.as_ref().ok_or_else(|| Error::Config("BLUE needs trainings of rank t".into()))?;
            Ok(1.0 / (0..t).map(|i| g[(i, i)].re).sum::<f64>())
        })
        .collect()
}

/// The weights $\alpha_{k}$ of the best linear unbiased combination of the LS estimates
/// of the training periods `trainings` (Lemma 4, eq. (54)); $\mathcal{P}_{k} /
/// \mathcal{P}_{\mathrm{tot}}$ for orthogonal trainings (58).
///
/// # Errors
/// [`Error::Config`] for no period or a training of rank below $t$, [`Error::Shape`]
/// for periods of different $t$.
pub fn blue_weights(trainings: &[Training]) -> Result<Vec<f64>> {
    let inv = inverse_kappas(trainings)?;
    let sum: f64 = inv.iter().sum();
    Ok(inv.iter().map(|k| k / sum).collect())
}

/// The combination $\sum_{k} \alpha_{k} \hat{H}_{k}$ of the estimates `estimates` with
/// the weights `weights` (eq. (52)).
///
/// # Errors
/// [`Error::Shape`] for no estimate, estimates of different shapes or as many weights
/// as estimates.
pub fn blue_combine(estimates: &[CMat], weights: &[f64]) -> Result<CMat> {
    ensure_shape(!estimates.is_empty() && estimates.len() == weights.len(), || {
        format!("{} estimates and {} weights", estimates.len(), weights.len())
    })?;
    let dim = estimates[0].dim();
    ensure_shape(estimates.iter().all(|h| h.dim() == dim), || "the estimates must have the same shape".into())?;
    let mut out = CMat::zeros(dim);
    for (h, w) in estimates.iter().zip(weights) {
        out.scaled_add(C64::from(*w), h);
    }
    Ok(out)
}

/// The MSE $\sigma_{n}^{2} r \sum_{k} \alpha_{k}^{2} \kappa_{k} = \sigma_{n}^{2} r /
/// \sum_{k} 1/\kappa_{k}$ of the BLUE combination of the LS estimates of `trainings`
/// (eqs. (59), (61)); $\sigma_{n}^{2} t^{2} r / \mathcal{P}_{\mathrm{tot}}$ for
/// orthogonal trainings.
///
/// # Errors
/// Those of [`blue_weights`], and [`Error::Config`] for a negative noise variance or no
/// receive antenna.
pub fn blue_mse(trainings: &[Training], noise_var: f64, n_rx: usize) -> Result<f64> {
    ensure(noise_var >= 0.0 && n_rx >= 1, || {
        format!("a non-negative noise variance and a receive antenna, got {noise_var} and {n_rx}")
    })?;
    let inv = inverse_kappas(trainings)?;
    Ok(noise_var * n_rx as f64 / inv.iter().sum::<f64>())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code: a failure is a test failure")]
    use super::*;

    fn exponential(t: usize, eps: f64, r: usize) -> CMat {
        CMat::from_shape_fn((t, t), |(i, j)| C64::from(r as f64 * eps.powi(i.abs_diff(j) as i32)))
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-10 * b.abs().max(1.0)
    }

    #[test]
    fn the_dft_training_has_orthogonal_rows_of_the_power() {
        let p = Training::orthogonal(3, 5, 6.0).unwrap();
        let g = matmul(&p.symbols().view(), &hermitian(&p.symbols().view()).view());
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 2.0 } else { 0.0 };
                assert!((g[(i, j)] - C64::from(want)).norm() < 1e-12);
            }
        }
        assert!(close(p.total_power(), 6.0));
    }

    #[test]
    fn the_mse_of_orthogonal_training_follows_eqs_12_20_49() {
        let (t, r, power, noise) = (4, 3, 20.0, 0.7);
        let p = Training::orthogonal(t, t, power).unwrap();
        let rh = exponential(t, 0.6, r);
        let tr = (r * t) as f64;
        let j_ls = noise * (t * t * r) as f64 / power;
        assert!(close(p.mse(&ChannelEstimator::Ls, &rh.view(), noise, r).unwrap(), j_ls));
        let j_sls = noise * (t * t * r) as f64 * tr / (noise * (t * t * r) as f64 + power * tr);
        let sls = ChannelEstimator::ScaledLs(TracePrior::Known(tr));
        let rmmse = ChannelEstimator::RelaxedMmse(TracePrior::Known(tr));
        assert!(close(p.mse(&sls, &rh.view(), noise, r).unwrap(), j_sls));
        assert!(close(p.mse(&rmmse, &rh.view(), noise, r).unwrap(), j_sls));
        assert!(j_sls < j_ls);
    }

    #[test]
    fn the_lmmse_mse_is_eq_28_and_eq_32_at_its_optimal_training() {
        let (t, r, noise) = (3, 2, 0.5);
        let rh = exponential(t, 0.8, r);
        let mmse = ChannelEstimator::Mmse(rh.clone());
        // (28) for the orthogonal training
        let p = Training::orthogonal(t, t, 9.0).unwrap();
        let inner = inverse_hpd(&rh.view()).unwrap()
            + matmul(&p.symbols().view(), &hermitian(&p.symbols().view()).view()) * C64::from(1.0 / (noise * r as f64));
        let e = inverse_hpd(&inner.view()).unwrap();
        let j28: f64 = (0..t).map(|i| e[(i, i)].re).sum();
        assert!(close(p.mse(&mmse, &rh.view(), noise, r).unwrap(), j28));
        // (32) at high SNR (every mode active), and it beats the orthogonal training
        let power = 200.0;
        let opt = Training::mmse_optimal(&rh.view(), noise, r, power, t).unwrap();
        assert!(close(opt.total_power(), power));
        let tr_inv: f64 = {
            let i = inverse_hpd(&rh.view()).unwrap();
            (0..t).map(|k| i[(k, k)].re).sum()
        };
        let j32 = (t * t) as f64 / (tr_inv + power / (noise * r as f64));
        assert!(close(opt.mse(&mmse, &rh.view(), noise, r).unwrap(), j32));
        let orth = Training::orthogonal(t, t, power).unwrap();
        assert!(opt.mse(&mmse, &rh.view(), noise, r).unwrap() < orth.mse(&mmse, &rh.view(), noise, r).unwrap());
    }

    #[test]
    fn the_water_filled_mse_is_active_modes_over_the_level_plus_the_inactive_eigenvalues() {
        // low power: some modes off, J = k / mu + sum of the inactive eigenvalues
        let (t, r, noise, power) = (4, 2, 1.0, 0.4);
        let rh = exponential(t, 0.9, r);
        let opt = Training::mmse_optimal(&rh.view(), noise, r, power, t).unwrap();
        let (_, lambda, _) = svd(&rh.view());
        let pw = water_filling_powers(&lambda, power / (noise * r as f64)).unwrap();
        let active: Vec<usize> = (0..t).filter(|&i| pw[i] > 0.0).collect();
        assert!(active.len() < t);
        let mu = pw[active[0]] + 1.0 / lambda[active[0]];
        let want = active.len() as f64 / mu + (0..t).filter(|i| pw[*i] == 0.0).map(|i| lambda[i]).sum::<f64>();
        let got = opt.mse(&ChannelEstimator::Mmse(rh.clone()), &rh.view(), noise, r).unwrap();
        assert!(close(got, want), "{got} {want}");
        // LS refuses the rank-deficient training
        assert!(opt.mse(&ChannelEstimator::Ls, &rh.view(), noise, r).is_err());
    }

    #[test]
    fn the_estimators_are_exact_without_noise_and_scaled_ls_shrinks_ls() {
        let (t, r) = (2, 3);
        let p = Training::orthogonal(t, 4, 8.0).unwrap();
        let h = CMat::from_shape_fn((r, t), |(i, j)| C64::new(i as f64 - 1.0, j as f64 + 0.5));
        let s = matmul(&h.view(), &p.symbols().view());
        for est in [ChannelEstimator::Ls, ChannelEstimator::ScaledLs(TracePrior::FromLs)] {
            let e = p.estimate(&s.view(), &est, 0.0).unwrap();
            assert!((&e - &h).iter().all(|z| z.norm() < 1e-12), "{est:?}");
        }
        let ls = p.estimate(&s.view(), &ChannelEstimator::Ls, 1.0).unwrap();
        let sls = p.estimate(&s.view(), &ChannelEstimator::ScaledLs(TracePrior::Known(6.0)), 1.0).unwrap();
        // gamma = tr / (sigma^2 r tr((PP^H)^-1) + tr) = 6 / (3 * 2 * 2 / 8 + 6)
        let gamma = 6.0 / (3.0 * 2.0 * 2.0 / 8.0 + 6.0);
        assert!((&sls - &(ls * C64::from(gamma))).iter().all(|z| z.norm() < 1e-12));
    }

    #[test]
    fn the_blue_weights_are_the_power_shares_for_orthogonal_trainings() {
        let trainings: Vec<Training> =
            [1.0, 3.0, 4.0].iter().map(|&pw| Training::orthogonal(2, 2, pw).unwrap()).collect();
        let w = blue_weights(&trainings).unwrap();
        for (wk, pk) in w.iter().zip([1.0, 3.0, 4.0]) {
            assert!(close(*wk, pk / 8.0));
        }
        // (59): sigma^2 t^2 r / P_tot
        assert!(close(blue_mse(&trainings, 0.5, 3).unwrap(), 0.5 * 4.0 * 3.0 / 8.0));
        let hs = vec![
            CMat::from_elem((1, 2), C64::from(1.0)),
            CMat::from_elem((1, 2), C64::from(3.0)),
            CMat::from_elem((1, 2), C64::from(2.0)),
        ];
        let c = blue_combine(&hs, &w).unwrap();
        assert!((c[(0, 1)] - C64::from((1.0 + 9.0 + 8.0) / 8.0)).norm() < 1e-12);
    }

    #[test]
    fn invalid_inputs_are_errors() {
        assert!(Training::orthogonal(3, 2, 1.0).is_err());
        assert!(Training::orthogonal(2, 2, 0.0).is_err());
        let p = Training::orthogonal(2, 2, 1.0).unwrap();
        let s = CMat::zeros((2, 3));
        assert!(matches!(p.estimate(&s.view(), &ChannelEstimator::Ls, 0.1), Err(Error::Shape(_))));
        let rh = identity(2);
        assert!(p.mse(&ChannelEstimator::ScaledLs(TracePrior::FromLs), &rh.view(), 0.1, 1).is_err());
        assert!(p.mse(&ChannelEstimator::ScaledLs(TracePrior::Known(-1.0)), &rh.view(), 0.1, 1).is_err());
        assert!(p.mse(&ChannelEstimator::Mmse(identity(3)), &rh.view(), 0.1, 1).is_err());
        assert!(blue_weights(&[]).is_err());
        assert!(blue_combine(&[CMat::zeros((1, 2))], &[0.5, 0.5]).is_err());
    }
}
