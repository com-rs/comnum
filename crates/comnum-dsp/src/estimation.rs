//! Waveform-agnostic channel estimation: least squares and linear MMSE.
//!
//! For known symbols $x_{i}$ received as $y_{i} = h_{i} x_{i} + w_{i}$,
//! $w_{i} \sim \mathcal{CN}(0, \sigma^{2})$:
//!
//! - least squares (LS): $\hat h_{i} = y_{i} / x_{i}$, unbiased, of error variance
//!   $\sigma^{2}/|x_{i}|^{2}$ ([`ls`]);
//! - linear MMSE: from the LS estimates $\hat h_{p}$ at the pilot positions $p$, the
//!   channel at the positions $q$ is
//!
//! $$
//! \hat h_{q} = R_{qp} \left(R_{pp} + \frac{\sigma^{2}}{E_{p}} I\right)^{-1} \hat h_{p},
//! $$
//!
//! with the channel correlations $R$ and the pilot energy $E_{p}$ ([`lmmse`], or
//! [`Lmmse`] to factor $R_{pp} + \frac{\sigma^{2}}{E_{p}} I$ once for the symbols that
//! share the pilots and correlations), the estimator of minimum mean square error among
//! the linear ones (van de Beek et al. 1995; Edfors et al. 1998); its error covariance is
//! $R_{qq} - R_{qp} (R_{pp} + \frac{\sigma^{2}}{E_{p}} I)^{-1} R_{pq}$ ([`lmmse_mse`]);
//! - [`frequency_correlation`], the correlation between subcarriers of a power-delay
//!   profile, which the LMMSE estimator needs;
//! - [`LowRank`], the rank-$p$ approximation of the LMMSE estimator of $N$ values from
//!   their $N$ LS estimates, by the eigen-decomposition $R_{hh} = U \Lambda U^{H}$ (Edfors
//!   et al. 1998, eqs. (5)–(9)): $\hat{h}_{p} = U \operatorname{diag}(\delta_{0}, \dots,
//!   \delta_{p-1}, 0, \dots) U^{H} \hat{h}_{\mathrm{ls}}$, $\delta_{k} = \lambda_{k}/(\lambda_{k} +
//!   \beta/\mathrm{SNR})$, with $\beta = E|x|^{2} E|1/x|^{2}$ of the constellation
//!   ([`ls_noise_factor`], $17/9$ for 16-QAM). Its mean square error on a channel of
//!   correlation $\tilde{R}_{hh}$ at $\widetilde{\mathrm{SNR}}$ (their eq. (10)),
//!
//! $$
//! E\|h - \hat{h}_{p}\|^{2} = \sum_{k < p} \left[\tilde{\lambda}_{k} (1 - \delta_{k})^{2} +
//! \frac{\beta}{\widetilde{\mathrm{SNR}}} \delta_{k}^{2}\right] + \sum_{k \ge p} \tilde{\lambda}_{k},
//! \qquad \tilde{\lambda}_{k} = [U^{H} \tilde{R}_{hh} U]_{kk},
//! $$
//!
//! is [`LowRank::mse`], and its floor $\sum_{k \ge p} \tilde{\lambda}_{k}$ (eq. (11)) is
//! [`LowRank::mse_floor`].
//!
//! The interpolation of the estimates between the pilots is [`crate::interpolation`].
//!
//! # Examples
//! ```
//! use comnum_dsp::estimation::ls;
//! use comnum_math::C64;
//! let h = ls(&[C64::new(0.0, 2.0)], &[C64::new(1.0, 1.0)])?;
//! assert!((h[0] - C64::new(1.0, 1.0)).norm() < 1e-12);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J.-J. van de Beek, O. Edfors, M. Sandell, S. K. Wilson and P. O. Börjesson, "On
//!    channel estimation in OFDM systems," *Proc. IEEE VTC*, 815–819, 1995.
//! 2. O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson and P. O. Börjesson, "OFDM
//!    channel estimation by singular value decomposition," *IEEE Trans. Commun.*, 46(7),
//!    931–939, 1998.
//! 3. S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*,
//!    Prentice Hall, 1993, ch. 12.

use comnum_core::{Error, Result, ensure, ensure_shape, same_len};
use comnum_math::C64;
use comnum_math::linalg::{CMat, cholesky, hermitian, identity, solve_cholesky, solve_hpd};

/// The frequency correlation of a channel of power-delay profile `(delays, powers)`
/// between two subcarriers `delta` bins apart on a grid of `n_fft` bins,
/// $R(\Delta k) = \sum_{l} p_{l}\, e^{-j 2\pi \Delta k\, d_{l} / N}$, the delays in samples
/// of the IFFT of size $N$: the correlations of the LMMSE estimator.
///
/// # Panics
/// If `delays` and `powers` have different lengths.
///
/// # Examples
/// ```
/// use comnum_dsp::estimation::frequency_correlation;
/// // two equal paths 4 samples apart: |R| = |cos(πΔk·4/N)|
/// let r = frequency_correlation(&[0.0, 4.0], &[0.5, 0.5], 2.0, 16);
/// assert!(r.norm() < 1e-15);
/// ```
pub fn frequency_correlation(delays: &[f64], powers: &[f64], delta: f64, n_fft: usize) -> C64 {
    assert_eq!(delays.len(), powers.len(), "frequency_correlation: one power per delay");
    delays
        .iter()
        .zip(powers)
        .map(|(&d, &p)| comnum_math::cis(-2.0 * std::f64::consts::PI * delta * d / n_fft as f64) * p)
        .sum()
}

/// Least-squares estimates $y_{i}/x_{i}$.
///
/// # Errors
/// [`Error::Shape`] if `y` and `x` have different lengths.
pub fn ls(y: &[C64], x: &[C64]) -> Result<Vec<C64>> {
    same_len(y.len(), x.len(), "ls: received and known symbols")?;
    Ok(y.iter().zip(x).map(|(a, b)| a / b).collect())
}

/// $R_{pp} + \frac{\sigma^{2}}{E_{p}} I$.
pub(crate) fn regularized(r_pp: &CMat, noise_to_pilot: f64) -> CMat {
    r_pp + &identity(r_pp.nrows()).mapv(|v| v * noise_to_pilot)
}

/// The linear MMSE estimates at the positions of `r_qp` (rows) from the LS estimates
/// `h_ls` at the pilots, with `noise_to_pilot` = $\sigma^{2}/E_{p}$: [`Lmmse::new`] then
/// [`Lmmse::apply`], which factors the system again at each call.
///
/// # Errors
/// [`Error::Shape`] for inconsistent sizes, [`Error::Numerical`] for a singular system.
pub fn lmmse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, h_ls: &[C64]) -> Result<Vec<C64>> {
    ensure(noise_to_pilot >= 0.0, || "negative noise".into())?;
    ensure_shape(r_pp.nrows() == h_ls.len() && r_qp.ncols() == h_ls.len(), || {
        format!("{} pilots for correlations of {} pilots", h_ls.len(), r_pp.nrows())
    })?;
    Lmmse::new(r_qp.clone(), r_pp, noise_to_pilot)?.apply(h_ls)
}

/// The linear MMSE estimator of [`lmmse`] for one set of pilots and correlations, the
/// Cholesky factor of $R_{pp} + \frac{\sigma^{2}}{E_{p}} I$ computed once: each
/// [`Lmmse::apply`] costs two triangular substitutions and the product by $R_{qp}$,
/// $O(P^{2} + QP)$ for $P$ pilots and $Q$ estimated positions instead of $O(P^{3})$,
/// with the same operations as [`lmmse`] (the same results, bit for bit).
///
/// # Examples
/// ```
/// use comnum_dsp::estimation::{lmmse, Lmmse};
/// use comnum_math::C64;
/// use ndarray::Array2;
/// let r = Array2::from_shape_fn((4, 4), |(i, j)| C64::new(0.9f64.powi((i as i32 - j as i32).abs()), 0.0));
/// let est = Lmmse::new(r.clone(), &r, 0.1)?;
/// let h_ls = [C64::new(1.0, 0.0), C64::new(0.8, 0.1), C64::new(0.7, 0.3), C64::new(0.5, 0.2)];
/// assert_eq!(est.apply(&h_ls)?, lmmse(&r, &r, 0.1, &h_ls)?);
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Lmmse {
    r_qp: CMat,
    factor: CMat,
}

impl Lmmse {
    /// The estimator at the positions of `r_qp` (rows) from the pilots of `r_pp`, with
    /// `noise_to_pilot` = $\sigma^{2}/E_{p}$.
    ///
    /// # Errors
    /// [`Error::Config`] for a negative noise, [`Error::Shape`] unless `r_pp` is square
    /// with one column of `r_qp` per pilot, [`Error::Numerical`] for a singular system.
    pub fn new(r_qp: CMat, r_pp: &CMat, noise_to_pilot: f64) -> Result<Lmmse> {
        ensure(noise_to_pilot >= 0.0, || "negative noise".into())?;
        ensure_shape(r_pp.nrows() == r_pp.ncols() && r_qp.ncols() == r_pp.nrows(), || {
            format!("correlations of {}×{} pilots and {} columns", r_pp.nrows(), r_pp.ncols(), r_qp.ncols())
        })?;
        let factor = cholesky(&regularized(r_pp, noise_to_pilot).view())?;
        Ok(Lmmse { r_qp, factor })
    }

    /// The number of pilots $P$.
    pub fn n_pilots(&self) -> usize {
        self.factor.nrows()
    }

    /// The estimates at the positions of the rows of $R_{qp}$ from the LS estimates
    /// `h_ls` at the pilots.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `h_ls` has one estimate per pilot.
    pub fn apply(&self, h_ls: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(h_ls.len() == self.n_pilots(), || {
            format!("{} pilots for correlations of {} pilots", h_ls.len(), self.n_pilots())
        })?;
        let b =
            ndarray::Array2::from_shape_vec((h_ls.len(), 1), h_ls.to_vec()).map_err(|e| Error::Shape(e.to_string()))?;
        let z = solve_cholesky(&self.factor.view(), &b.view())?;
        Ok(self.r_qp.dot(&z).column(0).to_vec())
    }
}

/// The mean square error of every LMMSE estimate (the diagonal of its error covariance),
/// given the variances `r_qq_diag` of the channel at the estimated positions.
///
/// # Errors
/// As [`lmmse`]; [`Error::Shape`] unless `r_qq_diag` has one variance per row of `r_qp`.
pub fn lmmse_mse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, r_qq_diag: &[f64]) -> Result<Vec<f64>> {
    ensure_shape(r_qq_diag.len() == r_qp.nrows(), || {
        format!("{} variances for {} estimated positions", r_qq_diag.len(), r_qp.nrows())
    })?;
    let rpq = hermitian(&r_qp.view());
    let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &rpq.view())?;
    let g = r_qp.dot(&z);
    Ok(r_qq_diag.iter().enumerate().map(|(i, v)| (v - g[[i, i]].re).max(0.0)).collect())
}

/// The noise enhancement of LS estimates averaged over a constellation of equiprobable
/// points, $\beta = E|x|^{2}\, E|1/x|^{2}$ (Edfors et al. 1998, Sec. III-A): 1 for a
/// constant modulus, $17/9$ for 16-QAM.
///
/// # Errors
/// [`Error::Config`] for no point or a point at the origin.
///
/// # Examples
/// ```
/// use comnum_dsp::estimation::ls_noise_factor;
/// use comnum_math::C64;
/// let qam16: Vec<C64> = (0..16).map(|i| C64::new(f64::from(2 * (i % 4) - 3), f64::from(2 * (i / 4) - 3))).collect();
/// assert!((ls_noise_factor(&qam16)? - 17.0 / 9.0).abs() < 1e-14);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ls_noise_factor(points: &[C64]) -> Result<f64> {
    ensure(!points.is_empty() && points.iter().all(|p| p.norm_sqr() > 0.0), || {
        "the constellation needs points away from the origin".into()
    })?;
    let n = points.len() as f64;
    let energy = points.iter().map(C64::norm_sqr).sum::<f64>() / n;
    let inverse = points.iter().map(|p| 1.0 / p.norm_sqr()).sum::<f64>() / n;
    Ok(energy * inverse)
}

/// The rank-$p$ LMMSE estimator of Edfors et al. (1998) from the eigen-decomposition of
/// the correlation $R_{hh}$ of the $N$ values (see the module).
///
/// # Examples
/// ```
/// use comnum_dsp::estimation::LowRank;
/// use comnum_math::C64;
/// use comnum_math::linalg::CMat;
/// // a flat channel: one eigenvalue N, rank 1 suffices
/// let r = CMat::from_elem((8, 8), C64::new(1.0, 0.0));
/// let est = LowRank::new(&r, 0.1, 1)?;
/// assert!((est.eigenvalues()[0] - 8.0).abs() < 1e-9);
/// assert!(est.mse_floor(&r)? < 1e-12);
/// let h = est.estimate(&[C64::new(1.0, 0.0); 8])?;
/// assert!((h[3].re - 8.0 / 8.1).abs() < 1e-9);
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LowRank {
    u: CMat,
    lambda: Vec<f64>,
    delta: Vec<f64>,
    rank: usize,
}

impl LowRank {
    /// The estimator of rank `rank` for the correlation `r_hh` ($N \times N$, Hermitian,
    /// positive semi-definite) and the noise of the LS estimates `noise` =
    /// $\beta/\mathrm{SNR}$.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `r_hh` is square, [`Error::Config`] for a rank of 0 or above
    /// $N$, or a negative noise.
    pub fn new(r_hh: &CMat, noise: f64, rank: usize) -> Result<LowRank> {
        let n = r_hh.nrows();
        ensure_shape(r_hh.ncols() == n, || format!("the correlation is {n} x {}, not square", r_hh.ncols()))?;
        ensure((1..=n).contains(&rank), || format!("the rank must be between 1 and {n}, got {rank}"))?;
        ensure(noise >= 0.0, || "negative noise".into())?;
        // a Hermitian positive semi-definite matrix: its SVD is its eigen-decomposition
        let (u, lambda, _) = comnum_math::linalg::svd(&r_hh.view());
        let delta = lambda.iter().map(|&l| if l + noise > 0.0 { l / (l + noise) } else { 0.0 }).collect();
        Ok(LowRank { u, lambda, delta, rank })
    }

    /// The eigenvalues $\lambda_{k}$ of $R_{hh}$, decreasing.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.lambda
    }

    /// The estimate $\hat{h}_{p} = \sum_{k < p} \delta_{k} (u_{k}^{H} \hat{h}_{\mathrm{ls}})\, u_{k}$
    /// (eq. (9)).
    ///
    /// # Errors
    /// [`Error::Shape`] unless `h_ls` has $N$ values.
    pub fn estimate(&self, h_ls: &[C64]) -> Result<Vec<C64>> {
        let n = self.u.nrows();
        same_len(h_ls.len(), n, "LS estimates and the correlation")?;
        let mut out = vec![C64::new(0.0, 0.0); n];
        for k in 0..self.rank {
            let col = self.u.column(k);
            let c: C64 = col.iter().zip(h_ls).map(|(u, h)| u.conj() * h).sum::<C64>() * self.delta[k];
            for (o, u) in out.iter_mut().zip(col.iter()) {
                *o += c * u;
            }
        }
        Ok(out)
    }

    /// $\tilde{\lambda}_{k} = [U^{H} \tilde{R}_{hh} U]_{kk}$ for the $p$ kept eigenvectors,
    /// and the energy left, $\operatorname{tr} \tilde{R}_{hh} - \sum_{k < p} \tilde{\lambda}_{k}$
    /// (the trace, as the eigenvectors of zero eigenvalues may be returned empty).
    fn projected(&self, r_true: &CMat) -> (Vec<f64>, f64) {
        let kept: Vec<f64> = (0..self.rank)
            .map(|k| {
                let col = self.u.column(k);
                let ru: Vec<C64> =
                    (0..r_true.nrows()).map(|i| (0..col.len()).map(|j| r_true[[i, j]] * col[j]).sum()).collect();
                col.iter().zip(&ru).map(|(u, v)| u.conj() * v).sum::<C64>().re
            })
            .collect();
        let trace: f64 = (0..r_true.nrows()).map(|i| r_true[[i, i]].re).sum();
        let left = (trace - kept.iter().sum::<f64>()).max(0.0);
        (kept, left)
    }

    /// The mean square error $E\|h - \hat{h}_{p}\|^{2}$ (summed over the $N$ values) on a
    /// channel of correlation `r_true` with LS estimates of noise `noise_true` =
    /// $\beta/\widetilde{\mathrm{SNR}}$ (eq. (10); the design's own for a matched estimator).
    ///
    /// # Errors
    /// [`Error::Shape`] unless `r_true` has the size of the design.
    pub fn mse(&self, r_true: &CMat, noise_true: f64) -> Result<f64> {
        ensure_shape(r_true.dim() == self.u.dim(), || "the true correlation has another size than the design".into())?;
        let (lt, left) = self.projected(r_true);
        let kept: f64 =
            (0..self.rank).map(|k| lt[k] * (1.0 - self.delta[k]).powi(2) + noise_true * self.delta[k].powi(2)).sum();
        Ok(kept + left)
    }

    /// The floor $\sum_{k \ge p} \tilde{\lambda}_{k}$ of the error at infinite SNR (eq. (11)).
    ///
    /// # Errors
    /// [`Error::Shape`] unless `r_true` has the size of the design.
    pub fn mse_floor(&self, r_true: &CMat) -> Result<f64> {
        ensure_shape(r_true.dim() == self.u.dim(), || "the true correlation has another size than the design".into())?;
        Ok(self.projected(r_true).1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "frequency_correlation: one power per delay")]
    fn frequency_correlation_rejects_a_power_per_delay_mismatch() {
        let _ = frequency_correlation(&[0.0, 1.0], &[1.0], 1.0, 16);
    }
    use comnum_core::rng::Rng;
    use ndarray::Array2;

    #[test]
    fn lmmse_beats_ls_and_follows_its_predicted_error() {
        // a smooth channel: R_ij = ρ^|i-j| with ρ = 0.95, 8 positions, SNR 0 dB
        let n = 8;
        let r = Array2::from_shape_fn((n, n), |(i, j)| C64::new(0.95f64.powi((i as i32 - j as i32).abs()), 0.0));
        let l = comnum_math::linalg::cholesky(&r.view()).unwrap();
        let mut rng = Rng::new(1);
        let (mut e_ls, mut e_mm) = (0.0, 0.0);
        for _ in 0..2000 {
            let g: Vec<C64> = (0..n).map(|_| rng.complex_gaussian()).collect();
            let h = comnum_math::linalg::matvec(&l.view(), &g);
            let y: Vec<C64> = h.iter().map(|v| v + rng.complex_gaussian()).collect();
            let x = vec![C64::new(1.0, 0.0); n];
            let hl = ls(&y, &x).unwrap();
            let hm = lmmse(&r, &r, 1.0, &hl).unwrap();
            e_ls += hl.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
            e_mm += hm.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
        }
        assert!((e_ls / 2000.0 / n as f64 - 1.0).abs() < 0.05);
        let predicted = lmmse_mse(&r, &r, 1.0, &vec![1.0; n]).unwrap().iter().sum::<f64>() / n as f64;
        assert!((e_mm / 2000.0 / n as f64 / predicted - 1.0).abs() < 0.08);
        assert!(predicted < 0.4);
    }

    /// The LMMSE estimate as it was computed before: [`solve_hpd`] for every vector.
    fn former_lmmse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, h_ls: &[C64]) -> Vec<C64> {
        let b = Array2::from_shape_vec((h_ls.len(), 1), h_ls.to_vec()).unwrap();
        let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &b.view()).unwrap();
        r_qp.dot(&z).column(0).to_vec()
    }

    #[test]
    fn the_factored_estimator_is_the_former_solve_for_every_symbol() {
        let mut rng = Rng::new(3);
        let (p, q) = (12, 40);
        let r_pp = Array2::from_shape_fn((p, p), |(i, j)| {
            frequency_correlation(&[0.0, 1.5, 4.0], &[0.6, 0.3, 0.1], (4 * i) as f64 - (4 * j) as f64, 64)
        });
        let r_qp = Array2::from_shape_fn((q, p), |(k, j)| {
            frequency_correlation(&[0.0, 1.5, 4.0], &[0.6, 0.3, 0.1], k as f64 - (4 * j) as f64, 64)
        });
        for noise in [1e-9, 0.01, 1.0] {
            let est = Lmmse::new(r_qp.clone(), &r_pp, noise).unwrap();
            for _ in 0..5 {
                let h: Vec<C64> = (0..p).map(|_| rng.complex_gaussian()).collect();
                assert_eq!(est.apply(&h).unwrap(), former_lmmse(&r_qp, &r_pp, noise, &h), "noise {noise}");
                assert_eq!(lmmse(&r_qp, &r_pp, noise, &h).unwrap(), est.apply(&h).unwrap());
            }
        }
        let est = Lmmse::new(r_qp.clone(), &r_pp, 0.1).unwrap();
        assert!(matches!(est.apply(&vec![C64::new(1.0, 0.0); p - 1]), Err(Error::Shape(_))));
        assert!(matches!(Lmmse::new(r_qp, &identity(p + 1), 0.1), Err(Error::Shape(_))));
        assert!(Lmmse::new(identity(3), &identity(3), -1.0).is_err());
    }

    #[test]
    fn ls_and_lmmse_mse_reject_inputs_of_different_lengths() {
        let one = C64::new(1.0, 0.0);
        assert!(matches!(ls(&[one; 3], &[one; 2]), Err(Error::Shape(_))));
        let r = identity(3);
        assert!(matches!(lmmse_mse(&r, &r, 0.1, &[1.0; 2]), Err(Error::Shape(_))));
        assert!(matches!(lmmse_mse(&r, &r, 0.1, &[1.0; 4]), Err(Error::Shape(_))));
        assert_eq!(lmmse_mse(&r, &r, 0.1, &[1.0; 3]).unwrap().len(), 3);
    }
}
