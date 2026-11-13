//! Error rates over flat fading channels by the moment generating function (MGF) of
//! the SNR: [`SnrFading`] and its MGF, the averages [`ser_mgf`], [`ber_binary_mgf`] and
//! [`ber_noncoherent_binary_mgf`] for a caller's MGF, the same over a named channel
//! ([`ser_fading`], [`ber_binary_fading`], [`ber_noncoherent_binary_fading`]), and the
//! closed form of maximal-ratio combining over Rayleigh branches ([`ber_rayleigh_mrc`]).
//!
//! # References
//!
//! 1. M. K. Simon and M.-S. Alouini, "A unified approach to the performance analysis of
//!    digital communication over generalized fading channels," *Proc. IEEE* 86(9), 1998.
//! 2. M.-S. Alouini and A. J. Goldsmith, "A unified approach for calculating error rates
//!    of linearly modulated signals over generalized fading channels," *IEEE Trans.
//!    Commun.* 47(9), 1999.

use super::ensure_order;
use crate::constellation::ConstellationKind;
use comnum_core::channel::FlatFading;
use comnum_core::{Error, Result, ensure};

/// $\frac{1}{\pi}\int_{0}^{b} \mathcal{M}(-c/\sin^{2}\eta)\, d\eta$, the integral of every error
/// rate of the MGF approach, by Gauss–Legendre quadrature of 96 nodes.
fn mgf_integral(mgf: &impl Fn(f64) -> f64, c: f64, b: f64) -> f64 {
    let (x, w) = comnum_math::gauss_legendre(96);
    x.iter().zip(&w).map(|(t, wt)| wt * mgf(-c / (b / 2.0 * (t + 1.0)).sin().powi(2))).sum::<f64>() * b
        / 2.0
        / std::f64::consts::PI
}

/// The average SER of `m`-PSK, square `m`-QAM or `m`-PAM (M-AM) over a fading channel
/// whose SNR per symbol $\gamma$ has the moment generating function `mgf` $(s) =
/// \mathbb{E}\, e^{s\gamma}$, $s \le 0$ ([`SnrFading::mgf`] for the usual channels), by
/// the MGF approach with Craig's form of the $Q$ function (M.-S. Alouini and A. J.
/// Goldsmith, "A unified approach for calculating error rates of linearly modulated
/// signals over generalized fading channels," *IEEE Trans. Commun.* 47(9), 1999, eqs.
/// (41), (45) and (48); E. Basar et al., "Wireless communications through
/// reconfigurable intelligent surfaces," *IEEE Access*, 2019, eqs. (16) and (21)): with
/// $g = \sin^{2}(\pi/M)$,
///
/// $$
/// P_{e}^{\mathrm{PSK}} = \frac{1}{\pi} \int_{0}^{(M-1)\pi/M} M_{\gamma}\Bigl(-\frac{g}{\sin^{2}\eta}\Bigr) d\eta ,
/// $$
///
/// with $c = 3/(2(M-1))$ and $q = 1 - 1/\sqrt{M}$,
///
/// $$
/// P_{e}^{\mathrm{QAM}} = \frac{4q}{\pi} \int_{0}^{\pi/2} M_{\gamma}\Bigl(-\frac{c}{\sin^{2}\eta}\Bigr) d\eta -
///   \frac{4q^{2}}{\pi} \int_{0}^{\pi/4} M_{\gamma}\Bigl(-\frac{c}{\sin^{2}\eta}\Bigr) d\eta ,
/// $$
///
/// and with $g = 3/(M^{2} - 1)$,
///
/// $$
/// P_{e}^{\mathrm{PAM}} = \frac{2(M-1)}{M\pi} \int_{0}^{\pi/2} M_{\gamma}\Bigl(-\frac{g}{\sin^{2}\eta}\Bigr) d\eta ,
/// $$
///
/// by Gauss–Legendre quadrature of 96 nodes on every interval. With the MGF
/// $e^{s\gamma_{0}}$ of a constant SNR, these are the SERs of [`ser_esn0`](super::ser_esn0) over AWGN.
///
/// # Errors
/// [`Error::Config`] for another family or an order that is not a power of two (and,
/// for QAM, not a square).
///
/// # Examples
/// ```
/// use comnum_mapping::{theory, constellation::ConstellationKind};
/// // BPSK over Rayleigh fading of mean SNR 10: M(s) = 1/(1 - 10 s)
/// let p = theory::ser_mgf(ConstellationKind::Psk, 2, |s| 1.0 / (1.0 - 10.0 * s))?;
/// assert!((p - 0.5 * (1.0 - (10.0f64 / 11.0).sqrt())).abs() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ser_mgf(kind: ConstellationKind, m: usize, mgf: impl Fn(f64) -> f64) -> Result<f64> {
    ensure_order(kind, m)?;
    let mf = m as f64;
    let pi = std::f64::consts::PI;
    match kind {
        ConstellationKind::Psk => {
            let g = (pi / mf).sin().powi(2);
            Ok(mgf_integral(&mgf, g, (mf - 1.0) * pi / mf))
        }
        ConstellationKind::Qam => {
            let (c, q) = (1.5 / (mf - 1.0), 1.0 - 1.0 / mf.sqrt());
            Ok(4.0 * q * mgf_integral(&mgf, c, pi / 2.0) - 4.0 * q * q * mgf_integral(&mgf, c, pi / 4.0))
        }
        ConstellationKind::Pam => {
            let g = 3.0 / (mf * mf - 1.0);
            Ok(2.0 * (mf - 1.0) / mf * mgf_integral(&mgf, g, pi / 2.0))
        }
        other => Err(Error::Config(format!("no SER by the MGF for {other:?}"))),
    }
}

/// The average BER of a coherent binary signal whose conditional BER is
/// $Q(\sqrt{2 g \gamma})$ over a fading channel of MGF `mgf` (M. K. Simon and M.-S.
/// Alouini, "A unified approach to the performance analysis of digital communication
/// over generalized fading channels," *Proc. IEEE* 86(9), 1998, eqs. (17) and (23);
/// Alouini & Goldsmith 1999, eq. (20)):
///
/// $$
/// P_{b} = \frac{1}{\pi} \int_{0}^{\pi/2} M_{\gamma}\Bigl(-\frac{g}{\sin^{2}\eta}\Bigr) d\eta ,
/// $$
///
/// with $\gamma$ the SNR per bit and `g` = 1 for BPSK (and Gray QPSK per bit), $1/2$
/// for coherent orthogonal BFSK. With maximal-ratio combining of independent branches
/// the MGF is the product of theirs (1999, eq. (37)).
///
/// # Errors
/// [`Error::Config`] for a `g` that is not positive.
///
/// # Examples
/// ```
/// use comnum_mapping::theory::{ber_binary_mgf, ber_rayleigh_mrc, SnrFading};
/// let g = 10.0;
/// let p = ber_binary_mgf(1.0, |s| SnrFading::Rayleigh.mgf(s, g).unwrap())?;
/// assert!((p - ber_rayleigh_mrc(g, 1)?).abs() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ber_binary_mgf(g: f64, mgf: impl Fn(f64) -> f64) -> Result<f64> {
    check_binary_constant(g)?;
    Ok(mgf_integral(&mgf, g, std::f64::consts::PI / 2.0))
}

/// The constant `g` of a binary signal, positive and finite.
fn check_binary_constant(g: f64) -> Result<()> {
    ensure(g > 0.0 && g.is_finite(), || format!("the constant g of the binary signal must be positive, got {g}"))
}

/// The average BER of a binary signal detected noncoherently or differentially, whose
/// conditional BER is $\frac{1}{2} e^{-g\gamma}$, over a fading channel of MGF `mgf`:
/// $P_{b} = \frac{1}{2} M_{\gamma}(-g)$ (Simon & Alouini 1998, eq. (53) averaged, the
/// case $\zeta = 0$ of eq. (63)), with `g` = 1 for DPSK and $1/2$ for noncoherent
/// orthogonal BFSK; over Rayleigh fading $\frac{1}{2(1 + \bar\gamma)}$ and
/// $\frac{1}{2 + \bar\gamma}$ (eq. (64)).
///
/// # Errors
/// [`Error::Config`] for a `g` that is not positive.
///
/// # Examples
/// ```
/// use comnum_mapping::theory::{ber_noncoherent_binary_mgf, SnrFading};
/// let p = ber_noncoherent_binary_mgf(1.0, |s| SnrFading::Rayleigh.mgf(s, 10.0).unwrap())?;
/// assert!((p - 1.0 / 22.0).abs() < 1e-15);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ber_noncoherent_binary_mgf(g: f64, mgf: impl Fn(f64) -> f64) -> Result<f64> {
    check_binary_constant(g)?;
    Ok(0.5 * mgf(-g))
}

/// The distribution of the SNR per symbol $\gamma$ of a flat fading channel, for the
/// error rates by the MGF approach ([`ser_mgf`], [`ber_binary_mgf`],
/// [`ber_noncoherent_binary_mgf`]): the pdfs of Table 2 of Simon & Alouini (1998) and
/// eqs. (3)–(9) of Alouini & Goldsmith (1999), of mean $\bar\gamma$.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SnrFading {
    /// No fading: $\gamma = \bar\gamma$.
    Awgn,
    /// Rayleigh fading, $p(\gamma) = e^{-\gamma/\bar\gamma}/\bar\gamma$.
    Rayleigh,
    /// Rice (Nakagami-n) fading of Rician factor $K = n^{2}$ (linear, the power of the
    /// line of sight over that of the scattered paths): 0 is Rayleigh, $\infty$ no fading.
    Rice {
        /// $K$ (linear).
        k_factor: f64,
    },
    /// Nakagami-m fading, $p(\gamma) = \frac{m^{m}\gamma^{m-1}}{\bar\gamma^{m}\Gamma(m)}
    /// e^{-m\gamma/\bar\gamma}$: $m = 1/2$ is the one-sided Gaussian, 1 Rayleigh, $\infty$
    /// no fading.
    Nakagami {
        /// $m \ge 1/2$.
        m: f64,
    },
    /// Hoyt (Nakagami-q) fading: $q = 0$ is the one-sided Gaussian, 1 Rayleigh.
    Hoyt {
        /// $0 \le q \le 1$.
        q: f64,
    },
}

impl From<FlatFading> for SnrFading {
    fn from(fading: FlatFading) -> SnrFading {
        match fading {
            FlatFading::Rayleigh => SnrFading::Rayleigh,
            FlatFading::Rice { k_factor } => SnrFading::Rice { k_factor },
            FlatFading::Nakagami { m } => SnrFading::Nakagami { m },
            FlatFading::Hoyt { q } => SnrFading::Hoyt { q },
        }
    }
}

impl SnrFading {
    /// The channel of a name, `awgn`, `rayleigh`, `rice`, `nakagami` or `hoyt`
    /// (case-insensitive), with its parameter ($K$, $m$ or $q$) when it has one: the
    /// names and the checks of [`FlatFading::from_name`], and `awgn`.
    ///
    /// # Errors
    /// [`Error::Config`] for another name, a missing or an invalid parameter.
    pub fn from_name(name: &str, parameter: Option<f64>) -> Result<SnrFading> {
        if name.eq_ignore_ascii_case("awgn") {
            return Ok(SnrFading::Awgn);
        }
        FlatFading::from_name(name, parameter).map(SnrFading::from)
    }

    /// The distribution of the gain of the channel, `None` without fading: what
    /// `comnum_channel::fading::flat_gains` draws to simulate it.
    pub fn gain_distribution(self) -> Option<FlatFading> {
        match self {
            SnrFading::Awgn => None,
            SnrFading::Rayleigh => Some(FlatFading::Rayleigh),
            SnrFading::Rice { k_factor } => Some(FlatFading::Rice { k_factor }),
            SnrFading::Nakagami { m } => Some(FlatFading::Nakagami { m }),
            SnrFading::Hoyt { q } => Some(FlatFading::Hoyt { q }),
        }
    }

    fn check(self) -> Result<()> {
        self.gain_distribution().map_or(Ok(()), FlatFading::check)
    }

    /// The moment generating function $M_{\gamma}(s) = \mathbb{E}\, e^{s\gamma}$, $s \le
    /// 0$, of the SNR of mean `mean_snr_lin` $= \bar\gamma$ (Alouini & Goldsmith 1999,
    /// eqs. (23), (26), (28), (30), written in $s = -g/\sin^{2}\phi$; Simon & Alouini
    /// 1998, Table 3):
    ///
    /// $$
    /// \begin{aligned}
    /// M_{\mathrm{Rayleigh}}(s) &= (1 - s\bar\gamma)^{-1}, \\\\
    /// M_{\mathrm{Rice}}(s) &= \frac{1 + K}{1 + K - s\bar\gamma}
    ///   \exp\Bigl(\frac{K s\bar\gamma}{1 + K - s\bar\gamma}\Bigr), \\\\
    /// M_{\mathrm{Nakagami}}(s) &= \Bigl(1 - \frac{s\bar\gamma}{m}\Bigr)^{-m}, \\\\
    /// M_{\mathrm{Hoyt}}(s) &= \Bigl(1 - 2s\bar\gamma + \frac{4q^{2} s^{2}\bar\gamma^{2}}{(1 + q^{2})^{2}}\Bigr)^{-1/2},
    /// \end{aligned}
    /// $$
    ///
    /// and $e^{s\bar\gamma}$ without fading. [`SnrFading::mgf_of_mean`] gives it as a
    /// function of $s$, the arguments checked once.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid parameter, a negative mean or a positive `s`.
    ///
    /// # Examples
    /// ```
    /// use comnum_mapping::theory::SnrFading;
    /// // K = 0 and m = 1 are Rayleigh fading
    /// let r = SnrFading::Rayleigh.mgf(-0.3, 5.0)?;
    /// assert!((SnrFading::Rice { k_factor: 0.0 }.mgf(-0.3, 5.0)? - r).abs() < 1e-15);
    /// assert!((SnrFading::Nakagami { m: 1.0 }.mgf(-0.3, 5.0)? - r).abs() < 1e-15);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn mgf(self, s: f64, mean_snr_lin: f64) -> Result<f64> {
        ensure(s <= 0.0, || format!("the MGF of the SNR is taken at s <= 0, got {s}"))?;
        Ok(self.mgf_of_mean(mean_snr_lin)?(s))
    }

    /// [`SnrFading::mgf`] of the SNR of mean `mean_snr_lin` as a function of $s$: the
    /// `mgf` argument of [`ser_mgf`], [`ber_binary_mgf`] and
    /// [`ber_noncoherent_binary_mgf`], which call it at $s \le 0$ only.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid parameter or a negative mean.
    ///
    /// # Examples
    /// ```
    /// use comnum_mapping::theory::{ser_mgf, SnrFading};
    /// use comnum_mapping::constellation::ConstellationKind;
    /// let m = SnrFading::Nakagami { m: 2.0 }.mgf_of_mean(10.0)?;
    /// assert!((m(-1.0) - 1.0 / 36.0).abs() < 1e-15);
    /// let p = ser_mgf(ConstellationKind::Psk, 4, m)?;
    /// assert!(p > 0.0 && p < 0.1);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn mgf_of_mean(self, mean_snr_lin: f64) -> Result<impl Fn(f64) -> f64> {
        self.check()?;
        ensure(mean_snr_lin >= 0.0 && mean_snr_lin.is_finite(), || {
            format!("the mean SNR must be finite and non-negative, got {mean_snr_lin}")
        })?;
        Ok(move |s: f64| self.at(s * mean_snr_lin))
    }

    /// $M_{\gamma}(s)$ at $x = s\bar\gamma$, the parameters checked.
    fn at(self, x: f64) -> f64 {
        match self {
            SnrFading::Awgn => x.exp(),
            SnrFading::Rayleigh => 1.0 / (1.0 - x),
            SnrFading::Rice { k_factor } => {
                (1.0 + k_factor) / (1.0 + k_factor - x) * (k_factor * x / (1.0 + k_factor - x)).exp()
            }
            SnrFading::Nakagami { m } => (1.0 - x / m).powf(-m),
            SnrFading::Hoyt { q } => {
                let r = 2.0 * q / (1.0 + q * q);
                1.0 / (1.0 - 2.0 * x + r * r * x * x).sqrt()
            }
        }
    }
}

/// [`ser_mgf`] over the channel `fading` with the mean $E_{s}/N_{0}$ `esn0_db` (dB).
///
/// # Errors
/// As [`ser_mgf`] and [`SnrFading::mgf`].
///
/// # Examples
/// ```
/// use comnum_mapping::{theory::{ser_fading, SnrFading}, constellation::ConstellationKind};
/// // 16-QAM over Rayleigh fading at 20 dB: the closed form (49) of Alouini & Goldsmith 1999
/// let (g, q) = (0.1 * 100.0f64, 0.75f64);
/// let mu = (g / (1.0 + g)).sqrt();
/// let closed = 2.0 * q * (1.0 - mu) + q * q * (4.0 / std::f64::consts::PI * mu * (1.0 / mu).atan() - 1.0);
/// let p = ser_fading(ConstellationKind::Qam, 16, SnrFading::Rayleigh, 20.0)?;
/// assert!((p - closed).abs() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ser_fading(kind: ConstellationKind, m: usize, fading: SnrFading, esn0_db: f64) -> Result<f64> {
    ser_mgf(kind, m, fading.mgf_of_mean(comnum_math::db2lin(esn0_db))?)
}

/// [`ber_binary_mgf`] over the channel `fading` with the mean $E_{b}/N_{0}$ `ebn0_db`
/// (dB) and `branches` branches of that mean combined by maximal ratio (Alouini &
/// Goldsmith 1999, eq. (38): the MGF to the power of the branches).
///
/// # Errors
/// As [`ber_binary_mgf`] and [`SnrFading::mgf`], and [`Error::Config`] for no branch.
///
/// # Examples
/// ```
/// use comnum_mapping::theory::{ber_binary_fading, ber_rayleigh_mrc, SnrFading};
/// let p = ber_binary_fading(1.0, SnrFading::Rayleigh, 10.0, 3)?;
/// assert!((p - ber_rayleigh_mrc(10.0, 3)?).abs() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ber_binary_fading(g: f64, fading: SnrFading, ebn0_db: f64, branches: usize) -> Result<f64> {
    ensure(branches >= 1, || "maximal-ratio combining needs at least one branch".into())?;
    let mgf = fading.mgf_of_mean(comnum_math::db2lin(ebn0_db))?;
    ber_binary_mgf(g, |s| mgf(s).powi(branches as i32))
}

/// [`ber_noncoherent_binary_mgf`] over the channel `fading` with the mean
/// $E_{b}/N_{0}$ `ebn0_db` (dB).
///
/// # Errors
/// As [`ber_noncoherent_binary_mgf`] and [`SnrFading::mgf`].
///
/// # Examples
/// ```
/// use comnum_mapping::theory::{ber_noncoherent_binary_fading, SnrFading};
/// // DPSK over Rayleigh fading at 10 dB: 1 / (2 (1 + 10))
/// let p = ber_noncoherent_binary_fading(1.0, SnrFading::Rayleigh, 10.0)?;
/// assert!((p - 1.0 / 22.0).abs() < 1e-15);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ber_noncoherent_binary_fading(g: f64, fading: SnrFading, ebn0_db: f64) -> Result<f64> {
    ber_noncoherent_binary_mgf(g, fading.mgf_of_mean(comnum_math::db2lin(ebn0_db))?)
}

/// The bit error rate of BPSK (or Gray QPSK, per bit) with maximal-ratio combining of
/// `branches` independent Rayleigh-faded branches of mean SNR per bit `mean_ebn0`
/// ($\bar\gamma$, linear) each (Proakis & Salehi, *Digital Communications*, 5th ed.,
/// eq. (13.4-15)):
///
/// $$
/// P_{b} = \left(\frac{1 - \mu}{2}\right)^{L} \sum_{k=0}^{L-1} \binom{L - 1 + k}{k}
/// \left(\frac{1 + \mu}{2}\right)^{k}, \qquad \mu = \sqrt{\frac{\bar\gamma}{1 + \bar\gamma}},
/// $$
///
/// with $\bar\gamma$ the mean $E_{b}/N_{0}$ per branch (linear) and $L$ the number of
/// branches; $L = 1$ is the flat Rayleigh channel, $P_{b} = \frac{1}{2}(1 - \mu)$.
///
/// # Errors
/// [`Error::Config`] for no branch.
///
/// # Examples
/// ```
/// use comnum_mapping::theory::ber_rayleigh_mrc;
/// let g = comnum_math::db2lin(10.0);
/// assert!((ber_rayleigh_mrc(g, 1)? - 0.5 * (1.0 - (g / (1.0 + g)).sqrt())).abs() < 1e-15);
/// // diversity: four branches at 10 dB each
/// assert!(ber_rayleigh_mrc(g, 4)? < 1e-5);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ber_rayleigh_mrc(mean_ebn0_lin: f64, branches: usize) -> Result<f64> {
    ensure(branches >= 1, || "maximal-ratio combining needs at least one branch".into())?;
    let mu = (mean_ebn0_lin / (1.0 + mean_ebn0_lin)).sqrt();
    let l = branches;
    let mut binom = 1.0;
    let mut sum = 0.0;
    for k in 0..l {
        if k > 0 {
            binom *= (l - 1 + k) as f64 / k as f64;
        }
        sum += binom * ((1.0 + mu) / 2.0).powi(k as i32);
    }
    Ok(((1.0 - mu) / 2.0).powi(l as i32) * sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::ser_esn0;

    #[test]
    fn the_mgfs_are_the_laplace_transforms_of_their_densities() {
        // the pdfs of Table 2 of Simon & Alouini 1998, integrated numerically against the
        // closed forms of Alouini & Goldsmith 1999, eqs. (23), (28), (30), (26)
        let mean = 4.0;
        let i0 = |x: f64| comnum_math::bessel_i0e(x) * x.exp();
        let rice = |k: f64, g: f64| {
            (1.0 + k) * (-k).exp() / mean * (-(1.0 + k) * g / mean).exp() * i0(2.0 * (k * (1.0 + k) * g / mean).sqrt())
        };
        let nakagami = |m: f64, g: f64| {
            (m * m.ln() + (m - 1.0) * g.ln() - m * mean.ln() - comnum_math::ln_gamma(m) - m * g / mean).exp()
        };
        let hoyt = |q: f64, g: f64| {
            (1.0 + q * q) / (2.0 * q * mean)
                * (-(1.0 + q * q).powi(2) * g / (4.0 * q * q * mean)).exp()
                * i0((1.0 - q.powi(4)) * g / (4.0 * q * q * mean))
        };
        for s in [-0.05, -0.5, -3.0] {
            let laplace =
                |pdf: &dyn Fn(f64) -> f64| comnum_math::simpson(|g| pdf(g) * (s * g).exp(), 1e-9, 200.0, 400_000);
            for (f, pdf) in [
                (SnrFading::Rice { k_factor: 3.0 }, Box::new(|g| rice(3.0, g)) as Box<dyn Fn(f64) -> f64>),
                (SnrFading::Rice { k_factor: 0.5 }, Box::new(|g| rice(0.5, g))),
                (SnrFading::Nakagami { m: 2.5 }, Box::new(|g| nakagami(2.5, g))),
                (SnrFading::Nakagami { m: 1.0 }, Box::new(|g| nakagami(1.0, g))),
                (SnrFading::Hoyt { q: 0.5 }, Box::new(|g| hoyt(0.5, g))),
                (SnrFading::Hoyt { q: 0.9 }, Box::new(|g| hoyt(0.9, g))),
            ] {
                let (got, num) = (f.mgf(s, mean).unwrap(), laplace(&*pdf));
                assert!((got / num - 1.0).abs() < 1e-6, "{f:?} at s = {s}: {got} against {num}");
            }
        }
    }

    #[test]
    fn the_limits_of_the_fading_families_are_rayleigh_and_awgn() {
        let (s, mean) = (-0.7, 6.0);
        let r = SnrFading::Rayleigh.mgf(s, mean).unwrap();
        for f in [SnrFading::Rice { k_factor: 0.0 }, SnrFading::Nakagami { m: 1.0 }, SnrFading::Hoyt { q: 1.0 }] {
            assert!((f.mgf(s, mean).unwrap() - r).abs() < 1e-15, "{f:?}");
        }
        let awgn = SnrFading::Awgn.mgf(s, mean).unwrap();
        assert!((SnrFading::Nakagami { m: 1e7 }.mgf(s, mean).unwrap() / awgn - 1.0).abs() < 1e-6);
        // Rice: (1 + K)/(1 + K - x) exp(K x / (1 + K - x)) = e^x (1 - x (1 - x) / K + …)
        assert!((SnrFading::Rice { k_factor: 1e7 }.mgf(s, mean).unwrap() / awgn - 1.0).abs() < 1e-5);
        // the SER over AWGN by the MGF is the closed form
        for (kind, m) in [(ConstellationKind::Psk, 2), (ConstellationKind::Qam, 16), (ConstellationKind::Pam, 4)] {
            let e = 12.0;
            let a = ser_fading(kind, m, SnrFading::Awgn, e).unwrap();
            assert!((a / ser_esn0(kind, m, e).unwrap() - 1.0).abs() < 1e-9, "{kind:?}-{m}");
        }
        // invalid inputs
        assert!(SnrFading::Rice { k_factor: -1.0 }.mgf(s, mean).is_err());
        assert!(SnrFading::Nakagami { m: 0.4 }.mgf(s, mean).is_err());
        assert!(SnrFading::Hoyt { q: 1.5 }.mgf(s, mean).is_err());
        assert!(SnrFading::Rayleigh.mgf(0.5, mean).is_err());
        assert!(SnrFading::from_name("rice", None).is_err());
        assert!(SnrFading::from_name("weibull", None).is_err());
        assert_eq!(SnrFading::from_name("Nakagami", Some(2.0)).unwrap(), SnrFading::Nakagami { m: 2.0 });
        assert!(ber_binary_mgf(0.0, |_| 1.0).is_err());
        assert!(ber_binary_fading(1.0, SnrFading::Rayleigh, 10.0, 0).is_err());
    }

    #[test]
    fn the_closed_forms_of_alouini_and_goldsmith_1999_hold() {
        for e in [0.0, 10.0, 20.0] {
            let g = comnum_math::db2lin(e);
            let mu = |c: f64| (c * g / (1.0 + c * g)).sqrt();
            // eq. (24): BPSK over Rayleigh
            let bpsk = ber_binary_fading(1.0, SnrFading::Rayleigh, e, 1).unwrap();
            assert!((bpsk - 0.5 * (1.0 - mu(1.0))).abs() < 1e-13);
            // eq. (49): square M-QAM over Rayleigh
            for m in [4usize, 16, 64] {
                let (c, q) = (1.5 / (m as f64 - 1.0), 1.0 - 1.0 / (m as f64).sqrt());
                let closed =
                    2.0 * q * (1.0 - mu(c)) + q * q * (4.0 / std::f64::consts::PI * mu(c) * (1.0 / mu(c)).atan() - 1.0);
                let p = ser_fading(ConstellationKind::Qam, m, SnrFading::Rayleigh, e).unwrap();
                assert!((p - closed).abs() < 1e-12, "{m}-QAM at {e} dB: {p} against {closed}");
            }
            // Appendix B, eq. (64): BPSK over Nakagami-m with integer m is the MRC
            // closed form of m Rayleigh branches of mean gamma / m
            for m in [1usize, 2, 3, 5] {
                let p = ber_binary_fading(1.0, SnrFading::Nakagami { m: m as f64 }, e, 1).unwrap();
                assert!((p - ber_rayleigh_mrc(g / m as f64, m).unwrap()).abs() < 1e-13, "m = {m}");
                // and the MRC of m i.i.d. Rayleigh branches is the Nakagami-m channel
                let q = ber_binary_fading(1.0, SnrFading::Rayleigh, e - 10.0 * (m as f64).log10(), m).unwrap();
                assert!((p - q).abs() < 1e-13);
            }
            // Simon & Alouini 1998, eq. (64) at zeta = 0: DPSK and noncoherent BFSK over Rayleigh
            assert!(
                (ber_noncoherent_binary_fading(1.0, SnrFading::Rayleigh, e).unwrap() - 0.5 / (1.0 + g)).abs() < 1e-15
            );
            assert!(
                (ber_noncoherent_binary_fading(0.5, SnrFading::Rayleigh, e).unwrap() - 1.0 / (2.0 + g)).abs() < 1e-15
            );
            // M-AM, eq. (45): 2-PAM is BPSK
            let pam = ser_fading(ConstellationKind::Pam, 2, SnrFading::Rice { k_factor: 2.0 }, e).unwrap();
            assert!((pam - ber_binary_fading(1.0, SnrFading::Rice { k_factor: 2.0 }, e, 1).unwrap()).abs() < 1e-14);
        }
    }
}
