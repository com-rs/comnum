//! Closed forms of the equalisers of a symbol-spaced channel: the floor of the linear
//! MMSE equaliser, the output SNR of the MMSE decision-feedback equaliser with correct
//! decisions, and the matched-filter bound, for infinitely long filters (Proakis &
//! Salehi, *Digital Communications*, 5th ed., Secs. 9.4-2 and 9.5-2; the DFE result is
//! Salz 1973). The textbook is quoted, not in `references/`.
//!
//! # Model
//!
//! The channel $r_{k} = \sum_{l} h_{l} s_{k-l} + n_{k}$ has unit-variance symbols and
//! white noise of variance $\sigma^{2}$ (`noise_var`, the variance of the complex
//! sample); its folded spectrum is $X(\omega) = |H(e^{j\omega})|^{2}$, $H$ the transfer
//! function of the taps. Then
//!
//! $$
//! J_{\min} = \frac{1}{2\pi}\int_{-\pi}^{\pi} \frac{\sigma^{2}}{X(\omega) + \sigma^{2}}\,
//! d\omega, \qquad
//! \mathrm{SNR}_{\mathrm{DFE}} = \exp\left[\frac{1}{2\pi}\int_{-\pi}^{\pi}
//! \ln\left(1 + \frac{X(\omega)}{\sigma^{2}}\right) d\omega\right] - 1, \qquad
//! \mathrm{SNR}_{\mathrm{MFB}} = \frac{\sum_{l} |h_{l}|^{2}}{\sigma^{2}}:
//! $$
//!
//! $J_{\min}$ is the mean square error of the infinite linear MMSE equaliser and
//! $1/J_{\min} - 1$ its output SNR after the removal of its bias; the DFE expression is the geometric
//! mean of $1 + X/\sigma^{2}$ over the band, the unbiased SNR of the infinite MMSE-DFE
//! (Cioffi, Dudevoir, Eyuboglu & Forney 1995, eq. (4)), which reaches the matched-filter
//! bound only when the channel has one tap. The integrals are taken on `n_fft` points of
//! the unit circle.
//!
//! # Examples
//! ```
//! use comnum_dsp::equalizer::theory::{dfe_mmse_snr, linear_mmse_mse, matched_filter_bound_snr};
//! use comnum_math::C64;
//! // one tap: every equaliser is the matched filter, J = σ²/(1 + σ²)
//! let h = [C64::new(1.0, 0.0)];
//! let j = linear_mmse_mse(&h, 0.1, 1024)?;
//! assert!((j - 0.1 / 1.1).abs() < 1e-12);
//! assert!((dfe_mmse_snr(&h, 0.1, 1024)? - 10.0).abs() < 1e-9);
//! assert_eq!(matched_filter_bound_snr(&h, 0.1)?, 10.0);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008,
//!    Secs. 9.4-2 (the infinite-length MSE equaliser) and 9.5-2 (the MSE DFE).
//! 2. J. Salz, "Optimum mean-square decision feedback equalization," *Bell Syst. Tech. J.*
//!    52(8), 1341–1373, 1973.
//! 3. J. M. Cioffi, G. P. Dudevoir, M. V. Eyuboglu, G. D. Forney, "MMSE decision-feedback
//!    equalizers and coding. Part I: Equalization results," *IEEE Trans. Commun.* 43(10),
//!    2582–2594, 1995.

use comnum_core::{Result, ensure};
use comnum_math::C64;
use comnum_math::fft;

/// The folded spectrum $X(\omega_{k}) = |H(e^{j\omega_{k}})|^{2}$ of `taps` on `n_fft`
/// points.
fn folded_spectrum(taps: &[C64], noise_var: f64, n_fft: usize) -> Result<Vec<f64>> {
    ensure(!taps.is_empty() && n_fft >= taps.len(), || {
        format!("{} taps need at least as many FFT points, got {n_fft}", taps.len())
    })?;
    ensure(noise_var > 0.0 && noise_var.is_finite(), || format!("the noise variance {noise_var} must be positive"))?;
    let mut h = taps.to_vec();
    h.resize(n_fft, C64::new(0.0, 0.0));
    Ok(fft::fft(&h).iter().map(C64::norm_sqr).collect())
}

/// The minimum mean square error $J_{\min}$ of the infinite linear MMSE equaliser of
/// `taps` under the noise variance `noise_var` (see the module); its unbiased output SNR
/// is $1/J_{\min} - 1$.
///
/// # Errors
/// [`comnum_core::Error::Config`] for no tap, fewer FFT points than taps, or a
/// non-positive noise variance.
pub fn linear_mmse_mse(taps: &[C64], noise_var: f64, n_fft: usize) -> Result<f64> {
    let x = folded_spectrum(taps, noise_var, n_fft)?;
    Ok(x.iter().map(|v| noise_var / (v + noise_var)).sum::<f64>() / n_fft as f64)
}

/// The unbiased output SNR of the infinite MMSE decision-feedback equaliser of `taps`
/// with correct decisions (see the module), linear.
///
/// # Errors
/// As [`linear_mmse_mse`].
pub fn dfe_mmse_snr(taps: &[C64], noise_var: f64, n_fft: usize) -> Result<f64> {
    let x = folded_spectrum(taps, noise_var, n_fft)?;
    Ok((x.iter().map(|v| (1.0 + v / noise_var).ln()).sum::<f64>() / n_fft as f64).exp() - 1.0)
}

/// The matched-filter bound $\sum_{l} |h_{l}|^{2} / \sigma^{2}$ on the SNR of any
/// detector of one symbol, linear.
///
/// # Errors
/// [`comnum_core::Error::Config`] for no tap or a non-positive noise variance.
pub fn matched_filter_bound_snr(taps: &[C64], noise_var: f64) -> Result<f64> {
    ensure(!taps.is_empty(), || "the channel needs a tap".into())?;
    ensure(noise_var > 0.0 && noise_var.is_finite(), || format!("the noise variance {noise_var} must be positive"))?;
    Ok(taps.iter().map(C64::norm_sqr).sum::<f64>() / noise_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equalizer::dfe::Dfe;
    use crate::equalizer::ls_equalizer;
    use comnum_core::rng::Rng;

    #[test]
    fn the_bounds_are_ordered_and_the_null_channel_shows_the_gap() {
        // 1 + z⁻¹: X(ω) = 2 + 2 cos ω, a null at Nyquist. The linear equaliser is
        // crippled, the DFE keeps most of the matched-filter bound
        let h = [C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
        let s2 = 0.05;
        let j = linear_mmse_mse(&h, s2, 4096).unwrap();
        let le = 1.0 / j - 1.0;
        let dfe = dfe_mmse_snr(&h, s2, 4096).unwrap();
        let mfb = matched_filter_bound_snr(&h, s2).unwrap();
        assert!(le < dfe && dfe < mfb, "{le} {dfe} {mfb}");
        assert!((mfb - 40.0).abs() < 1e-12);
        // the DFE of 1 + z⁻¹: the geometric mean of 1 + (2 + 2cos ω)/σ² is 1/σ² times
        // the geometric mean of σ² + 2 + 2 cos ω = ((1 + √(1 - r²))/2)… checked against
        // the closed form (a + b cos ω) → ((a + √(a² - b²))/2): a = σ² + 2, b = 2
        let a = s2 + 2.0;
        let exact = (a + (a * a - 4.0).sqrt()) / 2.0 / s2 - 1.0;
        assert!((dfe - exact).abs() < 1e-6 * exact, "{dfe} against {exact}");
        // and the linear floor: (1/2π) ∫ dω/(a + b cos ω) = 1/√(a² - b²)
        let exact_j = s2 / (a * a - 4.0).sqrt();
        assert!((j - exact_j).abs() < 1e-6 * exact_j, "{j} against {exact_j}");
        // 24 against 8: the DFE gains 4.8 dB on the null channel
        assert!(10.0 * (dfe / le).log10() > 4.0);
        assert!(linear_mmse_mse(&h, s2, 1).is_err());
        assert!(dfe_mmse_snr(&h, 0.0, 64).is_err());
        assert!(matched_filter_bound_snr(&[], 1.0).is_err());
    }

    #[test]
    fn the_equalisers_approach_their_infinite_length_bounds() {
        // 2-PAM over 1 + 0.6 z⁻¹ at σ² = 0.04: the least-squares linear equaliser of 41
        // taps on a long block sits at J_min, the adapted DFE near its SNR bound
        let pts = [C64::new(-1.0, 0.0), C64::new(1.0, 0.0)];
        let h = [C64::new(1.0, 0.0), C64::new(0.6, 0.0)];
        let (s2, n): (f64, usize) = (0.04, 20_000);
        let mut rng = Rng::new(8);
        let s: Vec<C64> = (0..n).map(|_| pts[rng.below(2)]).collect();
        let r: Vec<C64> = (0..n)
            .map(|k| {
                h[0] * s[k]
                    + if k > 0 { h[1] * s[k - 1] } else { C64::new(0.0, 0.0) }
                    + C64::new(s2.sqrt() * rng.gaussian(), 0.0)
            })
            .collect();
        let j = linear_mmse_mse(&h, s2, 4096).unwrap();
        let (_, y) = ls_equalizer(&r, 1, &s, 20, 0).unwrap();
        let mse = y.iter().zip(&s).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / n as f64;
        assert!((mse / j - 1.0).abs() < 0.05, "{mse} against {j}");
        // the DFE: unbiased SNR of its output on the second half of the block
        let dfe = Dfe { mu: 2e-3, training: n, ..Dfe::new(21, 2) };
        let out = dfe.run(&r, Some(&s), &pts).unwrap();
        let half = n / 2;
        let (ys, ss) = (&out.symbols[half..], &s[half..]);
        let g: C64 =
            ys.iter().zip(ss).map(|(y, s)| y * s.conj()).sum::<C64>() / ss.iter().map(C64::norm_sqr).sum::<f64>();
        let err = ys.iter().zip(ss).map(|(y, s)| (y - g * s).norm_sqr()).sum::<f64>() / half as f64;
        let snr = g.norm_sqr() / err;
        let bound = dfe_mmse_snr(&h, s2, 4096).unwrap();
        let gap_db = 10.0 * (bound / snr).log10();
        assert!(gap_db > -0.3 && gap_db < 1.0, "{snr} against the bound {bound}: {gap_db} dB");
        assert!(snr > 1.0 / j - 1.0);
    }
}
