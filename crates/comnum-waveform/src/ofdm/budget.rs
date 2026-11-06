//! The SNR-loss budget of an OFDM receiver (Speth, Fechtel, Fock & Meyr 1999).
//!
//! In tracking mode every residual imperfection of the inner receiver, a noisy channel
//! estimate, a timing offset, a carrier frequency offset, fading within a symbol, is
//! a rotation of the subcarriers plus an additive noise of power $\sigma^{2}$ (relative to
//! the power of a subcarrier, $E|a_{k}|^{2} = E|H_{k}|^{2} = 1$), and its cost is the
//! loss of SNR it causes, their eq. (14),
//!
//! $$
//! \Delta\gamma = \frac{\gamma_{\mathrm{ideal}}}{\gamma_{\mathrm{real}}}, \qquad
//! \sigma_{\mathrm{res}}^{2} = \sigma_{\varepsilon}^{2} + \sigma_{\Omega}^{2} +
//! \sigma_{\mathrm{fad}}^{2} < \frac{1}{\gamma}\left(1 - \frac{1}{\Delta\gamma_{\max}}\right)
//! $$
//!
//! (eqs. (50)–(51), $\gamma$ the SNR per subcarrier): the noises add, and a loss
//! $\Delta\gamma_{\max}$ allows the total [`noise_budget`]. The noises of the
//! imperfections:
//!
//! - a channel estimate of gain $G$ (its error is the noise divided by $G$, eq. (19)):
//!   $\Delta\gamma = 1 + 1/G$, eq. (20), [`estimator_loss_db`] and [`estimator_gain`];
//! - a timing offset of $n_{\varepsilon}$ samples beyond the part of the guard interval
//!   that the echoes leave free, eqs. (30)–(31), [`timing_noise_power`];
//! - a carrier frequency offset $\Delta f T_{u}$ (in subcarrier spacings), eq. (42),
//!   [`cfo_noise_power`], $\frac{\pi^{2}}{3}(\Delta f T_{u})^{2}$, and the largest one
//!   for a loss, eq. (43), [`max_cfo`];
//! - fading of maximum Doppler $f_{D}$ within a symbol (Jakes), eq. (47),
//!   [`doppler_noise_power`], $\frac{\pi^{2}}{6}(f_{D} T_{u})^{2}$, and the largest
//!   Doppler for a loss, [`max_doppler`] (the curves of their Fig. 7).
//!
//! The approximations are those of the paper: one imperfection at a time, small errors
//! (the attenuation of the wanted subcarrier neglected), every interference Gaussian;
//! [`super::sync::ici_power`] gives the exact interference of a frequency offset.
//!
//! # Examples
//! ```
//! use comnum_waveform::ofdm::budget::{estimator_gain, estimator_loss_db, max_doppler};
//! // Sec. IV-A: an estimator of gain 10 costs 0.41 dB; 0.1 dB needs a gain of 43
//! assert!((estimator_loss_db(10.0) - 0.414).abs() < 1e-3);
//! assert!((estimator_gain(0.1) - 42.9).abs() < 0.05);
//! // Fig. 7: at 20 dB of SNR, 1 dB of loss tolerates f_D T_u = 0.035
//! assert!((max_doppler(100.0, 1.0) - 0.0354).abs() < 1e-4);
//! ```
//!
//! # References
//!
//! 1. M. Speth, S. A. Fechtel, G. Fock and H. Meyr, "Optimum receiver design for wireless
//!    broad-band systems using OFDM — Part I," *IEEE Trans. Commun.*, 47(11),
//!    1668–1677, 1999.

use comnum_core::{Result, ensure};
use std::f64::consts::PI;

/// The SNR loss (dB) of a receiver that equalises with a channel estimate of gain $G$,
/// $10 \log_{10}(1 + 1/G)$ (eq. (20)).
pub fn estimator_loss_db(gain: f64) -> f64 {
    10.0 * (1.0 + 1.0 / gain).log10()
}

/// The gain $G = 1/(\Delta\gamma - 1)$ a channel estimator needs for a loss of `loss_db`
/// (dB): the inverse of [`estimator_loss_db`] (their Fig. 2).
pub fn estimator_gain(loss_db: f64) -> f64 {
    1.0 / (10f64.powf(loss_db / 10.0) - 1.0)
}

/// The largest total power $\sigma_{\mathrm{res}}^{2}$ of the residual noises (relative to
/// the power of a subcarrier) for a loss of `loss_db` (dB) at the SNR `snr_lin` (linear,
/// per subcarrier): $(1 - 1/\Delta\gamma_{\max})/\gamma$ (eqs. (32), (51)).
pub fn noise_budget(snr_lin: f64, loss_db: f64) -> f64 {
    (1.0 - 10f64.powf(-loss_db / 10.0)) / snr_lin
}

/// The power of the intersymbol and intercarrier interference of a timing offset
/// (eqs. (30)–(31)): over the taps `(delay, power)` of the channel (delays in samples,
/// powers of unit sum), $\sum_{i} p_{i}\,(2\Delta_{i}/N - (\Delta_{i}/N)^{2})$, with
/// $\Delta_{i}$ the samples of the FFT window of $N$ = `n_fft` points that fall outside
/// the symbol of tap $i$ for a window `offset` samples late (positive) or early
/// (negative) of the end of the guard interval of `guard` samples:
/// $\Delta_{i} = n_{\varepsilon} - \tau_{i}$ if $n_{\varepsilon} > \tau_{i}$,
/// $\tau_{i} - N_{g} - n_{\varepsilon}$ if $n_{\varepsilon} < \tau_{i} - N_{g}$, 0
/// otherwise (the second condition is printed $0 < n_{\varepsilon}T < -(T_{g} -
/// \tau_{i})$ in the paper, which cannot hold for an echo inside the guard interval).
///
/// # Errors
/// [`comnum_core::Error::Config`] for an empty FFT, a negative delay or power.
pub fn timing_noise_power(offset: f64, n_fft: usize, guard: usize, taps: &[(f64, f64)]) -> Result<f64> {
    ensure(n_fft > 0, || "the FFT needs at least one point".into())?;
    ensure(taps.iter().all(|&(d, p)| d >= 0.0 && p >= 0.0), || "the delays and powers must be non-negative".into())?;
    let n = n_fft as f64;
    Ok(taps
        .iter()
        .map(|&(tau, p)| {
            let excess = if offset > tau {
                offset - tau
            } else if offset < tau - guard as f64 {
                tau - guard as f64 - offset
            } else {
                0.0
            };
            let x = excess.min(n) / n;
            p * (2.0 * x - x * x)
        })
        .sum())
}

/// The power of the intercarrier interference of a small carrier frequency offset
/// `eps` (in subcarrier spacings), $\frac{\pi^{2}}{3}\varepsilon^{2}$ (eq. (42)): the
/// second-order term of [`super::sync::ici_power`] when every subcarrier is active.
pub fn cfo_noise_power(eps: f64) -> f64 {
    PI * PI / 3.0 * eps * eps
}

/// The power of the intercarrier interference of fading within a symbol, for a Jakes
/// spectrum of maximum Doppler `fd_tu` (normalised to the subcarrier spacing,
/// $f_{D} T_{u}$), $\frac{\pi^{2}}{6}(f_{D} T_{u})^{2}$ (eq. (47); taps of unit total
/// power).
pub fn doppler_noise_power(fd_tu: f64) -> f64 {
    PI * PI / 6.0 * fd_tu * fd_tu
}

/// The signal-to-interference ratio of CP-OFDM under Jakes' fading of maximum Doppler
/// `fd_tu` (normalised to the subcarrier spacing, $f_{D} T_{u}$), the prefix covering the
/// delay spread (no ISI), exact (Robertson & Kaiser 1999; Nissel, Schwarz & Rupp, *IEEE J.
/// Sel. Areas Commun.* 35(8), 2017, eq. (42)):
///
/// $$
/// \mathrm{SIR} = \frac{{}_{1}F_{2}\left(\tfrac{1}{2}; \tfrac{3}{2}, 2; -(\pi f_{D} T_{u})^{2}\right)}
/// {1 - {}_{1}F_{2}\left(\tfrac{1}{2}; \tfrac{3}{2}, 2; -(\pi f_{D} T_{u})^{2}\right)},
/// $$
///
/// the numerator the power kept on the subcarrier, the denominator the ICI, of a channel
/// of unit power. [`doppler_noise_power`] is its small-Doppler limit, $1/\mathrm{SIR}
/// \to \frac{\pi^{2}}{6}(f_{D} T_{u})^{2}$. With the prefix just covering the delay spread
/// $\tau_{\max}$, $\kappa = T_{\mathrm{cp}}/T_{u}$, $f_{D} T_{u} = f_{D} \tau_{\max}/\kappa$.
///
/// # Examples
/// ```
/// use comnum_waveform::ofdm::budget::{doppler_noise_power, jakes_ici_sir};
/// let fd_tu = 0.01;
/// assert!((1.0 / jakes_ici_sir(fd_tu) / doppler_noise_power(fd_tu) - 1.0).abs() < 1e-3);
/// ```
pub fn jakes_ici_sir(fd_tu: f64) -> f64 {
    let x = PI * fd_tu;
    let f = comnum_math::hyp1f2(0.5, 1.5, 2.0, -x * x);
    f / (1.0 - f)
}

/// The largest carrier frequency offset (in subcarrier spacings) for a loss of
/// `loss_db` (dB) at the SNR `snr_lin` (linear, per subcarrier), eq. (43):
/// [`cfo_noise_power`] equal to the [`noise_budget`].
pub fn max_cfo(snr_lin: f64, loss_db: f64) -> f64 {
    (3.0 * noise_budget(snr_lin, loss_db)).sqrt() / PI
}

/// The largest maximum Doppler $f_{D} T_{u}$ (normalised to the subcarrier spacing) for a
/// loss of `loss_db` (dB) at the SNR `snr_lin` (linear, per subcarrier): the bound of
/// Fig. 7 of Speth et al., [`doppler_noise_power`] equal to the [`noise_budget`]
/// ([`max_cfo`] times $\sqrt{2}$).
pub fn max_doppler(snr_lin: f64, loss_db: f64) -> f64 {
    (6.0 * noise_budget(snr_lin, loss_db)).sqrt() / PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::Ofdm;
    use crate::ofdm::sync::ici_power;

    #[test]
    fn the_cfo_closed_form_is_the_small_offset_limit_of_the_exact_ici() {
        // all subcarriers active, a large FFT: Moose's exact sum tends to pi^2/3 eps^2
        let ofdm = Ofdm { oversampling: 1, ..Ofdm::new(1024, 1024).unwrap() };
        for eps in [0.005, 0.01, 0.02] {
            let exact = ici_power(&ofdm, 512, eps);
            assert!((exact / cfo_noise_power(eps) - 1.0).abs() < 0.01, "eps {eps}: {exact}");
        }
    }

    #[test]
    fn the_timing_noise_vanishes_inside_the_free_guard_interval() {
        let taps = [(0.0, 0.5), (10.0, 0.5)];
        // a window up to 54 samples early with a guard of 64 and an echo at 10 is free
        for offset in [0.0, -20.0, -54.0] {
            assert_eq!(timing_noise_power(offset, 2048, 64, &taps).unwrap(), 0.0);
        }
        assert!(timing_noise_power(-55.0, 2048, 64, &taps).unwrap() > 0.0);
        assert!(timing_noise_power(1.0, 2048, 64, &taps).unwrap() > 0.0);
        // one path, late by n: 2n/N - (n/N)^2
        let n = 10.0;
        let one = timing_noise_power(n, 2048, 0, &[(0.0, 1.0)]).unwrap();
        assert!((one - (2.0 * n / 2048.0 - (n / 2048.0).powi(2))).abs() < 1e-15);
        assert!(
            timing_noise_power(1.0, 0, 0, &taps).is_err() && timing_noise_power(1.0, 8, 0, &[(-1.0, 1.0)]).is_err()
        );
    }

    #[test]
    fn the_budget_inverts_the_tolerances() {
        let (snr, loss) = (10f64.powf(1.5), 0.5);
        assert!((cfo_noise_power(max_cfo(snr, loss)) - noise_budget(snr, loss)).abs() < 1e-15);
        assert!((doppler_noise_power(max_doppler(snr, loss)) - noise_budget(snr, loss)).abs() < 1e-15);
        assert!((max_doppler(snr, loss) / max_cfo(snr, loss) - 2f64.sqrt()).abs() < 1e-14);
        assert!((estimator_loss_db(estimator_gain(0.7)) - 0.7).abs() < 1e-12);
    }
}
