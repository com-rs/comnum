//! The theory of DS-CDMA: the exact error rates of the conventional receiver and of
//! the decorrelator for given signatures, the near–far resistance, and the
//! large-system limits for random signatures.
//!
//! # Given signatures
//!
//! With $G = H^{H} H$ the Gram matrix of $H = S A$ (see [`super`]), BPSK symbols and
//! the noise variance $\sigma^{2}$ per complex chip, the matched filter of user $k$
//! decides on $\operatorname{Re}(h_{k}^{H} y)$, whose noise has the variance $\sigma^{2}
//! G_{kk} / 2$; the interference of the other users is a sum of $\pm
//! \operatorname{Re} G_{kj}$, so that ([`ber_matched_filter`])
//!
//! $$
//! P_{k} = 2^{1-K} \sum_{b \in \{\pm 1\}^{K-1}} Q\\!\left( \frac{G_{kk} + \sum_{j \ne k}
//! b_{j} \operatorname{Re} G_{kj}}{\sigma \sqrt{G_{kk} / 2}} \right),
//! $$
//!
//! which does not go to zero with the noise when the interference closes the eye
//! ($\sum_{j \ne k} \lvert \operatorname{Re} G_{kj} \rvert > G_{kk}$): the near–far
//! problem. The decorrelator removes the interference and enhances the noise
//! ([`ber_decorrelator`]),
//!
//! $$
//! P_{k} = Q\\!\left( \sqrt{\frac{2}{\sigma^{2} (G^{-1})_{kk}}} \right),
//! $$
//!
//! whatever the amplitudes of the others: against a single user of the same energy it
//! loses the factor $\bar\eta_{k} = 1 / (G_{kk} (G^{-1})_{kk})$, its asymptotic
//! efficiency, which is also the near–far resistance of the optimum detector
//! ([`near_far_resistance`]; Lupas and Verdú 1989).
//!
//! # Random signatures, large systems
//!
//! For independent random chips and $K, N \to \infty$ at the load $\beta = K/N$, the
//! SINR of every user at the output of a linear receiver converges to a deterministic
//! limit. With users of equal power and $\mathrm{SNR} = A^{2}/\sigma^{2}$
//! ([`sinr_large_system`]; Verdú and Shamai 1999, eqs. (10)–(12); Tse and Hanly 1999,
//! Proposition 3.3 and eq. (9)):
//!
//! $$
//! \mathrm{SINR}_{\mathrm{MF}} = \frac{\mathrm{SNR}}{1 + \beta\\,\mathrm{SNR}}, \qquad
//! \mathrm{SINR}_{\mathrm{dec}} = \mathrm{SNR} (1 - \beta), \qquad
//! \mathrm{SINR}_{\mathrm{MMSE}} = \mathrm{SNR} - \tfrac{1}{4} \mathcal{F}(\mathrm{SNR},
//! \beta),
//! $$
//!
//! $$
//! \mathcal{F}(x, z) = \left( \sqrt{x (1 + \sqrt{z})^{2} + 1} - \sqrt{x (1 -
//! \sqrt{z})^{2} + 1} \right)^{2}.
//! $$
//!
//! With unequal powers $P_{i}$, the SINR $\gamma$ of the MMSE receiver of a user of
//! power $P$ is the unique solution of ([`sinr_mmse_large_system`]; Tse and Hanly 1999,
//! Theorem 3.1 and eq. (6))
//!
//! $$
//! \gamma = \frac{P}{\sigma^{2} + \frac{1}{N} \sum_{i} I(P_{i}, P, \gamma)}, \qquad
//! I(P_{i}, P, \gamma) = \frac{P P_{i}}{P + P_{i} \gamma},
//! $$
//!
//! each interferer counting for an effective interference below its power, bounded by
//! $P/\gamma$ however strong it is.
//!
//! The spectral efficiency of real-valued chips with single-user decoding after a
//! linear receiver is $\frac{\beta}{2} \log_{2}(1 + \mathrm{SINR})$ bits per chip
//! ([`spectral_efficiency`]), and that of the optimum joint decoder
//! ([`spectral_efficiency_optimal`]; Verdú and Shamai 1999, eq. (9))
//!
//! $$
//! \mathsf{C}^{\mathrm{opt}} = \frac{\beta}{2} \log_{2}\\!\left(1 + \mathrm{SNR} -
//! \tfrac{1}{4} \mathcal{F}\right) + \frac{1}{2} \log_{2}\\!\left(1 + \mathrm{SNR} \beta -
//! \tfrac{1}{4} \mathcal{F}\right) - \frac{\log_{2} e}{8\\,\mathrm{SNR}} \mathcal{F}.
//! $$
//!
//! At a given $E_{b}/N_{0}$ the SNR follows from the efficiency itself, $\mathrm{SNR} =
//! \frac{2 \mathsf{C}}{\beta} \frac{E_{b}}{N_{0}}$ ([`spectral_efficiency_at_ebn0`], the
//! curves of their Fig. 1).

use comnum_core::{Result, ensure};
use comnum_math::linalg::{gram, inverse_hpd};
use comnum_math::{C64, bisect, qfunc};
use comnum_mimo::detection::Criterion;
use ndarray::ArrayView2;

/// The most interferers of the exact sum of [`ber_matched_filter`] ($2^{20}$ terms).
const MAX_INTERFERERS: usize = 20;

/// `Ok` for a user of `h` and a positive noise variance.
fn check(h: &ArrayView2<'_, C64>, user: usize, noise_var: f64) -> Result<()> {
    ensure(user < h.ncols(), || format!("user {user} of {}", h.ncols()))?;
    ensure(noise_var > 0.0 && noise_var.is_finite(), || format!("the noise variance must be positive, got {noise_var}"))
}

/// The exact bit error rate of user `user` with BPSK and the conventional receiver
/// (single-user matched filter), for the signatures `h` $= S A$ and the noise variance
/// `noise_var` per chip (see the module): the mean over the $2^{K-1}$ patterns of the
/// interfering bits.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a user that does not exist, a noise variance that
/// is not positive, or more than 21 users.
///
/// # Examples
/// ```
/// use comnum_math::sequences::gold_codes;
/// use comnum_waveform::dsss::{Dsss, theory::ber_matched_filter};
/// let one = Dsss::new(&gold_codes(5)?[..1])?;
/// // one user: Q(√(2/σ²))
/// let p = ber_matched_filter(&one.signatures(), 0, 0.25)?;
/// assert!((p - comnum_math::qfunc(8f64.sqrt())).abs() < 1e-15);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ber_matched_filter(h: &ArrayView2<'_, C64>, user: usize, noise_var: f64) -> Result<f64> {
    check(h, user, noise_var)?;
    ensure(h.ncols() <= MAX_INTERFERERS + 1, || {
        format!("the exact sum covers {} users at most, got {}", MAX_INTERFERERS + 1, h.ncols())
    })?;
    let g = gram(h);
    let own = g[[user, user]].re;
    let others: Vec<f64> = (0..h.ncols()).filter(|&j| j != user).map(|j| g[[user, j]].re).collect();
    let sigma = (noise_var * own / 2.0).sqrt();
    let patterns = 1u32 << others.len();
    let sum: f64 = (0..patterns)
        .map(|b| {
            let interference: f64 =
                others.iter().enumerate().map(|(j, r)| if (b >> j) & 1 == 0 { *r } else { -r }).sum();
            qfunc((own + interference) / sigma)
        })
        .sum();
    Ok(sum / f64::from(patterns))
}

/// The diagonal entry $(G^{-1})_{kk}$ of the inverse Gram matrix of `h`.
fn inverse_gram(h: &ArrayView2<'_, C64>, user: usize) -> Result<f64> {
    ensure(user < h.ncols(), || format!("user {user} of {}", h.ncols()))?;
    Ok(inverse_hpd(&gram(h).view())?[[user, user]].re)
}

/// The exact bit error rate of user `user` with BPSK and the decorrelator, $Q(\sqrt{2
/// / (\sigma^{2} (G^{-1})_{kk})})$ (see the module).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a user that does not exist or a noise variance
/// that is not positive; [`comnum_core::Error::Numerical`] for linearly dependent
/// signatures.
pub fn ber_decorrelator(h: &ArrayView2<'_, C64>, user: usize, noise_var: f64) -> Result<f64> {
    check(h, user, noise_var)?;
    Ok(qfunc((2.0 / (noise_var * inverse_gram(h, user)?)).sqrt()))
}

/// The near–far resistance $\bar\eta_{k} = 1 / (G_{kk} (G^{-1})_{kk}) \in (0, 1\]$ of
/// user `user`: the asymptotic efficiency of the decorrelator, and of the optimum
/// detector against interferers of the worst amplitudes (see the module); 1 for a
/// signature orthogonal to the others.
///
/// # Errors
/// As [`ber_decorrelator`].
pub fn near_far_resistance(h: &ArrayView2<'_, C64>, user: usize) -> Result<f64> {
    let inv = inverse_gram(h, user)?;
    Ok(1.0 / (h.column(user).iter().map(C64::norm_sqr).sum::<f64>() * inv))
}

/// $\mathcal{F}(x, z)$ of Verdú and Shamai 1999 (see the module).
fn f_function(x: f64, z: f64) -> f64 {
    let root = |sign: f64| (x * (1.0 + sign * z.sqrt()).powi(2) + 1.0).sqrt();
    (root(1.0) - root(-1.0)).powi(2)
}

/// `Ok` for a positive load and a positive SNR.
fn check_load(load: f64, snr: f64) -> Result<()> {
    ensure(load > 0.0 && load.is_finite(), || format!("the load K/N must be positive, got {load}"))?;
    ensure(snr > 0.0 && snr.is_finite(), || format!("the SNR must be positive, got {snr}"))
}

/// The large-system SINR of a linear receiver for random signatures, users of equal
/// power, the load `load` $= K/N$ and the linear `snr` $= A^{2}/\sigma^{2}$ (see the
/// module): the conventional receiver ([`Criterion::Mr`]), the decorrelator
/// ([`Criterion::Zf`], 0 for a load of 1 or more) or the MMSE receiver.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a load or an SNR that is not positive.
///
/// # Examples
/// ```
/// use comnum_mimo::detection::Criterion;
/// use comnum_waveform::dsss::theory::sinr_large_system;
/// // N = 128, K = 80, 20 dB (Tse and Hanly 1999, Fig. 2): 15.9 dB
/// let sinr = sinr_large_system(Criterion::Mmse, 80.0 / 128.0, 100.0)?;
/// assert!((10.0 * sinr.log10() - 15.92).abs() < 0.01);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn sinr_large_system(criterion: Criterion, load: f64, snr: f64) -> Result<f64> {
    check_load(load, snr)?;
    Ok(match criterion {
        Criterion::Mr => snr / (1.0 + load * snr),
        Criterion::Zf => snr * (1.0 - load).max(0.0),
        Criterion::Mmse => snr - f_function(snr, load) / 4.0,
    })
}

/// The large-system SINR of the MMSE receiver of a user received with the power
/// `power` among interferers of the powers `interferers`, random signatures of `n`
/// chips and the noise variance `noise_var`: the fixed point of eq. (6) of Tse and
/// Hanly 1999 (see the module), by bisection.
///
/// # Errors
/// [`comnum_core::Error::Config`] unless the powers, `n` and the noise variance are
/// positive.
pub fn sinr_mmse_large_system(power: f64, interferers: &[f64], n: usize, noise_var: f64) -> Result<f64> {
    ensure(n > 0, || "the spreading factor must be positive".to_string())?;
    ensure(power > 0.0 && noise_var > 0.0 && interferers.iter().all(|p| *p > 0.0), || {
        "the powers and the noise variance must be positive".to_string()
    })?;
    // γ (σ² + Σ I(P_i, P, γ) / N) / P increases with γ and is 1 at the solution
    let below = |x: f64| {
        let effective: f64 = interferers.iter().map(|p| power * p / (power + p * x)).sum();
        x * (noise_var + effective / n as f64) < power
    };
    Ok(bisect(below, 0.0, power / noise_var, 200))
}

/// The large-system spectral efficiency (bits per chip, real-valued chips) of a linear
/// receiver followed by single-user decoders, $\frac{\beta}{2} \log_{2}(1 +
/// \mathrm{SINR})$ with the SINR of [`sinr_large_system`] (Verdú and Shamai 1999, eqs.
/// (10)–(12)).
///
/// # Errors
/// As [`sinr_large_system`].
pub fn spectral_efficiency(criterion: Criterion, load: f64, snr: f64) -> Result<f64> {
    Ok(0.5 * load * (1.0 + sinr_large_system(criterion, load, snr)?).log2())
}

/// The large-system spectral efficiency (bits per chip) of the optimum joint decoder
/// for random signatures, eq. (9) of Verdú and Shamai 1999 (see the module).
///
/// # Errors
/// As [`sinr_large_system`].
pub fn spectral_efficiency_optimal(load: f64, snr: f64) -> Result<f64> {
    check_load(load, snr)?;
    let f = f_function(snr, load);
    Ok(0.5 * load * (1.0 + snr - f / 4.0).log2() + 0.5 * (1.0 + snr * load - f / 4.0).log2()
        - std::f64::consts::LOG2_E * f / (8.0 * snr))
}

/// The spectral efficiency $\mathsf{C}$ at the linear `ebn0` $= E_{b}/N_{0}$ of a
/// receiver whose efficiency at a given SNR is `efficiency(snr)` (a closure on
/// [`spectral_efficiency`] or [`spectral_efficiency_optimal`]), at the load `load`: the
/// positive solution of $\mathsf{C} = \mathtt{efficiency}(2 \mathsf{C} E_{b} / (\beta
/// N_{0}))$, 0 when there is none ($E_{b}/N_{0} \le \ln 2$, or a decorrelator at a load
/// of 1 or more).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a load or an $E_{b}/N_{0}$ that is not positive,
/// and the errors of `efficiency`.
///
/// # Examples
/// ```
/// use comnum_mimo::detection::Criterion;
/// use comnum_waveform::dsss::theory::{spectral_efficiency, spectral_efficiency_at_ebn0};
/// // the conventional receiver at a large load: log2(e)/2 - N0/(2 Eb)
/// let c = spectral_efficiency_at_ebn0(|snr| spectral_efficiency(Criterion::Mr, 500.0, snr), 500.0, 10.0)?;
/// assert!((c - (std::f64::consts::LOG2_E / 2.0 - 0.05)).abs() < 2e-3);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn spectral_efficiency_at_ebn0(efficiency: impl Fn(f64) -> Result<f64>, load: f64, ebn0: f64) -> Result<f64> {
    check_load(load, ebn0)?;
    // C(SNR) - β SNR N0 / (2 Eb) is positive below the solution, negative above
    let excess = |snr: f64| Ok::<f64, comnum_core::Error>(efficiency(snr)? - load * snr / (2.0 * ebn0));
    let floor = 1e-9;
    if excess(floor)? <= 0.0 {
        return Ok(0.0);
    }
    let mut ceiling = 1.0;
    while excess(ceiling)? > 0.0 {
        ceiling *= 2.0;
    }
    let mut failed = None;
    let snr = bisect(|snr| excess(snr).map_err(|e| failed = Some(e)).is_ok_and(|v| v > 0.0), floor, ceiling, 200);
    failed.map_or_else(|| efficiency(snr), Err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsss::Dsss;
    use comnum_math::sequences::{gold_codes, periodic_correlation, walsh_codes};

    #[test]
    fn orthogonal_signatures_give_the_single_user_error_rate() {
        let cdma = Dsss::new(&walsh_codes(8).unwrap()).unwrap();
        let h = cdma.signatures();
        let single = qfunc((2.0f64 / 0.2).sqrt());
        for k in 0..8 {
            assert!((ber_matched_filter(&h, k, 0.2).unwrap() - single).abs() < 1e-15);
            assert!((ber_decorrelator(&h, k, 0.2).unwrap() - single).abs() < 1e-12);
            assert!((near_far_resistance(&h, k).unwrap() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn the_matched_filter_has_a_floor_and_the_decorrelator_has_none() {
        // two users at the correlation -9/31 (the decimated sequence and one sum; the
        // sums are at -1/31 of each other when aligned), the second 12 dB above: its interference
        // closes the eye of the first (4 × 9/31 > 1), half of the patterns err
        let codes = gold_codes(5).unwrap();
        let other = (2..33).find(|&j| periodic_correlation(&codes[1], &codes[j]).unwrap()[0] == -9).unwrap();
        let pair = [codes[1].clone(), codes[other].clone()];
        let cdma = Dsss::new(&pair).unwrap().with_amplitudes(&[1.0, 4.0]).unwrap();
        let h = cdma.signatures();
        let rho = gram(&h)[[0, 1]].re / 4.0;
        assert!((rho + 9.0 / 31.0).abs() < 1e-12);
        let floor = ber_matched_filter(&h, 0, 1e-6).unwrap();
        assert!((floor - 0.5).abs() < 1e-12, "{floor}");
        // the decorrelator: the loss 1 - ρ² whatever the amplitude of the other
        let eta = near_far_resistance(&h, 0).unwrap();
        assert!((eta - (1.0 - rho * rho)).abs() < 1e-12);
        let p = ber_decorrelator(&h, 0, 0.1).unwrap();
        assert!((p - qfunc((2.0 * eta / 0.1).sqrt())).abs() < 1e-15);
        assert!(ber_matched_filter(&h, 2, 0.1).is_err() && ber_decorrelator(&h, 0, 0.0).is_err());
    }

    /// Tse and Hanly 1999, eq. (9), and Verdú and Shamai 1999, eq. (12), are the same
    /// limit, and the fixed point of eq. (6) with equal powers gives it too.
    #[test]
    fn the_mmse_limits_of_tse_hanly_and_verdu_shamai_agree() {
        for (alpha, snr) in [(0.25, 100.0), (0.625, 100.0), (1.0, 10.0), (1.5, 100.0), (2.0, 3.0)] {
            let a: f64 = (1.0 - alpha) * snr / 2.0;
            let eq9 = a - 0.5 + (a * a + (1.0 + alpha) * snr / 2.0 + 0.25).sqrt();
            let limit = sinr_large_system(Criterion::Mmse, alpha, snr).unwrap();
            assert!((limit - eq9).abs() < 1e-9 * eq9, "{alpha} {snr}: {limit} {eq9}");
            let n = 4000;
            let interferers = vec![snr; (alpha * n as f64) as usize];
            let fixed = sinr_mmse_large_system(snr, &interferers, n, 1.0).unwrap();
            assert!((fixed - eq9).abs() < 1e-9 * eq9);
        }
        // eq. (10) of Tse and Hanly: 1/(α - 1) without noise, α > 1
        let limit = sinr_large_system(Criterion::Mmse, 1.5, 1e9).unwrap();
        assert!((limit - 2.0).abs() < 1e-4, "{limit}");
        // a strong interferer costs the MMSE receiver at most P/γ
        let weak = sinr_mmse_large_system(1.0, &[1.0; 16], 32, 0.01).unwrap();
        let strong = sinr_mmse_large_system(1.0, &[1e6; 16], 32, 0.01).unwrap();
        assert!(strong < weak && (strong - (1.0 - 0.5) / 0.01).abs() < 1e-2 * strong, "{weak} {strong}");
    }

    #[test]
    fn the_receivers_are_ordered_and_meet_at_a_vanishing_load() {
        for (load, snr) in [(0.1, 10.0), (0.5, 100.0), (0.9, 31.6), (1.5, 10.0)] {
            let mr = spectral_efficiency(Criterion::Mr, load, snr).unwrap();
            let zf = spectral_efficiency(Criterion::Zf, load, snr).unwrap();
            let mmse = spectral_efficiency(Criterion::Mmse, load, snr).unwrap();
            let opt = spectral_efficiency_optimal(load, snr).unwrap();
            assert!(mmse > mr && mmse > zf && opt > mmse, "{load} {snr}: {mr} {zf} {mmse} {opt}");
            // below the channel without spreading, ½ log2(1 + β SNR)
            assert!(opt < 0.5 * (1.0 + load * snr).log2());
        }
        let snr = 20.0;
        for criterion in [Criterion::Mr, Criterion::Zf, Criterion::Mmse] {
            assert!((sinr_large_system(criterion, 1e-9, snr).unwrap() - snr).abs() < 1e-6);
        }
        assert_eq!(sinr_large_system(Criterion::Zf, 1.2, snr).unwrap(), 0.0);
        assert!(sinr_large_system(Criterion::Mmse, 0.0, snr).is_err());
    }

    /// Verdú and Shamai 1999, Sec. I-C: at a given Eb/N0 the conventional receiver
    /// tends to log2(e)/2 - N0/(2 Eb) at a large load, and nothing is sent below
    /// Eb/N0 = ln 2.
    #[test]
    fn the_efficiency_at_a_given_ebn0_has_the_limits_of_verdu_shamai() {
        let ebn0 = 10.0;
        let at = |criterion, load| {
            spectral_efficiency_at_ebn0(|snr| spectral_efficiency(criterion, load, snr), load, ebn0).unwrap()
        };
        assert!((at(Criterion::Mr, 2000.0) - (std::f64::consts::LOG2_E / 2.0 - 0.05)).abs() < 1e-3);
        assert_eq!(at(Criterion::Zf, 1.0), 0.0);
        // the decorrelator: β C*((1 - β) Eb/N0), with C* the single-user efficiency
        // (4^C - 1)/(2C) = Eb/N0 (their eq. (11))
        let c = at(Criterion::Zf, 0.5) / 0.5;
        assert!(((4f64.powf(c) - 1.0) / (2.0 * c) - 0.5 * ebn0).abs() < 1e-9);
        let none = spectral_efficiency_at_ebn0(|snr| spectral_efficiency_optimal(0.5, snr), 0.5, 0.6).unwrap();
        assert_eq!(none, 0.0);
        // the solution is consistent: SNR = 2 C Eb / (β N0)
        let c = spectral_efficiency_at_ebn0(|snr| spectral_efficiency_optimal(0.75, snr), 0.75, ebn0).unwrap();
        assert!((spectral_efficiency_optimal(0.75, 2.0 * c * ebn0 / 0.75).unwrap() - c).abs() < 1e-9);
    }
}
