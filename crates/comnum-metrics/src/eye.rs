//! Level statistics of a sampled intensity-modulated signal: the Q factor of every eye
//! between two adjacent levels, from the means and standard deviations of the samples
//! of each level (Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., 2002,
//! eq. (4.5.11)):
//!
//! $$
//! Q_{i} = \frac{\mu_{i+1} - \mu_{i}}{\sigma_{i+1} + \sigma_{i}}, \qquad i = 0, \dots, M - 2,
//! $$
//!
//! the bit error rate of the eye being $\frac{1}{2}\operatorname{erfc}(Q_{i}/\sqrt{2})$
//! ([`comnum_math::qfunc`]) under Gaussian noise. For on–off keying there is one eye; the
//! measured Q of a simulated receiver is compared with the closed forms of
//! `comnum_hardware::optical::direct_detection`.
//!
//! Without noise the eye is a worst case over the patterns: its opening is the gap
//! between the lowest sample of the upper level and the highest of the lower one
//! (Winters & Gitlin 1990, eq. (7)),
//!
//! $$
//! \mathrm{eye}_{i} = \min_{k:\, l_{k} = i + 1} y_{k} - \max_{k:\, l_{k} = i} y_{k},
//! $$
//!
//! negative when the eye is closed; the power penalty of an intersymbol interference in
//! direct detection is $-10 \log_{10}(\mathrm{eye}/Y)$ with $Y$ the opening without it
//! (eq. (8); $-20 \log_{10}$ for a field).
//!
//! # References
//! 1. G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002, Sec. 4.5.1.
//! 2. J. H. Winters, R. D. Gitlin, "Electrical signal processing techniques in
//!    long-haul, fiber-optic systems," *Proc. IEEE ICC*, 397–403, 1990.

use comnum_core::{Result, ensure, ensure_shape, same_len};

/// The mean and standard deviation of the samples of every level (`levels[k]` the index
/// of the level transmitted at sample `k`, below `m`); a level without a sample has
/// `NaN`.
///
/// # Errors
/// [`comnum_core::Error::Shape`] for sequences of different lengths or a level index at
/// or above `m`; [`comnum_core::Error::Config`] for fewer than two levels.
pub fn level_statistics(samples: &[f64], levels: &[usize], m: usize) -> Result<Vec<(f64, f64)>> {
    check_levels(samples, levels, m)?;
    let mut sum = vec![0.0; m];
    let mut sum2 = vec![0.0; m];
    let mut count = vec![0usize; m];
    for (&s, &l) in samples.iter().zip(levels) {
        sum[l] += s;
        sum2[l] += s * s;
        count[l] += 1;
    }
    Ok((0..m)
        .map(|l| {
            if count[l] == 0 {
                (f64::NAN, f64::NAN)
            } else {
                let mean = sum[l] / count[l] as f64;
                (mean, (sum2[l] / count[l] as f64 - mean * mean).max(0.0).sqrt())
            }
        })
        .collect())
}

fn check_levels(samples: &[f64], levels: &[usize], m: usize) -> Result<()> {
    ensure(m >= 2, || format!("{m} levels have no eye"))?;
    same_len(samples.len(), levels.len(), "samples and levels")?;
    ensure_shape(levels.iter().all(|&l| l < m), || format!("a level index is not below {m}"))
}

/// The worst-case opening of the $M - 1$ eyes between adjacent levels (see the module):
/// the lowest sample of the upper level minus the highest of the lower one, negative
/// for a closed eye, `NaN` when a level has no sample.
///
/// # Errors
/// As [`level_statistics`].
///
/// # Examples
/// ```
/// use comnum_metrics::eye::eye_openings;
/// let samples = [0.1, 0.3, 0.8, 1.0];
/// assert_eq!(eye_openings(&samples, &[0, 0, 1, 1], 2)?, vec![0.5]);
/// assert!(eye_openings(&samples, &[0, 1, 0, 1], 2)?[0] < 0.0);  // closed
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn eye_openings(samples: &[f64], levels: &[usize], m: usize) -> Result<Vec<f64>> {
    check_levels(samples, levels, m)?;
    let mut lo = vec![f64::INFINITY; m];
    let mut hi = vec![f64::NEG_INFINITY; m];
    for (&s, &l) in samples.iter().zip(levels) {
        lo[l] = lo[l].min(s);
        hi[l] = hi[l].max(s);
    }
    Ok((1..m).map(|l| if lo[l].is_finite() && hi[l - 1].is_finite() { lo[l] - hi[l - 1] } else { f64::NAN }).collect())
}

/// The Q factor of the $M - 1$ eyes between adjacent levels (see the module).
///
/// # Errors
/// As [`level_statistics`].
///
/// # Examples
/// ```
/// use comnum_metrics::eye::level_q_factors;
/// // two levels at 0 and 1 with deviations 0.1 and 0.1: Q = 5
/// let samples = [0.1, -0.1, 0.9, 1.1, 0.1, -0.1, 1.1, 0.9];
/// let levels = [0, 0, 1, 1, 0, 0, 1, 1];
/// let q = level_q_factors(&samples, &levels, 2)?;
/// assert!((q[0] - 5.0).abs() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn level_q_factors(samples: &[f64], levels: &[usize], m: usize) -> Result<Vec<f64>> {
    let stats = level_statistics(samples, levels, m)?;
    Ok(stats.windows(2).map(|w| (w[1].0 - w[0].0) / (w[1].1 + w[0].1)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_q_of_four_levels_and_the_errors_of_the_input() {
        let samples = [0.0, 0.2, 1.0, 1.2, 2.0, 2.2, 3.0, 3.2];
        let levels = [0, 0, 1, 1, 2, 2, 3, 3];
        let s = level_statistics(&samples, &levels, 4).unwrap();
        assert!((s[1].0 - 1.1).abs() < 1e-12 && (s[1].1 - 0.1).abs() < 1e-12);
        let q = level_q_factors(&samples, &levels, 4).unwrap();
        assert_eq!(q.len(), 3);
        assert!(q.iter().all(|v| (v - 5.0).abs() < 1e-12));
        // a missing level is NaN
        let partial = level_statistics(&samples, &levels, 5).unwrap();
        assert!(partial[4].0.is_nan());
        assert!(level_q_factors(&samples, &levels[1..], 4).is_err());
        assert!(level_q_factors(&samples, &levels, 3).is_err());
        assert!(level_q_factors(&samples, &levels, 1).is_err());
    }

    #[test]
    fn the_worst_case_eye_is_the_gap_between_adjacent_levels() {
        let samples = [0.0, 0.2, 1.0, 1.2, 2.0, 2.2, 3.0, 3.2];
        let levels = [0, 0, 1, 1, 2, 2, 3, 3];
        let eyes = eye_openings(&samples, &levels, 4).unwrap();
        assert!(eyes.iter().all(|e| (e - 0.8).abs() < 1e-12));
        assert!(eye_openings(&samples, &levels, 5).unwrap()[3].is_nan());
        assert!(eye_openings(&samples, &[0, 1, 0, 1, 0, 1, 0, 1], 2).unwrap()[0] < 0.0);
        assert!(eye_openings(&samples, &levels, 3).is_err());
    }
}
