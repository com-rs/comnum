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

use comnum_core::{Result, ensure, ensure_shape, same_len};

/// The mean and standard deviation of the samples of every level (`levels[k]` the index
/// of the level transmitted at sample `k`, below `m`); a level without a sample has
/// `NaN`.
///
/// # Errors
/// [`comnum_core::Error::Shape`] for sequences of different lengths or a level index at
/// or above `m`; [`comnum_core::Error::Config`] for fewer than two levels.
pub fn level_statistics(samples: &[f64], levels: &[usize], m: usize) -> Result<Vec<(f64, f64)>> {
    ensure(m >= 2, || format!("{m} levels have no eye"))?;
    same_len(samples.len(), levels.len(), "samples and levels")?;
    ensure_shape(levels.iter().all(|&l| l < m), || format!("a level index is not below {m}"))?;
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
}
