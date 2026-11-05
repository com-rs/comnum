//! The modulation mapper of 5G NR (TS 38.211 Sec. 5.1).

use comnum_core::{Result, ensure};
use comnum_mapping::constellation::Constellation;
use comnum_math::C64;

/// The constellation of $Q_{m}$ = `qm` bits per symbol of TS 38.211 Sec. 5.1.3 to 5.1.7:
/// QPSK (2), 16-QAM (4), 64-QAM (6), 256-QAM (8), 1024-QAM (10), at unit mean energy.
///
/// The bits $b_{0}, b_{2}, \ldots$ of a label select the real part and $b_{1}, b_{3},
/// \ldots$ the imaginary part, each by the nested form of the standard, e.g. for 16-QAM
///
/// $$
/// d = \frac{1}{\sqrt{10}}\left\\{(1 - 2b_{0})\left[2 - (1 - 2b_{2})\right] + j (1 - 2b_{1})
/// \left[2 - (1 - 2b_{3})\right]\right\\},
/// $$
///
/// and for $k$ bits $s_{0}, \ldots, s_{k-1}$ on an axis, $(1 - 2s_{0})\\, t_{1}$ with
/// $t_{k} = 1$ and $t_{j} = 2^{k-j} - (1 - 2s_{j})\\, t_{j+1}$; `points()[i]` carries the
/// bits of `i`, $b_{0}$ the most significant.
///
/// # Errors
/// [`comnum_core::Error::Config`] for another $Q_{m}$.
///
/// # Examples
/// ```
/// use comnum_systems::nr::modulation::constellation;
/// use comnum_math::C64;
/// let c = constellation(4)?;
/// // b = 0011: 3 + 3j over sqrt(10)
/// assert!((c.points()[0b0011] - C64::new(3.0, 3.0) / 10f64.sqrt()).norm() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn constellation(qm: usize) -> Result<Constellation> {
    ensure(matches!(qm, 2 | 4 | 6 | 8 | 10), || format!("Qm must be 2, 4, 6, 8 or 10, got {qm}"))?;
    let k = qm / 2;
    let axis = |s: &[usize]| -> f64 {
        let mut t = 1.0;
        for j in (1..k).rev() {
            t = f64::from(1u32 << (k - j)) - (1.0 - 2.0 * s[j] as f64) * t;
        }
        (1.0 - 2.0 * s[0] as f64) * t
    };
    let points: Vec<C64> = (0..1usize << qm)
        .map(|label| {
            let b: Vec<usize> = (0..qm).map(|j| (label >> (qm - 1 - j)) & 1).collect();
            let re: Vec<usize> = b.iter().step_by(2).copied().collect();
            let im: Vec<usize> = b.iter().skip(1).step_by(2).copied().collect();
            C64::new(axis(&re), axis(&im))
        })
        .collect();
    Ok(Constellation::from_points(points)?.normalized())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code: a failure is a test failure")]
    use super::*;

    #[test]
    fn the_mappers_follow_sec_5_1_3_to_5_1_6() {
        // QPSK (5.1.3): ((1 - 2b0) + j(1 - 2b1)) / sqrt 2
        let q = constellation(2).unwrap();
        assert!((q.points()[0b10] - C64::new(-1.0, 1.0) / 2f64.sqrt()).norm() < 1e-12);
        // 64-QAM (5.1.5): b = 100110 -> real (1-2)(4 - (1)(2 - (-1))) = -1, imag (1)(4 - (-1)(2 - 1)) = 5
        let c = constellation(6).unwrap();
        let want = C64::new(-1.0, 5.0) / 42f64.sqrt();
        assert!((c.points()[0b10_0110] - want).norm() < 1e-12);
        // 256-QAM (5.1.6): the largest point 15 + 15j over sqrt 170 for b = 00111111
        let d = constellation(8).unwrap();
        assert!((d.points()[0b0011_1111] - C64::new(15.0, 15.0) / 170f64.sqrt()).norm() < 1e-12);
        for qm in [2, 4, 6, 8, 10] {
            let e: f64 =
                constellation(qm).unwrap().points().iter().map(C64::norm_sqr).sum::<f64>() / f64::from(1u32 << qm);
            assert!((e - 1.0).abs() < 1e-12);
        }
        assert!(constellation(3).is_err());
    }
}
