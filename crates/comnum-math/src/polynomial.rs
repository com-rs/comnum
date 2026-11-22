//! Roots of a polynomial with real coefficients by the Durand–Kerner iteration (the
//! simultaneous Newton steps of Weierstrass), polished by Newton's method: the poles of
//! analogue filter prototypes (the reverse Bessel polynomial of a Bessel filter), small
//! degrees.
//!
//! # Model
//!
//! For $p(x) = \sum_{k=0}^{n} c_{k} x^{k}$, $c_{n} \ne 0$, the $n$ estimates $z_{i}$ are
//! updated together,
//!
//! $$
//! z_{i} \leftarrow z_{i} - \frac{p(z_{i})/c_{n}}{\prod_{j \ne i}(z_{i} - z_{j})},
//! $$
//!
//! from distinct starting points on a circle, until the largest step is below the
//! tolerance; each root then takes a few Newton steps on $p$ itself.
//!
//! # References
//! 1. E. Durand, *Solutions numériques des équations algébriques*, vol. 1, Masson, 1960.
//! 2. I. O. Kerner, "Ein Gesamtschrittverfahren zur Berechnung der Nullstellen von
//!    Polynomen," *Numer. Math.* 8, 290–294, 1966.

use crate::C64;
use comnum_core::{Error, Result, ensure};

/// The value of $\sum_{k} c_{k} z^{k}$ at `z` (Horner).
fn horner(coeffs: &[f64], z: C64) -> C64 {
    coeffs.iter().rev().fold(C64::new(0.0, 0.0), |acc, &c| acc * z + c)
}

/// The $n$ roots of $\sum_{k=0}^{n} c_{k} x^{k}$ (`coeffs[k]` $= c_{k}$, the constant
/// term first), in no particular order; conjugate pairs come out as such to the
/// tolerance of the iteration ($10^{-12}$ relative, then polished).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a degree below 1 or a zero leading coefficient,
/// [`comnum_core::Error::Numerical`] if the iteration does not converge in 500 steps.
///
/// # Examples
/// ```
/// use comnum_math::polynomial::roots;
/// // x² - 3x + 2 = (x - 1)(x - 2)
/// let mut r: Vec<f64> = roots(&[2.0, -3.0, 1.0])?.iter().map(|z| z.re).collect();
/// r.sort_by(f64::total_cmp);
/// assert!((r[0] - 1.0).abs() < 1e-12 && (r[1] - 2.0).abs() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn roots(coeffs: &[f64]) -> Result<Vec<C64>> {
    let n = coeffs.len().saturating_sub(1);
    ensure(n >= 1, || "a polynomial of degree 0 has no root".into())?;
    ensure(coeffs[n] != 0.0 && coeffs.iter().all(|c| c.is_finite()), || {
        "the leading coefficient must be finite and non-zero".into()
    })?;
    let monic: Vec<f64> = coeffs.iter().map(|c| c / coeffs[n]).collect();
    // a radius that bounds the roots (Cauchy), starting points spread on it
    let radius = 1.0 + monic[..n].iter().fold(0.0f64, |m, c| m.max(c.abs()));
    let mut z: Vec<C64> = (0..n)
        .map(|i| C64::from_polar(radius * 0.9, 2.0 * std::f64::consts::PI * (i as f64 + 0.25) / n as f64 + 0.4))
        .collect();
    let mut converged = false;
    for _ in 0..500 {
        let mut largest = 0.0f64;
        for i in 0..n {
            let denominator = (0..n).filter(|&j| j != i).fold(C64::new(1.0, 0.0), |d, j| d * (z[i] - z[j]));
            let step = horner(&monic, z[i]) / denominator;
            z[i] -= step;
            largest = largest.max(step.norm() / z[i].norm().max(1.0));
        }
        if largest < 1e-12 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Error::Numerical("the roots did not converge".into()));
    }
    // Newton polish on the original polynomial
    let derivative: Vec<f64> = (1..=n).map(|k| k as f64 * coeffs[k]).collect();
    for zi in &mut z {
        for _ in 0..3 {
            let d = horner(&derivative, *zi);
            if d.norm() > 0.0 {
                *zi -= horner(coeffs, *zi) / d;
            }
        }
    }
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_roots_of_a_product_of_known_factors_are_found_and_errors_are_reported() {
        // (x - 1)(x - 2)(x² + 1)(x + 3) = x^5 + 0 x^4 - 6 x^3 + ... expand numerically
        let factors: [Vec<f64>; 4] = [vec![-1.0, 1.0], vec![-2.0, 1.0], vec![1.0, 0.0, 1.0], vec![3.0, 1.0]];
        let mut p = vec![1.0];
        for f in &factors {
            let mut q = vec![0.0; p.len() + f.len() - 1];
            for (i, a) in p.iter().enumerate() {
                for (j, b) in f.iter().enumerate() {
                    q[i + j] += a * b;
                }
            }
            p = q;
        }
        let r = roots(&p).unwrap();
        for expected in
            [C64::new(1.0, 0.0), C64::new(2.0, 0.0), C64::new(0.0, 1.0), C64::new(0.0, -1.0), C64::new(-3.0, 0.0)]
        {
            assert!(r.iter().any(|z| (z - expected).norm() < 1e-10), "{expected} missing from {r:?}");
        }
        assert!(roots(&[1.0]).is_err());
        assert!(roots(&[1.0, 0.0]).is_err());
    }
}
