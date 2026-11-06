//! Compensation of the quadrature imbalance of a coherent front end by orthonormalisation
//! of its in-phase and quadrature outputs (Savory 2010, Sec. II-B, eqs. (6)–(8)).
//!
//! When the two outputs $r_{1}, r_{2}$ of unit power are correlated,
//! $R = \begin{bmatrix}1 & \rho\\ \rho & 1\end{bmatrix}$ with $\rho = \langle r_{1}r_{2}\rangle$
//! ($\rho = \sin 2\theta$ for a phase error $2\theta$ between the arms, eq. (6)), two
//! orthonormalisations remove the correlation:
//!
//! - Gram–Schmidt ([`gram_schmidt`], eq. (7)): $r_{1}$ is kept, $r_{2}$ loses its
//!   projection on $r_{1}$ and is divided by $\alpha = \sqrt{\langle r_{2}^{2}\rangle - \langle r_{1}r_{2}\rangle^{2}/\langle r_{1}^{2}\rangle}$
//!   ($\alpha = \cos 2\theta$; the paper writes "multiplied", eq. (7));
//! - Löwdin ([`lowdin`], eq. (8)): $L = R^{-1/2}$, the symmetric orthonormalisation, which
//!   turns both outputs by $\theta$ towards each other; for $r = M[I, Q]^{T}$ with
//!   $M = \begin{bmatrix}\cos\theta & \sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}$ it
//!   restores $I$ and $Q$ exactly ($R = M^{2}$, $R^{-1/2} = M^{-1}$).
//!
//! The moments are the sample means of the block; the outputs are normalised to the
//! power of $r_{1}$.
//!
//! # Examples
//! ```
//! use comnum_dsp::compensation::iq::{iq_correlation, lowdin};
//! use comnum_math::C64;
//! // a phase error of 2 theta = 0.4 rad between the arms of a QPSK signal
//! let th: f64 = 0.2;
//! let x: Vec<C64> = (0..400).map(|k| C64::new([1.0, -1.0][k % 2], [1.0, -1.0][(k / 2) % 2])).collect();
//! let r: Vec<C64> = x.iter().map(|z| C64::new(th.cos() * z.re + th.sin() * z.im, th.sin() * z.re + th.cos() * z.im)).collect();
//! assert!((iq_correlation(&r) - (2.0 * th).sin()).abs() < 1e-12);
//! let y = lowdin(&r)?;
//! assert!(y.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. S. J. Savory, "Digital coherent optical receivers: algorithms and subsystems,"
//!    *IEEE J. Sel. Topics Quantum Electron.* 16(5), 1164–1179, 2010: eqs. (6)–(8), Fig. 3.

use comnum_core::{Error, Result};
use comnum_math::C64;

/// The sample moments $\langle r_{1}^{2}\rangle$, $\langle r_{1}r_{2}\rangle$, $\langle r_{2}^{2}\rangle$.
fn moments(r: &[C64]) -> (f64, f64, f64) {
    let n = r.len() as f64;
    let (a, b, c) = r.iter().fold((0.0, 0.0, 0.0), |(a, b, c), z| (a + z.re * z.re, b + z.re * z.im, c + z.im * z.im));
    (a / n, b / n, c / n)
}

/// The correlation coefficient $\langle r_{1}r_{2}\rangle / \sqrt{\langle r_{1}^{2}\rangle\langle r_{2}^{2}\rangle}$
/// of the in-phase and quadrature parts of `r` ($\sin 2\theta$ in eq. (6)).
pub fn iq_correlation(r: &[C64]) -> f64 {
    let (a, b, c) = moments(r);
    b / (a * c).sqrt()
}

fn check(r: &[C64]) -> Result<(f64, f64, f64)> {
    if r.len() < 2 {
        return Err(Error::Shape(format!("an orthonormalisation needs at least 2 samples, got {}", r.len())));
    }
    let (a, b, c) = moments(r);
    if !(a > 0.0 && c > 0.0 && a * c > b * b) {
        return Err(Error::Numerical("the in-phase and quadrature parts are not linearly independent".into()));
    }
    Ok((a, b, c))
}

/// Gram–Schmidt orthonormalisation (eq. (7)): $r_{1}$ kept, $r_{2} - \frac{\langle r_{1}r_{2}\rangle}{\langle r_{1}^{2}\rangle}r_{1}$
/// divided by $\alpha$ and scaled to the power of $r_{1}$.
///
/// # Errors
/// [`Error::Shape`] for fewer than 2 samples; [`Error::Numerical`] for parts that are not
/// linearly independent.
pub fn gram_schmidt(r: &[C64]) -> Result<Vec<C64>> {
    let (a, b, c) = check(r)?;
    let alpha = (c - b * b / a).sqrt();
    let scale = a.sqrt() / alpha;
    Ok(r.iter().map(|z| C64::new(z.re, (z.im - b / a * z.re) * scale)).collect())
}

/// Löwdin (symmetric) orthonormalisation (eq. (8)): the parts multiplied by $R^{-1/2}$,
/// $R$ their correlation matrix, normalised to the power of $r_{1}$.
///
/// # Errors
/// As [`gram_schmidt`].
pub fn lowdin(r: &[C64]) -> Result<Vec<C64>> {
    let (a, b, c) = check(r)?;
    // normalise both parts to unit power, then R = [[1, rho], [rho, 1]] and
    // R^{-1/2} = (u (1 + rho)^{-1/2} u^T + v (1 - rho)^{-1/2} v^T), u, v = [1, +-1]/sqrt(2)
    let (sa, sc) = (a.sqrt(), c.sqrt());
    let rho = b / (sa * sc);
    let (p, m) = (1.0 / (1.0 + rho).sqrt(), 1.0 / (1.0 - rho).sqrt());
    let (d, o) = ((p + m) / 2.0, (p - m) / 2.0);
    Ok(r.iter()
        .map(|z| {
            let (x, y) = (z.re / sa, z.im / sc);
            C64::new(d * x + o * y, o * x + d * y) * sa
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both outputs are uncorrelated and keep the power of the first part; dependent
    /// parts and short blocks are errors.
    #[test]
    fn both_orthonormalisations_decorrelate_the_parts() {
        let r: Vec<C64> = (0..1000)
            .map(|k| {
                let (i, q) = (((k * 7919) % 13) as f64 - 6.0, ((k * 104_729) % 11) as f64 - 5.0);
                C64::new(i, 0.3 * i + 0.8 * q)
            })
            .collect();
        for y in [gram_schmidt(&r).unwrap(), lowdin(&r).unwrap()] {
            assert!(iq_correlation(&y).abs() < 1e-12);
            let (a, _, c) = moments(&y);
            assert!((a - c).abs() < 1e-9 * a);
        }
        assert!(gram_schmidt(&[C64::new(1.0, 2.0), C64::new(2.0, 4.0)]).is_err());
        assert!(lowdin(&[C64::new(1.0, 2.0)]).is_err());
    }
}
