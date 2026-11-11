//! Binary pseudo-random sequences (linear-feedback shift registers, the length-31 Gold
//! sequence of 3GPP) and the Zadoff–Chu sequences.
//!
//! # Model
//!
//! A Fibonacci register of order $r$ produces $x(i + r) = \sum_{t \in T} x(i + t) \bmod 2$
//! from its first $r$ bits ([`lfsr`]): every m-sequence and Gold sequence of the
//! standards is one or a sum of two. The pseudo-random sequence of 3GPP (TS 38.211,
//! Sec. 5.2.1; TS 36.211, Sec. 7.2) is
//!
//! $$
//! c(n) = \left(x_{1}(n + N_{c}) + x_{2}(n + N_{c})\right) \bmod 2, \qquad N_{c} = 1600,
//! $$
//!
//! with $x_{1}(n+31) = (x_{1}(n+3) + x_{1}(n)) \bmod 2$, $x_{1}(0) = 1$,
//! $x_{1}(1..30) = 0$, and $x_{2}(n+31) = (x_{2}(n+3) + x_{2}(n+2) + x_{2}(n+1) +
//! x_{2}(n)) \bmod 2$ initialised by $c_{\mathrm{init}} = \sum_{i} x_{2}(i) 2^{i}$
//! ([`gold_3gpp`]): the scrambling of the coded bits and the sequences of the reference
//! signals.
//!
//! The Zadoff–Chu sequence of root $u$ and odd length $N_{\mathrm{ZC}}$ ([`zadoff_chu`]),
//!
//! $$
//! x_{u}(n) = e^{-j\pi u n (n + 1) / N_{\mathrm{ZC}}}, \qquad 0 \le n < N_{\mathrm{ZC}},
//! \quad \gcd(u, N_{\mathrm{ZC}}) = 1,
//! $$
//!
//! has a constant amplitude and a zero periodic autocorrelation at every nonzero lag
//! (a CAZAC sequence; Chu 1972), and two roots $u_{1} \ne u_{2}$ with $u_{1} - u_{2}$
//! prime to a prime $N_{\mathrm{ZC}}$ cross-correlate with the constant magnitude
//! $1/\sqrt{N_{\mathrm{ZC}}}$: the primary synchronisation signal of LTE (TS 36.211,
//! Sec. 6.11.1.1, $N_{\mathrm{ZC}} = 63$, its middle element punctured) and the
//! preambles of the random access.
//!
//! # Examples
//! ```
//! use comnum_math::sequences::{gold_3gpp, lfsr};
//! // x³ + x + 1: the m-sequence of period 7
//! assert_eq!(lfsr(&[0, 1], 3, &[1, 0, 0], 10), vec![1, 0, 0, 1, 0, 1, 1, 1, 0, 0]);
//! assert_eq!(gold_3gpp(0, 8).len(), 8);
//! let x = comnum_math::sequences::zadoff_chu(25, 63)?;
//! assert!(x.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 38.211, *NR; Physical channels and modulation*, Sec. 5.2.1.
//! 2. 3GPP TS 36.211 V18.0.2, *E-UTRA; Physical channels and modulation*, Sec. 6.11.1.1, 7.2.
//! 3. D. Chu, "Polyphase codes with good periodic correlation properties," *IEEE Trans.
//!    Inf. Theory* 18(4), 531–532, 1972.

use crate::C64;
use comnum_core::{Result, ensure};

/// The sequence of a linear-feedback shift register of order `order`: the first
/// `order` bits are `init` (zero-padded), then $x_{i + \mathrm{order}} = \sum_{t} x_{i + t}
/// \bmod 2$ over the `taps`, up to `len` bits in all (the Fibonacci form of every
/// m-sequence, Gold sequence and PRBS of the standards). The taps are below `order`:
/// a tap at `order` or beyond, which is not a bit of the register, is left out.
///
/// # Examples
/// ```
/// use comnum_math::sequences::lfsr;
/// // x³ + x + 1: the m-sequence of period 7
/// let x = lfsr(&[0, 1], 3, &[1, 0, 0], 10);
/// assert_eq!(x, vec![1, 0, 0, 1, 0, 1, 1, 1, 0, 0]);
/// assert_eq!(lfsr(&[0, 1, 5], 3, &[1, 0, 0], 10), x);
/// ```
pub fn lfsr(taps: &[usize], order: usize, init: &[u8], len: usize) -> Vec<u8> {
    let mut x = vec![0u8; len.max(order)];
    for (v, &b) in x.iter_mut().zip(init.iter().take(order)) {
        *v = b & 1;
    }
    for i in 0..len.saturating_sub(order) {
        x[i + order] = taps.iter().filter(|&&t| t < order).fold(0, |acc, &t| acc ^ x[i + t]);
    }
    x.truncate(len);
    x
}

/// The first `n` bits of the Gold sequence $c$ of 3GPP with the initial value `c_init`
/// (31 bits; see the module).
pub fn gold_3gpp(c_init: u32, n: usize) -> Vec<u8> {
    const NC: usize = 1600;
    let len = n + NC + 31;
    let init2: Vec<u8> = (0..31).map(|i| ((c_init >> i) & 1) as u8).collect();
    let x1 = lfsr(&[0, 3], 31, &[1], len);
    let x2 = lfsr(&[0, 1, 2, 3], 31, &init2, len);
    (0..n).map(|i| x1[i + NC] ^ x2[i + NC]).collect()
}

/// The Zadoff–Chu sequence $x_{u}(n) = e^{-j\pi u n (n + 1) / N_{\mathrm{ZC}}}$ of root `u`
/// and odd length `n_zc` (see the module).
///
/// # Errors
/// [`comnum_core::Error::Config`] for an even or zero length, or a root that is zero
/// modulo $N_{\mathrm{ZC}}$ or not prime to it.
///
/// # Examples
/// ```
/// use comnum_math::sequences::zadoff_chu;
/// let x = zadoff_chu(29, 63)?;
/// // zero periodic autocorrelation at every nonzero lag
/// let r: comnum_math::C64 = (0..63).map(|n| x[n] * x[(n + 5) % 63].conj()).sum();
/// assert!(r.norm() < 1e-9);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn zadoff_chu(u: u32, n_zc: u32) -> Result<Vec<C64>> {
    ensure(n_zc % 2 == 1, || format!("a Zadoff–Chu sequence has an odd length, got {n_zc}"))?;
    let gcd = |mut a: u64, mut b: u64| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    ensure(u % n_zc != 0 && gcd(u64::from(u), u64::from(n_zc)) == 1, || {
        format!("the root {u} must be nonzero and prime to the length {n_zc}")
    })?;
    let n_zc = u64::from(n_zc);
    // the phase u n (n + 1) / N reduced modulo 2N exactly, in integers
    Ok((0..n_zc)
        .map(|n| {
            let k = (u64::from(u) * n % (2 * n_zc)) * (n + 1) % (2 * n_zc);
            C64::from_polar(1.0, -std::f64::consts::PI * k as f64 / n_zc as f64)
        })
        .collect())
}
