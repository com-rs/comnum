//! Binary pseudo-random sequences (linear-feedback shift registers, the length-31 Gold
//! sequence of 3GPP), the spreading codes of spread-spectrum links (m-sequences, Gold
//! and Kasami sets, Walsh–Hadamard and Barker codes) with their correlations, and the
//! Zadoff–Chu sequences.
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
//! ## Spreading codes
//!
//! A binary code $a$ of length $N$ is sent as the chips $(-1)^{a_{i}}$; what a
//! receiver sees of two codes is their periodic correlation ([`periodic_correlation`])
//!
//! $$
//! \theta_{a,b}(l) = \sum_{i=0}^{N-1} (-1)^{a_{i} + b_{(i + l) \bmod N}},
//! $$
//!
//! or, for one isolated period, the aperiodic one ([`aperiodic_correlation`]), the same
//! sum over the indices $0 \le i, i + l < N$ only.
//!
//! | Codes | Length $N$ | Number | Periodic correlations |
//! |---|---|---|---|
//! | [`m_sequence`] of order $r$ | $2^{r} - 1$ | 1 | $\theta_{a,a}(l \ne 0) = -1$ |
//! | [`gold_codes`], $r \not\equiv 0 \pmod 4$ | $2^{r} - 1$ | $2^{r} + 1$ | $\{-1, -t(r), t(r) - 2\}$, $t(r) = 1 + 2^{\lfloor (r + 2)/2 \rfloor}$ |
//! | [`kasami_codes`] (small set), $r$ even | $2^{r} - 1$ | $2^{r/2}$ | $\{-1, -s(r), s(r) - 2\}$, $s(r) = 1 + 2^{r/2}$ |
//! | [`walsh_codes`] | $2^{m}$ | $2^{m}$ | orthogonal at lag 0 only |
//! | [`barker_code`] | 2, 3, 4, 5, 7, 11, 13 | 1 | aperiodic sidelobes $\lvert\cdot\rvert \le 1$ |
//!
//! An m-sequence is the output of a register whose polynomial is primitive: it visits
//! every nonzero state, is balanced ($2^{r-1}$ ones) and has the two-valued
//! autocorrelation of the table. Two m-sequences $u$ and $v = u\[q\]$ (the decimation
//! $v_{i} = u_{q i}$) form a preferred pair for $q = 2^{k} + 1$, $r / \gcd(r, k)$ odd and
//! $\gcd(r, k) \le 2$ (Gold 1967; here $q = 3$ for $r$ odd, $q = 5$ for $r \equiv 2
//! \pmod 4$; none exists for $r \equiv 0 \pmod 4$): the set $\{u, v, u \oplus T^{k} v\}$,
//! $0 \le k < N$, has the three correlation values of the table between any two of its
//! members and at every nonzero lag of each (Sarwate and Pursley 1980).
//! The small Kasami set $\{u, u \oplus T^{k} w\}$, $w = u\[2^{r/2} + 1\]$ of period
//! $2^{r/2} - 1$, has fewer members and reaches the Welch bound.
//! The rows of the Sylvester–Hadamard matrix, $a^{(j)}_{i} = \langle i, j \rangle \bmod 2$
//! on the binary digits of $i$ and $j$, are orthogonal when aligned and not otherwise:
//! the channelisation codes of synchronous downlinks.
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
//! use comnum_math::sequences::{gold_3gpp, gold_codes, lfsr, periodic_correlation};
//! // x³ + x + 1: the m-sequence of period 7
//! assert_eq!(lfsr(&[0, 1], 3, &[1, 0, 0], 10), vec![1, 0, 0, 1, 0, 1, 1, 1, 0, 0]);
//! assert_eq!(gold_3gpp(0, 8).len(), 8);
//! let x = comnum_math::sequences::zadoff_chu(25, 63)?;
//! assert!(x.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
//! // 33 Gold codes of length 31, correlations in {-1, -9, 7}
//! let gold = gold_codes(5)?;
//! assert_eq!((gold.len(), gold[0].len()), (33, 31));
//! assert!(periodic_correlation(&gold[2], &gold[7])?.iter().all(|t| [-1, -9, 7].contains(t)));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 38.211, *NR; Physical channels and modulation*, Sec. 5.2.1.
//! 2. 3GPP TS 36.211 V18.0.2, *E-UTRA; Physical channels and modulation*, Sec. 6.11.1.1, 7.2.
//! 3. D. Chu, "Polyphase codes with good periodic correlation properties," *IEEE Trans.
//!    Inf. Theory* 18(4), 531–532, 1972.
//! 4. R. Gold, "Optimal binary sequences for spread spectrum multiplexing," *IEEE Trans.
//!    Inf. Theory* 13(4), 619–621, 1967.
//! 5. D. V. Sarwate and M. B. Pursley, "Crosscorrelation properties of pseudorandom and
//!    related sequences," *Proc. IEEE* 68(5), 593–619, 1980.
//! 6. R. H. Barker, "Group synchronizing of binary digital systems," in *Communication
//!    Theory*, W. Jackson, Ed. London: Butterworths, 1953, pp. 273–287.

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

/// The feedback taps of a primitive polynomial of every order from 2 to 16 (the
/// exponents below the order of $x^{r} + \sum_{t} x^{t}$), index `order - 2`.
const PRIMITIVE_TAPS: [&[usize]; 15] = [
    &[0, 1],        // x² + x + 1
    &[0, 1],        // x³ + x + 1
    &[0, 1],        // x⁴ + x + 1
    &[0, 2],        // x⁵ + x² + 1
    &[0, 1],        // x⁶ + x + 1
    &[0, 1],        // x⁷ + x + 1
    &[0, 2, 3, 4],  // x⁸ + x⁴ + x³ + x² + 1
    &[0, 4],        // x⁹ + x⁴ + 1
    &[0, 3],        // x¹⁰ + x³ + 1
    &[0, 2],        // x¹¹ + x² + 1
    &[0, 1, 4, 6],  // x¹² + x⁶ + x⁴ + x + 1
    &[0, 1, 3, 4],  // x¹³ + x⁴ + x³ + x + 1
    &[0, 1, 6, 10], // x¹⁴ + x¹⁰ + x⁶ + x + 1
    &[0, 1],        // x¹⁵ + x + 1
    &[0, 1, 3, 12], // x¹⁶ + x¹² + x³ + x + 1
];

/// One period, $N = 2^{r} - 1$ bits, of the m-sequence of order `order` $= r$: the
/// register of a primitive polynomial of that degree started from $1, 0, \dots, 0$
/// (see the module; $x^{5} + x^{2} + 1$, $x^{7} + x + 1$, $x^{10} + x^{3} + 1$ among them).
///
/// # Errors
/// [`comnum_core::Error::Config`] for an order outside 2 to 16.
///
/// # Examples
/// ```
/// use comnum_math::sequences::{m_sequence, periodic_correlation};
/// let u = m_sequence(5)?;
/// assert_eq!(u.iter().filter(|&&b| b == 1).count(), 16);
/// let theta = periodic_correlation(&u, &u)?;
/// assert_eq!(theta[0], 31);
/// assert!(theta[1..].iter().all(|&t| t == -1));
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn m_sequence(order: usize) -> Result<Vec<u8>> {
    ensure((2..=16).contains(&order), || format!("an m-sequence of order 2 to 16, got {order}"))?;
    Ok(lfsr(PRIMITIVE_TAPS[order - 2], order, &[1], (1 << order) - 1))
}

/// The decimation $v_{i} = u_{q i \bmod N}$ of one period `u` of a sequence.
fn decimate(u: &[u8], q: usize) -> Vec<u8> {
    (0..u.len()).map(|i| u[q * i % u.len()]).collect()
}

/// `base` followed by its sums with the `shifts` first cyclic shifts of `other`: the
/// way the Gold and Kasami sets are built from two sequences.
fn sums_with_shifts(base: &[u8], other: &[u8], shifts: usize) -> Vec<Vec<u8>> {
    let n = base.len();
    let shifted = |k: usize| (0..n).map(|i| base[i] ^ other[(i + k) % n]).collect::<Vec<u8>>();
    std::iter::once(base.to_vec()).chain((0..shifts).map(shifted)).collect()
}

/// The $2^{r} + 1$ Gold codes of length $N = 2^{r} - 1$ for `order` $= r$: the
/// m-sequence $u$ of [`m_sequence`], its decimation $v$ and the sums $u \oplus T^{k} v$,
/// $0 \le k < N$, in that order (see the module for the preferred pair and the three
/// correlation values).
///
/// # Errors
/// [`comnum_core::Error::Config`] for an order outside 3 to 16 or a multiple of 4 (no
/// preferred pair exists).
///
/// # Examples
/// ```
/// use comnum_math::sequences::{gold_codes, periodic_correlation};
/// let codes = gold_codes(6)?;
/// assert_eq!((codes.len(), codes[0].len()), (65, 63));
/// // t(6) = 17: the values -1, -17 and 15
/// let theta = periodic_correlation(&codes[3], &codes[40])?;
/// assert!(theta.iter().all(|t| [-1, -17, 15].contains(t)));
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn gold_codes(order: usize) -> Result<Vec<Vec<u8>>> {
    ensure((3..=16).contains(&order) && order % 4 != 0, || {
        format!("Gold codes of order 3 to 16 and not a multiple of 4, got {order}")
    })?;
    let u = m_sequence(order)?;
    let v = decimate(&u, if order % 2 == 1 { 3 } else { 5 });
    let mut codes = sums_with_shifts(&u, &v, u.len());
    codes.insert(1, v);
    Ok(codes)
}

/// The small set of $2^{r/2}$ Kasami codes of length $N = 2^{r} - 1$ for an even
/// `order` $= r$: the m-sequence $u$ of [`m_sequence`] and the sums $u \oplus T^{k} w$,
/// $0 \le k < 2^{r/2} - 1$, with $w = u\[2^{r/2} + 1\]$ (see the module).
///
/// # Errors
/// [`comnum_core::Error::Config`] for an order that is odd or outside 4 to 16.
///
/// # Examples
/// ```
/// use comnum_math::sequences::{kasami_codes, periodic_correlation};
/// let codes = kasami_codes(6)?;
/// assert_eq!((codes.len(), codes[0].len()), (8, 63));
/// // s(6) = 9: the values -1, -9 and 7
/// let theta = periodic_correlation(&codes[1], &codes[5])?;
/// assert!(theta.iter().all(|t| [-1, -9, 7].contains(t)));
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn kasami_codes(order: usize) -> Result<Vec<Vec<u8>>> {
    ensure((4..=16).contains(&order) && order % 2 == 0, || {
        format!("Kasami codes of an even order from 4 to 16, got {order}")
    })?;
    let u = m_sequence(order)?;
    let half = 1usize << (order / 2);
    Ok(sums_with_shifts(&u, &decimate(&u, half + 1), half - 1))
}

/// The `n` Walsh–Hadamard codes of length `n`, a power of two: the rows of the
/// Sylvester–Hadamard matrix in its natural order, $a^{(j)}_{i} = \langle i, j \rangle
/// \bmod 2$ (see the module).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a length that is not a power of two.
///
/// # Examples
/// ```
/// use comnum_math::sequences::walsh_codes;
/// let w = walsh_codes(4)?;
/// assert_eq!(w[1], vec![0, 1, 0, 1]);
/// assert_eq!(w[3], vec![0, 1, 1, 0]);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn walsh_codes(n: usize) -> Result<Vec<Vec<u8>>> {
    ensure(n.is_power_of_two(), || format!("Walsh codes have a power-of-two length, got {n}"))?;
    Ok((0..n).map(|j| (0..n).map(|i| ((i & j).count_ones() & 1) as u8).collect()).collect())
}

/// The Barker code of length `n` (2, 3, 4, 5, 7, 11 or 13), the binary codes whose
/// aperiodic autocorrelation has sidelobes of magnitude at most 1 (see the module;
/// the length 11 is the spreading code of the 1 and 2 Mbit/s modes of IEEE 802.11).
///
/// # Errors
/// [`comnum_core::Error::Config`] for another length.
///
/// # Examples
/// ```
/// use comnum_math::sequences::{aperiodic_correlation, barker_code};
/// let r = aperiodic_correlation(&barker_code(13)?, &barker_code(13)?)?;
/// assert_eq!(r[12], 13);
/// assert!(r.iter().enumerate().all(|(i, &v)| i == 12 || v.abs() <= 1));
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn barker_code(n: usize) -> Result<Vec<u8>> {
    let code: &[u8] = match n {
        2 => &[0, 1],
        3 => &[0, 0, 1],
        4 => &[0, 0, 1, 0],
        5 => &[0, 0, 0, 1, 0],
        7 => &[0, 0, 0, 1, 1, 0, 1],
        11 => &[0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        13 => &[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        _ => return Err(comnum_core::Error::Config(format!("no Barker code of length {n}"))),
    };
    Ok(code.to_vec())
}

/// `Ok` for two codes of the same nonzero length.
fn check_pair(a: &[u8], b: &[u8]) -> Result<()> {
    ensure(!a.is_empty() && a.len() == b.len(), || {
        format!("two codes of the same nonzero length, got {} and {}", a.len(), b.len())
    })
}

/// The sign product $(-1)^{a + b}$ of two bits.
fn chip_product(a: u8, b: u8) -> i32 {
    1 - 2 * i32::from((a ^ b) & 1)
}

/// The periodic correlation $\theta_{a,b}(l) = \sum_{i} (-1)^{a_{i} + b_{(i + l) \bmod N}}$
/// of two binary codes of length $N$ at the lags $l = 0 \dots N - 1$ (see the module),
/// by the direct sum ($N^{2}$ operations, exact).
///
/// # Errors
/// [`comnum_core::Error::Config`] for empty codes or codes of different lengths.
pub fn periodic_correlation(a: &[u8], b: &[u8]) -> Result<Vec<i32>> {
    check_pair(a, b)?;
    let n = a.len();
    Ok((0..n).map(|l| (0..n).map(|i| chip_product(a[i], b[(i + l) % n])).sum()).collect())
}

/// The aperiodic correlation $C_{a,b}(l) = \sum_{i} (-1)^{a_{i} + b_{i + l}}$ over
/// $0 \le i, i + l < N$, at the $2N - 1$ lags $l = -(N - 1) \dots N - 1$ (the lag 0 at
/// the index $N - 1$): what a matched filter gives on one isolated period.
///
/// # Errors
/// [`comnum_core::Error::Config`] for empty codes or codes of different lengths.
pub fn aperiodic_correlation(a: &[u8], b: &[u8]) -> Result<Vec<i32>> {
    check_pair(a, b)?;
    let n = a.len() as isize;
    let at = |l: isize| (0.max(-l)..n.min(n - l)).map(|i| chip_product(a[i as usize], b[(i + l) as usize])).sum();
    Ok((1 - n..n).map(at).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The values taken by the periodic correlations between every pair of `codes`
    /// (every lag) and by the autocorrelation of each off its peak.
    fn correlation_values(codes: &[Vec<u8>]) -> BTreeSet<i32> {
        let mut values = BTreeSet::new();
        for (i, a) in codes.iter().enumerate() {
            for (j, b) in codes.iter().enumerate().skip(i) {
                let theta = periodic_correlation(a, b).unwrap();
                values.extend(theta.iter().skip(usize::from(i == j)));
            }
        }
        values
    }

    #[test]
    fn every_tabulated_polynomial_is_primitive() {
        for order in 2..=16 {
            // the register comes back to its first state after 2^r - 1 steps and not before
            let n = (1usize << order) - 1;
            let x = lfsr(PRIMITIVE_TAPS[order - 2], order, &[1], n + order);
            let state = |i: usize| &x[i..i + order];
            assert_eq!(state(n), state(0), "order {order}");
            assert!((1..n).all(|i| state(i) != state(0)), "order {order}");
            assert_eq!(m_sequence(order).unwrap().iter().filter(|&&b| b == 1).count(), 1 << (order - 1));
        }
        assert!(m_sequence(1).is_err() && m_sequence(17).is_err());
    }

    #[test]
    fn an_m_sequence_has_the_two_valued_autocorrelation() {
        for order in [3, 5, 8, 10] {
            let u = m_sequence(order).unwrap();
            let theta = periodic_correlation(&u, &u).unwrap();
            assert_eq!(theta[0], u.len() as i32);
            assert!(theta[1..].iter().all(|&t| t == -1), "order {order}");
        }
    }

    /// Gold 1967; Sarwate and Pursley 1980: the correlations of a Gold set
    /// take the values -1, -t(r) and t(r) - 2, t(r) = 1 + 2^⌊(r + 2)/2⌋.
    #[test]
    fn gold_codes_have_the_three_correlation_values_of_gold_1967() {
        for (order, t) in [(3, 5), (5, 9), (6, 17), (7, 17)] {
            let codes = gold_codes(order).unwrap();
            assert_eq!(codes.len(), (1 << order) + 1);
            assert_eq!(correlation_values(&codes), BTreeSet::from([-1, -t, t - 2]), "order {order}");
        }
        // the pair alone, at the orders of the larger sets
        for (order, t) in [(9, 33), (10, 65), (11, 65)] {
            let codes = gold_codes(order).unwrap();
            let theta = periodic_correlation(&codes[0], &codes[1]).unwrap();
            assert_eq!(theta.into_iter().collect::<BTreeSet<_>>(), BTreeSet::from([-1, -t, t - 2]));
        }
        assert!(gold_codes(8).is_err() && gold_codes(2).is_err());
    }

    /// Sarwate and Pursley 1980: the small Kasami set has the values -1,
    /// -s(r) and s(r) - 2, s(r) = 1 + 2^(r/2).
    #[test]
    fn kasami_codes_have_the_three_correlation_values_of_the_small_set() {
        for (order, s) in [(4, 5), (6, 9), (8, 17)] {
            let codes = kasami_codes(order).unwrap();
            assert_eq!(codes.len(), 1 << (order / 2));
            assert_eq!(correlation_values(&codes), BTreeSet::from([-1, -s, s - 2]), "order {order}");
        }
        assert!(kasami_codes(5).is_err());
    }

    #[test]
    fn walsh_codes_are_orthogonal_when_aligned() {
        let w = walsh_codes(16).unwrap();
        for (i, a) in w.iter().enumerate() {
            for (j, b) in w.iter().enumerate() {
                assert_eq!(periodic_correlation(a, b).unwrap()[0], if i == j { 16 } else { 0 });
            }
        }
        assert!(walsh_codes(12).is_err());
    }

    #[test]
    fn barker_codes_have_unit_aperiodic_sidelobes() {
        for n in [2, 3, 4, 5, 7, 11, 13] {
            let c = barker_code(n).unwrap();
            let r = aperiodic_correlation(&c, &c).unwrap();
            assert_eq!(r.len(), 2 * n - 1);
            assert_eq!(r[n - 1], n as i32);
            assert!(r.iter().enumerate().all(|(i, &v)| i == n - 1 || v.abs() <= 1), "length {n}");
        }
        assert!(barker_code(6).is_err());
        assert!(periodic_correlation(&[0, 1], &[0]).is_err());
    }
}
