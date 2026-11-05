//! Sources of bits and symbols (OptiCommPy `optic.comm.sources`).
//!
//! - [`bits`]: independent fair bits from a seeded generator, and [`qpsk`], one
//!   unit-energy QPSK point (the pilots and preambles drawn from a seed);
//! - [`prbs`]: maximal-length pseudo-random binary sequences with the polynomials of
//!   ITU-T O.150 (orders 9, 11, 15, 20, 23, 29, 31, and 7), produced by a Fibonacci
//!   linear-feedback shift register from any register state: the sequence of order `n`
//!   has period `2ⁿ - 1` and contains every non-zero `n`-bit word exactly once per
//!   period;
//! - [`O150Sequence`]: the eight test sequences of ITU-T O.150 (05/96), Sec. 5, exactly:
//!   the polynomial, the inversion of the output (Sec. 4), the zero suppression of the
//!   second sequence of period $2^{20} - 1$ (Sec. 5.5) and the starting phase of the
//!   511-bit sequence (Sec. 5.1), the sequences of the error-rate test sets;
//! - [`symbols`]: random symbols of a constellation with a given probability mass
//!   function, uniform by default, or Maxwell–Boltzmann shaped with
//!   [`maxwell_boltzmann`] (probabilistic shaping, `p(x) ∝ exp(-λ|x|²)`).
//!
//! # Examples
//! ```
//! use comnum_mapping::source;
//! use comnum_core::rng::Rng;
//! let b = source::bits(1000, &mut Rng::new(3));
//! assert!(b.iter().all(|&v| v <= 1));
//! let p = source::prbs(7, 127 * 2, 1).unwrap();
//! assert_eq!(&p[..127], &p[127..]);           // period 2⁷ - 1
//! assert_eq!(p[..127].iter().filter(|&&v| v == 1).count(), 64);
//! // the 2¹⁵ − 1 sequence of the O.151 test sets: inverted, it starts with 15 zeros
//! let t = source::O150Sequence::Prbs15.bits(16);
//! assert_eq!(&t[..15], &[0; 15]);
//! assert_eq!(t[15], 1);
//! ```
//!
//! # References
//!
//! 1. ITU-T Recommendation O.150 (05/96), *General requirements for instrumentation for
//!    performance measurements on digital transmission equipment*, Sec. 3 to 5 and
//!    Table 1.

use crate::constellation::Constellation;
use comnum_core::Result;
use comnum_core::rng::Rng;
use comnum_math::C64;

/// `n` independent fair bits (`0` or `1`).
pub fn bits(n: usize, rng: &mut Rng) -> Vec<u8> {
    (0..n).map(|_| rng.bit()).collect()
}

/// One unit-energy QPSK point $e^{j\pi(2i + 1)/4}$, $i$ uniform in `0..4` (one draw of
/// [`Rng::below`]): the pilots and preambles drawn from a seed.
///
/// # Examples
/// ```
/// use comnum_core::rng::Rng;
/// let s = comnum_mapping::source::qpsk(&mut Rng::new(1));
/// assert!((s.norm() - 1.0).abs() < 1e-15 && (s.re.abs() - s.im.abs()).abs() < 1e-15);
/// ```
#[inline]
pub fn qpsk(rng: &mut Rng) -> C64 {
    C64::from_polar(1.0, std::f64::consts::FRAC_PI_4 * (2 * rng.below(4) + 1) as f64)
}

/// Feedback taps `(n, k)` of the PRBS polynomials `xⁿ + xᵏ + 1`: those of ITU-T O.150,
/// Sec. 5.1 to 5.4 and 5.6 to 5.8, and `x⁷ + x⁶ + 1`, not in O.150.
const PRBS_TAPS: [(u32, u32); 8] = [(7, 6), (9, 5), (11, 9), (15, 14), (20, 3), (23, 18), (29, 27), (31, 28)];

/// `length` bits of the PRBS of order `order`, from the register state `seed`.
///
/// The register is a Fibonacci LFSR ([`comnum_math::sequences::lfsr`]) with the polynomial
/// `xⁿ + xᵏ + 1` of the order;
/// `seed` gives its initial content (its `n` low bits; an all-zero register is replaced
/// by 1, since it would lock the sequence at zero). The output bit is the feedback bit:
/// the sequence of O.150 at another phase and never inverted; [`O150Sequence`] gives the
/// test sequences of the Recommendation exactly.
///
/// # Errors
/// [`comnum_core::Error::Config`] for an order outside `{7, 9, 11, 15, 20, 23, 29, 31}`.
pub fn prbs(order: u32, length: usize, seed: u64) -> Result<Vec<u8>> {
    let (n, k) =
        PRBS_TAPS.iter().copied().find(|&(n, _)| n == order).ok_or_else(|| {
            comnum_core::Error::Config(format!("PRBS order {order} not in 7, 9, 11, 15, 20, 23, 29, 31"))
        })?;
    let (n, k) = (n as usize, k as usize);
    let reg = match seed & ((1u64 << n) - 1) {
        0 => 1,
        r => r,
    };
    // the register holds the last n outputs, the newest in its bit 0: the output bit
    // b(t) = b(t - n) ⊕ b(t - k) is the LFSR x(i + n) = x(i) ⊕ x(i + n - k)
    let init: Vec<u8> = (0..n).map(|i| ((reg >> (n - 1 - i)) & 1) as u8).collect();
    Ok(comnum_math::sequences::lfsr(&[0, n - k], n, &init, n + length).split_off(n))
}

/// The pseudo-random test sequences of ITU-T O.150 (05/96), Sec. 5 and Table 1.
///
/// Each is produced by an $n$-stage shift register whose $k$-th and $n$-th stage outputs
/// are added modulo two and fed back to the first stage,
///
/// $$
/// Q_{i+1}(t+1) = Q_{i}(t), \quad i = 1, \dots, n - 1, \qquad Q_{1}(t+1) = Q_{k}(t) \oplus Q_{n}(t),
/// $$
///
/// the polynomial $x^{n} + x^{k} + 1$, period $2^{n} - 1$. The output is the last stage
/// $Q_{n}(t)$, negated for the inverted sequences, whose longest run of zeros is then $n$
/// instead of $n - 1$ (Sec. 4). The second sequence of period $2^{20} - 1$ (Sec. 5.5)
/// forces its output to one before 14 zeros,
/// $RD(t) = Q_{20}(t) + \overline{Q_{6}(t) + \dots + Q_{19}(t)}$ ($+$ the logical OR).
///
/// | Sequence | Clause | $n$ | $k$ | Output | Longest run of zeros |
/// | --- | --- | --- | --- | --- | --- |
/// | `Prbs9` | 5.1 | 9 | 5 | non-inverted | 8 |
/// | `Prbs11` | 5.2 | 11 | 9 | non-inverted | 10 |
/// | `Prbs15` | 5.3 | 15 | 14 | inverted | 15 |
/// | `Prbs20` | 5.4 | 20 | 3 | non-inverted | 19 |
/// | `Prbs20Zs` | 5.5 | 20 | 17 | zero suppression | 14 |
/// | `Prbs23` | 5.6 | 23 | 18 | inverted | 23 |
/// | `Prbs29` | 5.7 | 29 | 27 | inverted | 29 |
/// | `Prbs31` | 5.8 | 31 | 28 | inverted | 31 |
///
/// The Recommendation fixes the starting phase of the 511-bit sequence only, "the first
/// ONE of 9 consecutive ONEs" (Sec. 5.1); [`O150Sequence::bits`] starts every register
/// with its stages at one, which gives that phase and makes an inverted sequence begin
/// with its $n$ zeros. A test set synchronises to the received sequence (Sec. 4.2), so
/// the phase does not matter for a measurement.
///
/// The printed formula of Sec. 5.5 ends its OR with $Q_{6}(k)$ instead of $Q_{19}(k)$,
/// and Sec. 5.8 calls the 31-stage register "twenty-nine-stage": two misprints that the
/// text of the same clauses corrects (Sec. 5.1 of the Recommendation).
///
/// # Examples
/// ```
/// use comnum_mapping::source::O150Sequence;
/// let s = O150Sequence::Prbs9;
/// let b = s.bits(511 + 9);
/// assert_eq!(&b[..9], &[1; 9]);              // Sec. 5.1: the first ONE of 9 ONEs
/// assert_eq!(&b[511..], &b[..9]);            // period 2⁹ − 1
/// assert_eq!(O150Sequence::from_name("prbs20z")?, O150Sequence::Prbs20Zs);
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum O150Sequence {
    /// $2^{9} - 1$ bits, Sec. 5.1 (O.153, data circuits up to 14 400 bit/s).
    Prbs9,
    /// $2^{11} - 1$ bits, Sec. 5.2 (O.152, 64 kbit/s and N × 64 kbit/s).
    Prbs11,
    /// $2^{15} - 1$ bits, inverted, Sec. 5.3 (O.151, 1544 to 44 736 kbit/s).
    Prbs15,
    /// $2^{20} - 1$ bits, Sec. 5.4 (O.153, data circuits up to 72 kbit/s).
    Prbs20,
    /// $2^{20} - 1$ bits with zero suppression, Sec. 5.5 (O.151, 1544 to 44 736 kbit/s).
    Prbs20Zs,
    /// $2^{23} - 1$ bits, inverted, Sec. 5.6 (O.151, 34 368 and 139 264 kbit/s).
    Prbs23,
    /// $2^{29} - 1$ bits, inverted, Sec. 5.7 (specific measurement tasks).
    Prbs29,
    /// $2^{31} - 1$ bits, inverted, Sec. 5.8 (specific measurement tasks).
    Prbs31,
}

impl O150Sequence {
    /// The eight sequences, in the order of Sec. 5.
    pub const ALL: [O150Sequence; 8] = [
        O150Sequence::Prbs9,
        O150Sequence::Prbs11,
        O150Sequence::Prbs15,
        O150Sequence::Prbs20,
        O150Sequence::Prbs20Zs,
        O150Sequence::Prbs23,
        O150Sequence::Prbs29,
        O150Sequence::Prbs31,
    ];

    /// The sequence of a name: `prbs9`, `prbs11`, `prbs15`, `prbs20`, `prbs20z` (the zero
    /// suppression of Sec. 5.5), `prbs23`, `prbs29`, `prbs31`.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for another name.
    pub fn from_name(name: &str) -> Result<O150Sequence> {
        O150Sequence::ALL.into_iter().find(|s| s.name() == name.to_ascii_lowercase()).ok_or_else(|| {
            let known: Vec<&str> = O150Sequence::ALL.iter().map(|s| s.name()).collect();
            comnum_core::Error::Config(format!("O.150 sequence {name:?} not in {}", known.join(", ")))
        })
    }

    /// The name of [`O150Sequence::from_name`].
    pub fn name(self) -> &'static str {
        match self {
            O150Sequence::Prbs9 => "prbs9",
            O150Sequence::Prbs11 => "prbs11",
            O150Sequence::Prbs15 => "prbs15",
            O150Sequence::Prbs20 => "prbs20",
            O150Sequence::Prbs20Zs => "prbs20z",
            O150Sequence::Prbs23 => "prbs23",
            O150Sequence::Prbs29 => "prbs29",
            O150Sequence::Prbs31 => "prbs31",
        }
    }

    /// The number of stages $n$.
    pub fn stages(self) -> u32 {
        match self {
            O150Sequence::Prbs9 => 9,
            O150Sequence::Prbs11 => 11,
            O150Sequence::Prbs15 => 15,
            O150Sequence::Prbs20 | O150Sequence::Prbs20Zs => 20,
            O150Sequence::Prbs23 => 23,
            O150Sequence::Prbs29 => 29,
            O150Sequence::Prbs31 => 31,
        }
    }

    /// The other feedback stage $k$: the polynomial is $x^{n} + x^{k} + 1$.
    pub fn feedback(self) -> u32 {
        match self {
            O150Sequence::Prbs9 => 5,
            O150Sequence::Prbs11 => 9,
            O150Sequence::Prbs15 => 14,
            O150Sequence::Prbs20 => 3,
            O150Sequence::Prbs20Zs => 17,
            O150Sequence::Prbs23 => 18,
            O150Sequence::Prbs29 => 27,
            O150Sequence::Prbs31 => 28,
        }
    }

    /// Whether the output is the negated register output (Sec. 4).
    pub fn inverted(self) -> bool {
        matches!(self, O150Sequence::Prbs15 | O150Sequence::Prbs23 | O150Sequence::Prbs29 | O150Sequence::Prbs31)
    }

    /// The period $2^{n} - 1$.
    pub fn period(self) -> u64 {
        (1u64 << self.stages()) - 1
    }

    /// The longest run of zeros of the sequence, Table 1.
    pub fn longest_zeros(self) -> u32 {
        match self {
            O150Sequence::Prbs20Zs => 14,
            s if s.inverted() => s.stages(),
            s => s.stages() - 1,
        }
    }

    /// The first `length` bits of the sequence, from the register with every stage at
    /// one (see the type).
    pub fn bits(self, length: usize) -> Vec<u8> {
        self.iter().take(length).collect()
    }

    /// The endless sequence, bit after bit, from the register with every stage at one: a
    /// whole period of the long sequences without storing it.
    pub fn iter(self) -> O150Bits {
        O150Bits { sequence: self, register: (1u32 << self.stages()) - 1 }
    }
}

/// The bits of an [`O150Sequence`] ([`O150Sequence::iter`]).
#[derive(Clone, Debug)]
pub struct O150Bits {
    sequence: O150Sequence,
    /// Bit $i - 1$ holds the stage $Q_{i}$.
    register: u32,
}

impl Iterator for O150Bits {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        let (n, k) = (self.sequence.stages(), self.sequence.feedback());
        let q = self.register;
        let last = (q >> (n - 1)) & 1;
        let out = match self.sequence {
            // Q_6 to Q_19, the next 14 outputs of Sec. 5.5
            O150Sequence::Prbs20Zs => last | u32::from(q & (((1u32 << 14) - 1) << 5) == 0),
            _ => last,
        };
        let fb = last ^ ((q >> (k - 1)) & 1);
        self.register = ((q << 1) | fb) & ((1u32 << n) - 1);
        Some(out as u8 ^ u8::from(self.sequence.inverted()))
    }
}

/// Maxwell–Boltzmann probabilities `p_i ∝ exp(-λ |x_i|²)` over the points of `c`.
///
/// `λ = 0` gives the uniform distribution; larger `λ` favour the low-energy points
/// (probabilistic constellation shaping).
pub fn maxwell_boltzmann(c: &Constellation, lambda: f64) -> Vec<f64> {
    let w: Vec<f64> = c.points().iter().map(|p| (-lambda * p.norm_sqr()).exp()).collect();
    let s: f64 = w.iter().sum();
    w.iter().map(|v| v / s).collect()
}

/// `n` random symbols of `c` drawn with the probabilities `px` (uniform if `None`,
/// normalised to a sum of 1 otherwise, as [`Constellation::priors`]).
///
/// Returns the indices (Gray labels) and the symbols.
///
/// # Errors
/// As [`Constellation::priors`].
pub fn symbols(c: &Constellation, n: usize, px: Option<&[f64]>, rng: &mut Rng) -> Result<(Vec<usize>, Vec<C64>)> {
    let m = c.order();
    let idx: Vec<usize> = if px.is_none() {
        (0..n).map(|_| rng.below(m)).collect()
    } else {
        let mut cdf = Vec::with_capacity(m);
        let mut acc = 0.0;
        for v in c.priors(px)? {
            acc += v;
            cdf.push(acc);
        }
        (0..n)
            .map(|_| {
                let u = rng.uniform() * acc;
                cdf.partition_point(|&c| c <= u).min(m - 1)
            })
            .collect()
    };
    let x = idx.iter().map(|&i| c.points()[i]).collect();
    Ok((idx, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prbs_periods_and_balance() {
        for &(n, _) in &PRBS_TAPS[..4] {
            let period = (1usize << n) - 1;
            let p = prbs(n, 2 * period, 0xACE1).unwrap();
            assert_eq!(&p[..period], &p[period..], "order {n}");
            let ones = p[..period].iter().filter(|&&v| v == 1).count();
            assert_eq!(ones, period.div_ceil(2), "order {n}");
        }
        assert!(prbs(8, 10, 1).is_err());
        assert_eq!(prbs(29, 10, 1).unwrap().len(), 10);
    }

    #[test]
    fn the_511_bit_sequence_begins_with_its_nine_ones() {
        // O.150 Sec. 5.1: "The sequence begins with the first ONE of 9 consecutive ONEs"
        let b = O150Sequence::Prbs9.bits(2 * 511);
        assert_eq!(&b[..9], &[1; 9]);
        assert_eq!(b[510], 0, "the bit before the run (one period earlier) is a zero");
        assert_eq!(&b[..511], &b[511..]);
    }

    #[test]
    fn an_inverted_sequence_is_the_negated_register_output() {
        // same register as prbs() with the polynomial x^15 + x^14 + 1: a shifted phase
        let t = O150Sequence::Prbs15.bits(32_767);
        assert_eq!(&t[..15], &[0; 15]);
        assert_eq!(t.iter().filter(|&&v| v == 0).count(), 16_384, "the 2^14 ones of the m-sequence are zeros");
        let p = prbs(15, 2 * 32_767, 1).unwrap();
        let negated: Vec<u8> = t.iter().map(|v| v ^ 1).collect();
        assert!((0..32_767).any(|d| p[d..d + 32_767] == negated[..]), "a phase of the same m-sequence");
    }

    #[test]
    fn the_zero_suppression_differs_from_the_plain_sequence_of_the_same_length() {
        // O.150, note of Sec. 5.4: the sequences of 5.4 and 5.5 are not identical
        let (a, b) = (O150Sequence::Prbs20.bits(4096), O150Sequence::Prbs20Zs.bits(4096));
        assert_ne!(a, b);
        let longest = b.split(|&v| v == 1).map(<[u8]>::len).max().unwrap();
        assert!(longest <= 14);
    }

    #[test]
    fn the_names_round_trip() {
        for s in O150Sequence::ALL {
            assert_eq!(O150Sequence::from_name(s.name()).unwrap(), s);
            assert_eq!(s.period(), (1u64 << s.stages()) - 1);
        }
        assert_eq!(O150Sequence::from_name("PRBS23").unwrap(), O150Sequence::Prbs23);
        assert!(O150Sequence::from_name("prbs7").is_err());
    }

    #[test]
    fn shaped_symbols_follow_the_distribution() {
        let c = Constellation::qam(16).unwrap().normalized();
        let px = maxwell_boltzmann(&c, 1.0);
        let (idx, _) = symbols(&c, 200_000, Some(&px), &mut Rng::new(5)).unwrap();
        for i in [0usize, 5, 15] {
            let f = idx.iter().filter(|&&k| k == i).count() as f64 / idx.len() as f64;
            assert!((f - px[i]).abs() < 0.005, "point {i}: {f} vs {}", px[i]);
        }
    }
}
