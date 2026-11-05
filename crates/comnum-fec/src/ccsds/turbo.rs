//! The turbo codes of CCSDS 131.0-B-5 Sec. 6, [`CcsdsTurbo`], and their permutation.

use crate::block::BlockCode;
use crate::block::SoftDecode;
use crate::block::container_traits;
use crate::block::expect_len;
use crate::conv::Trellis;
use crate::interleaving::Interleaver;
use crate::turbo::TurboDecoder;
use comnum_core::Error;
use comnum_core::Result;
use comnum_core::ensure;
use comnum_core::signal::hard_decisions;

/// The rates of the turbo codes (Sec. 6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TurboRate {
    /// 1/2.
    R1_2,
    /// 1/3.
    R1_3,
    /// 1/4.
    R1_4,
    /// 1/6.
    R1_6,
}

impl TurboRate {
    /// The coded bits per information bit, $1/r$ (the inverse of the nominal rate).
    pub fn bits_per_info_bit(self) -> usize {
        match self {
            TurboRate::R1_2 => 2,
            TurboRate::R1_3 => 3,
            TurboRate::R1_4 => 4,
            TurboRate::R1_6 => 6,
        }
    }

    /// The name of the rate, `1/2` to `1/6`.
    pub fn name(self) -> &'static str {
        match self {
            TurboRate::R1_2 => "1/2",
            TurboRate::R1_3 => "1/3",
            TurboRate::R1_4 => "1/4",
            TurboRate::R1_6 => "1/6",
        }
    }

    /// The forward vectors of encoders a and b (octal, the current register input the
    /// most significant): $G_{1} = 33$, $G_{2} = 25$, $G_{3} = 37$ (Sec. 6.3 h)).
    fn forwards(self) -> (&'static [u32], &'static [u32]) {
        match self {
            TurboRate::R1_2 | TurboRate::R1_3 => (&[0o33], &[0o33]),
            TurboRate::R1_4 => (&[0o25, 0o37], &[0o33]),
            TurboRate::R1_6 => (&[0o33, 0o25, 0o37], &[0o33, 0o37]),
        }
    }
}

impl std::str::FromStr for TurboRate {
    type Err = Error;

    /// `1/2`, `1/3`, `1/4` or `1/6`.
    fn from_str(s: &str) -> Result<TurboRate> {
        match s.trim() {
            "1/2" => Ok(TurboRate::R1_2),
            "1/3" => Ok(TurboRate::R1_3),
            "1/4" => Ok(TurboRate::R1_4),
            "1/6" => Ok(TurboRate::R1_6),
            _ => Err(Error::Config(format!("the CCSDS turbo rate must be 1/2, 1/3, 1/4 or 1/6, got {s:?}"))),
        }
    }
}

/// The information block lengths of the turbo codes and their $k_{2}$ (Tables 6-1 and
/// 6-3; $k_{1} = 8$).
pub const TURBO_LENGTHS: [usize; 4] = [1784, 3568, 7136, 8920];

/// The permutation of the turbo codes of `k` bits (Sec. 6.3 g)): the bit $s$ read by the
/// second encoder is the bit $\pi(s)$ of the block, $k = k_{1} k_{2}$, $k_{1} = 8$,
/// $m = (s - 1) \bmod 2$, $i = \lfloor (s - 1)/(2 k_{2}) \rfloor$, $j = \lfloor (s - 1)/2
/// \rfloor - i k_{2}$, $t = (19 i + 1) \bmod (k_{1}/2)$, $q = t \bmod 8 + 1$,
/// $c = (p_{q} j + 21 m) \bmod k_{2}$, $\pi(s) = 2 (t + c k_{1}/2 + 1) - m$, $p = (31, 37, 43,
/// 47, 53, 59, 61, 67)$. As an [`Interleaver`], output $s - 1$ takes input $\pi(s) - 1$.
///
/// # Errors
/// [`Error::Config`] for another $k$.
pub fn turbo_permutation(k: usize) -> Result<Interleaver> {
    const PRIMES: [usize; 8] = [31, 37, 43, 47, 53, 59, 61, 67];
    ensure(TURBO_LENGTHS.contains(&k), || format!("the CCSDS turbo codes have k = 1784, 3568, 7136 or 8920, got {k}"))?;
    let (k1, k2) = (8, k / 8);
    let perm = (1..=k)
        .map(|s| {
            let m = (s - 1) % 2;
            let i = (s - 1) / (2 * k2);
            let j = (s - 1) / 2 - i * k2;
            let t = (19 * i + 1) % (k1 / 2);
            let q = t % 8 + 1;
            let c = (PRIMES[q - 1] * j + 21 * m) % k2;
            2 * (t + c * k1 / 2 + 1) - m - 1
        })
        .collect();
    Interleaver::from_permutation(perm)
}

/// A turbo code of CCSDS (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct CcsdsTurbo {
    rate: TurboRate,
    /// The trellises of encoders a and b (the systematic output first).
    a: Trellis,
    b: Trellis,
    interleaver: Interleaver,
    /// The decoder settings.
    pub decoder: TurboDecoder,
}

/// The termination steps of a constituent encoder.
const TURBO_TAIL: usize = 4;

impl CcsdsTurbo {
    /// The code of `k` information bits (1784, 3568, 7136 or 8920) and rate `rate`.
    ///
    /// # Errors
    /// [`Error::Config`] for another $k$.
    pub fn new(k: usize, rate: TurboRate) -> Result<CcsdsTurbo> {
        let interleaver = turbo_permutation(k)?;
        let (fa, fb) = rate.forwards();
        let a = Trellis::recursive_parities(5, 0o23, fa)?;
        let b = Trellis::recursive_parities(5, 0o23, fb)?;
        Ok(CcsdsTurbo { rate, a, b, interleaver, decoder: TurboDecoder::default() })
    }

    /// The rate.
    pub fn rate(&self) -> TurboRate {
        self.rate
    }

    /// The permutation of the second encoder.
    pub fn interleaver(&self) -> &Interleaver {
        &self.interleaver
    }

    /// The outputs of bit time `t` (from 0) as indices into the outputs of encoders a and b:
    /// `(encoder, output)`, encoder 0 = a, output 0 = systematic.
    fn slots(&self, t: usize) -> &'static [(u8, u8)] {
        match self.rate {
            TurboRate::R1_2 if t % 2 == 0 => &[(0, 0), (0, 1)],
            TurboRate::R1_2 => &[(0, 0), (1, 1)],
            TurboRate::R1_3 => &[(0, 0), (0, 1), (1, 1)],
            TurboRate::R1_4 => &[(0, 0), (0, 1), (0, 2), (1, 1)],
            TurboRate::R1_6 => &[(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2)],
        }
    }
}

impl BlockCode for CcsdsTurbo {
    fn k(&self) -> usize {
        self.interleaver.len()
    }

    /// $(k + 4)/r$ (Table 6-2).
    fn n(&self) -> usize {
        (self.k() + TURBO_TAIL) * self.rate.bits_per_info_bit()
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k(), "bits")?;
        let ca = self.a.encode(info, true);
        let cb = self.b.encode(&self.interleaver.interleave(info), true);
        let (na, nb) = (self.a.n_out(), self.b.n_out());
        let mut out = Vec::with_capacity(self.n());
        for t in 0..self.k() + TURBO_TAIL {
            for &(e, o) in self.slots(t) {
                out.push(if e == 0 { ca[t * na + usize::from(o)] } else { cb[t * nb + usize::from(o)] });
            }
        }
        Ok(out)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        Ok(hard_decisions(&self.app(llr, None)?))
    }
}

impl SoftDecode for CcsdsTurbo {
    /// The iterative decoder of [`crate::turbo`] on the two constituent codes: the
    /// a posteriori LLRs of the second decoder after the last iteration, deinterleaved;
    /// the a priori LLRs add to the systematic channel LLRs.
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        let k = self.k();
        expect_len(out.len(), k, "a posteriori LLRs")?;
        if let Some(a) = apriori {
            expect_len(a.len(), k, "a priori LLRs")?;
        }
        let steps = k + TURBO_TAIL;
        let (na, nb) = (self.a.n_out(), self.b.n_out());
        // the channel LLRs of every output of both encoders, zero where not sent
        let (mut la, mut lb) = (vec![0.0; steps * na], vec![0.0; steps * nb]);
        let mut pos = 0;
        for t in 0..steps {
            for &(e, o) in self.slots(t) {
                if e == 0 {
                    la[t * na + usize::from(o)] = llr[pos];
                } else {
                    lb[t * nb + usize::from(o)] = llr[pos];
                }
                pos += 1;
            }
        }
        if let Some(a) = apriori {
            for (t, v) in a.iter().enumerate() {
                la[t * na] += v;
            }
        }
        let sys: Vec<f64> = (0..k).map(|t| la[t * na]).collect();
        let sys2 = self.interleaver.interleave(&sys);
        for (t, &v) in sys2.iter().enumerate() {
            lb[t * nb] = v;
        }
        if self.decoder.iterations == 0 {
            out.copy_from_slice(&sys);
            return Ok(0);
        }
        crate::turbo::iterate(
            &self.a,
            &self.b,
            &la,
            &lb,
            &sys,
            &sys2,
            self.interleaver.perm(),
            steps,
            self.decoder,
            out,
        )
    }
}

container_traits!(CcsdsTurbo);
