//! CCSDS 131.0-B-5 (*TM Synchronization and Channel Coding*): the attached sync markers,
//! the pseudo-randomizers, the convolutional code and the channel access data units; the
//! Reed–Solomon, LDPC and turbo codes are in `comnum_fec::ccsds`.
//!
//! # Model
//!
//! - **Markers** (Sec. 9.3, Table 8-1): the 32-bit attached sync marker [`ASM`] =
//!   `1ACFFC1D` (uncoded, convolutional, Reed–Solomon, concatenated and rate-7/8 LDPC
//!   links, and the code sync marker of the 7/8 LDPC code), the 64-bit marker
//!   [`ASM_64`] = `034776C7272895B0` (turbo rate 1/2, AR4JA LDPC, and their code sync
//!   marker), the turbo markers of rates 1/3, 1/4, 1/6 ([`turbo_asm`]), and the 32-bit
//!   marker of the embedded data stream [`ASM_EMBEDDED`] = `352EF853`; bit 0, the first
//!   transmitted, is the most significant bit of the first byte.
//! - **Pseudo-randomizers** (Sec. 10.4, [`Randomizer`]): the sequence of 131 071 bits of
//!   $h(x) = x^{17} + x^{14} + 1$ started at `11000111000111000` (Issue 5), and the legacy
//!   sequence of 255 bits of $h(x) = x^{8} + x^{7} + x^{5} + x^{3} + 1$ started at all ones,
//!   added to every bit of a codeblock after its marker and restarted at every codeblock.
//! - **Convolutional code** (Sec. 3, [`Convolutional`]): constraint length 7, $G_{1} =
//!   171_{8}$ then $G_{2} = 133_{8}$ at every bit time, the output of $G_{2}$ inverted at
//!   rate 1/2 (Sec. 3.3.2) and not at the punctured rates 2/3 to 7/8 of Table 3-1 (the
//!   patterns of DVB-S, [`ConvolutionalRate`]); a continuous code, started here in state 0.
//! - **CADU** ([`cadu`], Table 9-1): the marker then the randomized codeblock.
//!
//! # Examples
//! ```
//! use comnum_systems::ccsds::{ASM, Randomizer, cadu};
//! // Sec. 10.4.3 NOTE 2: the first 40 bits of the sequence of 131 071 bits
//! let s = Randomizer::Long.sequence(40);
//! assert_eq!(&s[..8], &[0, 0, 0, 1, 1, 1, 0, 0]);
//! // a CADU: the 32 bits of 1ACFFC1D, then the randomized codeblock
//! let frame = vec![0u8; 64];
//! let c = cadu(&ASM, &frame, Some(Randomizer::Long));
//! assert_eq!(&c[..8], &[0, 0, 0, 1, 1, 0, 1, 0]);          // 0x1A
//! assert_eq!(&c[32..], &Randomizer::Long.sequence(64)[..]);
//! ```
//!
//! # References
//!
//! 1. CCSDS 131.0-B-5, *TM Synchronization and Channel Coding*, Recommended Standard
//!    (Blue Book), Issue 5, September 2023, Secs. 3, 9 and 10, Tables 3-1, 8-1 and 9-1.

pub use crate::dvb::InnerRate as ConvolutionalRate;
use comnum_core::{Error, Result};
use comnum_fec::ccsds::TurboRate;
use comnum_fec::conv::{Trellis, viterbi};
use comnum_fec::puncture::Puncturer;
use comnum_math::sequences::lfsr;

/// The attached sync marker of 32 bits, `1ACFFC1D` (Figure 9-1).
pub const ASM: [u8; 4] = [0x1A, 0xCF, 0xFC, 0x1D];
/// The marker of 64 bits, `034776C7272895B0`: turbo rate 1/2 and the AR4JA LDPC codes
/// (Figure 9-2), and the code sync marker of the AR4JA codes (Table 8-1).
pub const ASM_64: [u8; 8] = [0x03, 0x47, 0x76, 0xC7, 0x27, 0x28, 0x95, 0xB0];
/// The marker of the turbo code of rate 1/3, 96 bits (Figure 9-3).
pub const ASM_TURBO_1_3: [u8; 12] = [0x25, 0xD5, 0xC0, 0xCE, 0x89, 0x90, 0xF6, 0xC9, 0x46, 0x1B, 0xF7, 0x9C];
/// The marker of the turbo code of rate 1/4, 128 bits (Figure 9-4).
pub const ASM_TURBO_1_4: [u8; 16] =
    [0x03, 0x47, 0x76, 0xC7, 0x27, 0x28, 0x95, 0xB0, 0xFC, 0xB8, 0x89, 0x38, 0xD8, 0xD7, 0x6A, 0x4F];
/// The marker of the turbo code of rate 1/6, 192 bits (Figure 9-5).
pub const ASM_TURBO_1_6: [u8; 24] = [
    0x25, 0xD5, 0xC0, 0xCE, 0x89, 0x90, 0xF6, 0xC9, 0x46, 0x1B, 0xF7, 0x9C, 0xDA, 0x2A, 0x3F, 0x31, 0x76, 0x6F, 0x09,
    0x36, 0xB9, 0xE4, 0x08, 0x63,
];
/// The marker of the embedded data stream, `352EF853` (Figure 9-6).
pub const ASM_EMBEDDED: [u8; 4] = [0x35, 0x2E, 0xF8, 0x53];

/// The attached sync marker of a turbo code (Sec. 9.3).
pub fn turbo_asm(rate: TurboRate) -> &'static [u8] {
    match rate {
        TurboRate::R1_2 => &ASM_64,
        TurboRate::R1_3 => &ASM_TURBO_1_3,
        TurboRate::R1_4 => &ASM_TURBO_1_4,
        TurboRate::R1_6 => &ASM_TURBO_1_6,
    }
}

/// The pseudo-randomizers of Sec. 10.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Randomizer {
    /// $h(x) = x^{17} + x^{14} + 1$, period 131 071, started at `11000111000111000`
    /// (the cell $X^{16}$ the leftmost character).
    Long,
    /// $h(x) = x^{8} + x^{7} + x^{5} + x^{3} + 1$, period 255, started at all ones (legacy).
    Short,
}

impl Randomizer {
    /// The period of the sequence.
    pub fn period(self) -> usize {
        match self {
            Randomizer::Long => 131_071,
            Randomizer::Short => 255,
        }
    }

    /// The first `n` bits of the sequence: $s_{n+17} = s_{n} \oplus s_{n+14}$ from
    /// $(s_{0}, \dots, s_{16}) = 00011100011100011$, or $s_{n+8} = s_{n} \oplus s_{n+3} \oplus
    /// s_{n+5} \oplus s_{n+7}$ from eight ones.
    pub fn sequence(self, n: usize) -> Vec<u8> {
        match self {
            Randomizer::Long => lfsr(&[0, 14], 17, &[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1], n),
            Randomizer::Short => lfsr(&[0, 3, 5, 7], 8, &[1; 8], n),
        }
    }

    /// The bits of a codeblock randomized (or derandomized): the sequence added from its
    /// first bit (Sec. 10.3).
    pub fn apply(self, bits: &[u8]) -> Vec<u8> {
        bits.iter().zip(self.sequence(bits.len())).map(|(&b, s)| (b & 1) ^ s).collect()
    }

    /// The LLRs of a randomized codeblock derandomized: negated where the sequence is one
    /// (Sec. 10.3, NOTE 2).
    pub fn apply_llr(self, llr: &[f64]) -> Vec<f64> {
        llr.iter().zip(self.sequence(llr.len())).map(|(&l, s)| if s == 1 { -l } else { l }).collect()
    }
}

/// The bits of bytes, the most significant first.
fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |k| (b >> k) & 1)).collect()
}

/// A channel access data unit (Table 9-1): the bits of the marker `asm`, then the bits of
/// the codeblock (or codeword, or frame) `block`, randomized if `randomizer` is given; the
/// marker is never randomized.
pub fn cadu(asm: &[u8], block: &[u8], randomizer: Option<Randomizer>) -> Vec<u8> {
    let mut out = bytes_to_bits(asm);
    match randomizer {
        Some(r) => out.extend(r.apply(block)),
        None => out.extend(block.iter().map(|b| b & 1)),
    }
    out
}

/// The convolutional code of CCSDS (Sec. 3): constraint length 7, $G_{1} = 171_{8}$ then
/// $G_{2} = 133_{8}$, $G_{2}$ inverted at rate 1/2, punctured by Table 3-1 at the other
/// rates (see the module).
#[derive(Clone, Debug)]
pub struct Convolutional {
    rate: ConvolutionalRate,
    trellis: Trellis,
    puncturer: Puncturer,
}

impl Convolutional {
    /// The code of rate `rate` (1/2, 2/3, 3/4, 5/6, 7/8).
    ///
    /// # Errors
    /// Never: the codes are those of the standard.
    pub fn new(rate: ConvolutionalRate) -> Result<Convolutional> {
        Ok(Convolutional { rate, trellis: crate::dvb::inner_trellis()?, puncturer: rate.puncturer()? })
    }

    /// The rate.
    pub fn rate(&self) -> ConvolutionalRate {
        self.rate
    }

    /// The channel symbols of `bits` from state 0 (no tail): $C_{1}(t)$,
    /// $\overline{C_{2}(t)}$ at rate 1/2, the sequences of Table 3-1 otherwise.
    ///
    /// # Errors
    /// Never for whole bits.
    pub fn encode(&self, bits: &[u8]) -> Result<Vec<u8>> {
        let c = self.trellis.encode(bits, false);
        if self.rate == ConvolutionalRate::R1_2 {
            Ok(c.iter().enumerate().map(|(i, &b)| if i % 2 == 1 { b ^ 1 } else { b }).collect())
        } else {
            self.puncturer.puncture(&c)
        }
    }

    /// The `n_bits` input bits of the channel LLRs of [`Convolutional::encode`], by Viterbi
    /// decoding (soft decisions) of the whole stream, unterminated.
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of LLRs is not that of `n_bits` bits.
    pub fn decode(&self, llr: &[f64], n_bits: usize) -> Result<Vec<u8>> {
        let full = if self.rate == ConvolutionalRate::R1_2 {
            if llr.len() != 2 * n_bits {
                return Err(Error::Shape(format!("{} LLRs for {n_bits} bits at rate 1/2", llr.len())));
            }
            llr.iter().enumerate().map(|(i, &l)| if i % 2 == 1 { -l } else { l }).collect()
        } else {
            self.puncturer.depuncture(llr, 2 * n_bits)?
        };
        viterbi(&self.trellis, &full, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    #[test]
    fn the_randomizers_start_as_note_2_of_sec_10_4_3() {
        let bits = |s: &str| s.chars().filter(|c| !c.is_whitespace()).map(|c| u8::from(c == '1')).collect::<Vec<u8>>();
        assert_eq!(Randomizer::Long.sequence(40), bits("0001 1100 0111 0001 1011 1001 0001 1011 1010 1001"));
        assert_eq!(Randomizer::Short.sequence(40), bits("1111 1111 0100 1000 0000 1110 1100 0000 1001 1010"));
        for r in [Randomizer::Long, Randomizer::Short] {
            let p = r.period();
            let s = r.sequence(p + 40);
            assert_eq!(&s[p..], &s[..40], "period {p}");
        }
    }

    #[test]
    fn the_markers_are_those_of_the_figures() {
        // Figure 9-1: 0001 1010 1100 1111 1111 1100 0001 1101
        let b: String = bytes_to_bits(&ASM).iter().map(|v| char::from(b'0' + v)).collect();
        assert_eq!(b, "00011010110011111111110000011101");
        // the second halves of the rate-1/4 and rate-1/6 markers complement the first ones
        assert!(ASM_TURBO_1_4[..8].iter().zip(&ASM_TURBO_1_4[8..]).all(|(a, b)| a ^ b == 0xFF));
        assert!(ASM_TURBO_1_6[..12].iter().zip(&ASM_TURBO_1_6[12..]).all(|(a, b)| a ^ b == 0xFF));
        assert_eq!(turbo_asm(TurboRate::R1_2), &ASM_64);
    }

    #[test]
    fn the_convolutional_code_inverts_g2_at_rate_one_half() {
        let code = Convolutional::new(ConvolutionalRate::R1_2).unwrap();
        // a single one: the impulse responses of G1 = 1111001 and G2 = 1011011, G2 inverted
        let mut bits = vec![0u8; 7];
        bits[0] = 1;
        let c = code.encode(&bits).unwrap();
        let c1: Vec<u8> = c.iter().step_by(2).copied().collect();
        let c2: Vec<u8> = c.iter().skip(1).step_by(2).map(|b| b ^ 1).collect();
        assert_eq!(c1, [1, 1, 1, 1, 0, 0, 1]);
        assert_eq!(c2, [1, 0, 1, 1, 0, 1, 1]);
    }

    #[test]
    fn every_rate_round_trips() {
        let mut rng = Rng::new(1);
        let bits: Vec<u8> = (0..700).map(|_| rng.bit()).collect();
        for rate in ConvolutionalRate::ALL {
            let code = Convolutional::new(rate).unwrap();
            let c = code.encode(&bits).unwrap();
            let llr: Vec<f64> = c.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
            let d = code.decode(&llr, bits.len()).unwrap();
            // the last bits of an unterminated stream are less protected: compare the body
            assert_eq!(&d[..680], &bits[..680], "rate {}", rate.name());
        }
    }
}
