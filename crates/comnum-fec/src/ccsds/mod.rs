//! The channel codes of CCSDS 131.0-B-5 (*TM Synchronization and Channel Coding*): the
//! Reed–Solomon codes in the dual basis with their interleaving and virtual fill, the
//! (8160, 7136) LDPC code, the AR4JA LDPC codes and the turbo codes.
//!
//! # Reed–Solomon codes (Sec. 4)
//!
//! [`CcsdsRs`] is the (255, 223) ($E = 16$) or (255, 239) ($E = 8$) code over
//! $\mathrm{GF}(2^{8})$ of the field polynomial $F(x) = x^{8} + x^{7} + x^{2} + x + 1$ and the
//! generator $g(x) = \prod_{j=128-E}^{127+E} (x - \alpha^{11 j})$ (Secs. 4.3.3, 4.3.4), with
//! the symbols in the dual (Berlekamp) basis (Sec. 4.3.9): the octets of a frame are the
//! dual-basis symbols, $z_{0}$ their most significant (first transmitted) bit, and
//! [`to_dual_basis`], [`from_dual_basis`] are the matrices $T_{\alpha\ell}$ and
//! $T_{\alpha\ell}^{-1}$ that take them to and from the conventional basis of the encoder
//! (Annex F). $I$ codewords are interleaved symbol by symbol, $I \in \{1, 2, 3, 4, 5, 8\}$
//! (Sec. 4.3.5, 4.4.1), each shortened by a virtual fill of $q$ leading zero symbols that
//! are neither encoded nor sent (Sec. 4.3.7): a frame of $(255 - 2E - q) I$ octets, a
//! codeblock of $(255 - q) I$ octets.
//!
//! # LDPC codes (Sec. 7)
//!
//! - [`CcsdsLdpc::c2`]: the (8160, 7136) code of rate 223/255 (Sec. 7.3), built on the
//!   (8176, 7156) quasi-cyclic code whose parity-check matrix is a $2 \times 16$ array of
//!   $511 \times 511$ circulants of weight 2 (Table 7-1); its encoder is the systematic
//!   generator of the (8176, 7154) subcode, whose $28$ circulants are given by their first
//!   rows (Annex C, Table C-1): 18 virtual zeros are prepended to the 7136 information bits,
//!   the 1022 parity bits follow them, the 18 zeros are dropped and two zeros appended.
//! - [`CcsdsLdpc::ar4ja`]: the nine AR4JA codes of rates 1/2, 2/3, 4/5 and $k = 1024, 4096,
//!   16384$ (Sec. 7.4): $H$ is $3 \times (K + 3)$ blocks of $M \times M$ sums of identities and
//!   permutations $\Pi_{k}$, $\pi_{k}(i) = \frac{M}{4}((\theta_{k} + \lfloor 4i/M \rfloor) \bmod 4) +
//!   (\phi_{k}(\lfloor 4i/M \rfloor, M) + i) \bmod \frac{M}{4}$ (Tables 7-3, 7-4), $K = 2, 4, 8$;
//!   the codeword $[u\ p]$ satisfies $H [u\ p]^{\mathrm{T}} = 0$ (the generator $[I\ W]$,
//!   $W = (P^{-1} Q)^{\mathrm{T}}$, Sec. 7.4.3) and its last $M$ bits are punctured.
//!   The encoder solves $P p = Q u$ by the block structure of $P$, with the inverse of one
//!   $M \times M$ matrix.
//!
//! Both decode by the belief propagation of [`crate::ldpc`], the virtual zeros known
//! ([`KNOWN_BIT_LLR`](comnum_core::signal::KNOWN_BIT_LLR)) and the punctured bits at a zero LLR.
//!
//! # Turbo codes (Sec. 6)
//!
//! [`CcsdsTurbo`]: two 16-state recursive codes of backward vector $G_{0} = 10011$
//! ($w_{t} = u_{t} \oplus w_{t-3} \oplus w_{t-4}$) and forward vectors $G_{1} = 11011$,
//! $G_{2} = 10101$, $G_{3} = 11111$, on $k = 1784, 3568, 7136, 8920$ bits, the second
//! encoder fed by the permutation of Sec. 6.3 g) ([`turbo_permutation`]). Both encoders
//! are terminated by 4 steps of feedback input; the outputs of every bit time leave in the
//! order of Sec. 6.3 j): rate 1/3 $(0a, 1a, 1b)$, rate 1/4 $(0a, 2a, 3a, 1b)$, rate 1/6
//! $(0a, 1a, 2a, 3a, 1b, 3b)$, rate 1/2 $(0a, 1a)$ and $(0a, 1b)$ alternately: $n = (k + 4)/r$.
//! The decoder is the iterative BCJR decoder of [`crate::turbo`].
//!
//! # Examples
//! ```
//! use comnum_fec::block::BlockCode;
//! use comnum_fec::ccsds::{CcsdsRs, from_dual_basis, to_dual_basis};
//! // Annex F, example 2: [u] = 01011001 is [z] = 11101000 in the dual basis
//! assert_eq!(to_dual_basis(0b0101_1001), 0b1110_1000);
//! assert_eq!(from_dual_basis(0b1110_1000), 0b0101_1001);
//! // (255, 223), interleaving depth 5: a frame of 1115 octets, 1280 check bits (Sec. 4.3.8)
//! let rs = CcsdsRs::new(16, 5, 0)?;
//! assert_eq!((rs.frame_len(), rs.codeblock_len()), (1115, 1275));
//! assert_eq!(rs.n() - rs.k(), 1280);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. CCSDS 131.0-B-5, *TM Synchronization and Channel Coding*, Recommended Standard
//!    (Blue Book), Issue 5, September 2023, Secs. 4, 6 and 7, Annexes C, F and G.

use crate::block::{BlockCode, container_traits, expect_len};
use crate::rs::ReedSolomon;
use comnum_core::signal::hard_decisions;
use comnum_core::{Result, ensure};

mod ldpc;
mod turbo;

pub use ldpc::{Ar4jaRate, CcsdsLdpc};
pub use turbo::{CcsdsTurbo, TURBO_LENGTHS, TurboRate, turbo_permutation};

/// $T_{\alpha\ell}$ of Sec. 4.3.9: row $r$ is the dual-basis representation of
/// $\alpha^{7-r}$, the bit $z_{0}$ the most significant.
const DUAL_BASIS: [u8; 8] = [0x8D, 0xEF, 0xEC, 0x86, 0xFA, 0x99, 0xAF, 0x7B];
/// $T_{\alpha\ell}^{-1}$ of Sec. 4.3.9: row $r$ multiplies $z_{r}$.
const DUAL_BASIS_INV: [u8; 8] = [0xC5, 0x42, 0x2E, 0xFD, 0xF0, 0x79, 0xAC, 0xCC];
/// The field polynomial $F(x) = x^{8} + x^{7} + x^{2} + x + 1$ (Sec. 4.3.3).
const RS_FIELD: u32 = 0x187;
/// The interleaving depths of Sec. 4.3.5.1.
pub const RS_DEPTHS: [usize; 6] = [1, 2, 3, 4, 5, 8];

/// The dual-basis symbol $[z_{0} \dots z_{7}] = [u_{7} \dots u_{0}]\, T_{\alpha\ell}$ of the
/// conventional symbol `u` ($u_{7}$, the coefficient of $\alpha^{7}$, its most significant
/// bit; $z_{0}$ the most significant bit of the result), Sec. 4.3.9.
pub fn to_dual_basis(u: u8) -> u8 {
    (0..8).filter(|r| (u >> (7 - r)) & 1 == 1).fold(0, |z, r| z ^ DUAL_BASIS[r])
}

/// The conventional symbol $[u_{7} \dots u_{0}] = [z_{0} \dots z_{7}]\, T_{\alpha\ell}^{-1}$ of
/// the dual-basis symbol `z`, Sec. 4.3.9.
pub fn from_dual_basis(z: u8) -> u8 {
    (0..8).filter(|r| (z >> (7 - r)) & 1 == 1).fold(0, |u, r| u ^ DUAL_BASIS_INV[r])
}

/// An interleaved Reed–Solomon codeblock of CCSDS (see the module): the code of
/// correction capability $E$, the interleaving depth $I$ and the virtual fill of $q$
/// symbols per codeword.
#[derive(Clone, Debug, PartialEq)]
pub struct CcsdsRs {
    rs: ReedSolomon,
    depth: usize,
    fill: usize,
}

impl CcsdsRs {
    /// The codeblock of correction capability `e` (8 or 16 symbols per codeword), depth
    /// `depth` (1, 2, 3, 4, 5 or 8) and virtual fill `fill` ($q$ symbols per codeword,
    /// $Q = q I$ in all; 0 for the full length).
    ///
    /// # Errors
    /// [`Error::Config`](comnum_core::Error::Config) for another $E$ or $I$, or a fill that leaves no information.
    pub fn new(e: usize, depth: usize, fill: usize) -> Result<CcsdsRs> {
        ensure(e == 8 || e == 16, || format!("the CCSDS Reed-Solomon codes correct E = 8 or 16 symbols, got {e}"))?;
        ensure(RS_DEPTHS.contains(&depth), || {
            format!("the interleaving depth must be 1, 2, 3, 4, 5 or 8, got {depth}")
        })?;
        ensure(fill < 255 - 2 * e, || format!("a virtual fill of {fill} symbols leaves no information"))?;
        let rs = ReedSolomon::with_field(8, RS_FIELD, 128 - e, 11, e, 255 - fill)?;
        Ok(CcsdsRs { rs, depth, fill })
    }

    /// The correction capability $E$ of a codeword.
    pub fn e(&self) -> usize {
        self.rs.t()
    }

    /// The interleaving depth $I$.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// The virtual fill $q$ of a codeword.
    pub fn fill(&self) -> usize {
        self.fill
    }

    /// The conventional code of a codeword (the field of $F(x)$, the roots
    /// $\alpha^{11 j}$, shortened by the fill).
    pub fn code(&self) -> &ReedSolomon {
        &self.rs
    }

    /// Octets of a frame, $(255 - 2E - q) I$.
    pub fn frame_len(&self) -> usize {
        self.rs.k_symbols() * self.depth
    }

    /// Octets of a codeblock, $(255 - q) I$.
    pub fn codeblock_len(&self) -> usize {
        self.rs.n_symbols() * self.depth
    }

    /// The codeblock of a frame (dual-basis octets): the frame, then the $2 E I$ check
    /// symbols, codeword $i$ holding the symbols $i, i + I, i + 2I, \dots$ (Sec. 4.4.1).
    ///
    /// # Errors
    /// [`Error::Shape`](comnum_core::Error::Shape) unless the frame has [`CcsdsRs::frame_len`] octets.
    pub fn encode_bytes(&self, frame: &[u8]) -> Result<Vec<u8>> {
        expect_len(frame.len(), self.frame_len(), "octets")?;
        let (k, i_depth) = (self.rs.k_symbols(), self.depth);
        let parity = 2 * self.e();
        let mut out = frame.to_vec();
        out.resize(self.codeblock_len(), 0);
        for i in 0..i_depth {
            let msg: Vec<u16> = (0..k).map(|t| u16::from(from_dual_basis(frame[t * i_depth + i]))).collect();
            let word = self.rs.encode_symbols(&msg)?;
            for (t, &s) in word[k..].iter().enumerate() {
                out[(k + t) * i_depth + i] = to_dual_basis(s as u8);
            }
            debug_assert_eq!(word.len(), k + parity);
        }
        Ok(out)
    }

    /// Decode a received codeblock: the frame and the number of corrected symbols
    /// (`None` if a codeword could not be decoded; its frame symbols are then those
    /// received).
    ///
    /// # Errors
    /// [`Error::Shape`](comnum_core::Error::Shape) unless the codeblock has [`CcsdsRs::codeblock_len`] octets.
    pub fn decode_bytes(&self, block: &[u8]) -> Result<(Vec<u8>, Option<usize>)> {
        expect_len(block.len(), self.codeblock_len(), "octets")?;
        let (n, k, i_depth) = (self.rs.n_symbols(), self.rs.k_symbols(), self.depth);
        let mut frame = block[..self.frame_len()].to_vec();
        let mut total = Some(0);
        for i in 0..i_depth {
            let r: Vec<u16> = (0..n).map(|t| u16::from(from_dual_basis(block[t * i_depth + i]))).collect();
            let (msg, fixed) = self.rs.decode_symbols(&r)?;
            match fixed {
                Some(c) => {
                    total = total.map(|t| t + c);
                    for (t, &s) in msg.iter().enumerate().take(k) {
                        frame[t * i_depth + i] = to_dual_basis(s as u8);
                    }
                }
                None => total = None,
            }
        }
        Ok((frame, total))
    }
}

fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8).map(|b| b.iter().fold(0u8, |acc, &v| (acc << 1) | (v & 1))).collect()
}

fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |k| (b >> k) & 1)).collect()
}

impl BlockCode for CcsdsRs {
    fn k(&self) -> usize {
        8 * self.frame_len()
    }

    fn n(&self) -> usize {
        8 * self.codeblock_len()
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k(), "bits")?;
        Ok(bytes_to_bits(&self.encode_bytes(&bits_to_bytes(info))?))
    }

    /// Hard decisions on the LLRs, then [`CcsdsRs::decode_bytes`].
    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        let (frame, _) = self.decode_bytes(&bits_to_bytes(&hard_decisions(llr)))?;
        Ok(bytes_to_bits(&frame))
    }
}

container_traits!(CcsdsRs);

#[cfg(test)]
mod tests {
    use super::ldpc::ar4ja_pi;
    use super::*;
    use comnum_core::rng::Rng;
    use comnum_math::gf2m::Field;

    fn random_bits(n: usize, rng: &mut Rng) -> Vec<u8> {
        (0..n).map(|_| rng.bit()).collect()
    }

    fn bpsk(c: &[u8], amplitude: f64) -> Vec<f64> {
        c.iter().map(|&b| if b == 0 { amplitude } else { -amplitude }).collect()
    }

    #[test]
    fn the_dual_basis_matrices_are_inverse_and_reproduce_annex_f() {
        for u in 0..=255u8 {
            assert_eq!(from_dual_basis(to_dual_basis(u)), u);
        }
        // Annex F, example 1: [z] = 10111001 gives [u] = 00101010 = alpha^213
        assert_eq!(from_dual_basis(0b1011_1001), 0b0010_1010);
        let f = Field::new(8, RS_FIELD).unwrap();
        assert_eq!(f.pow(213), 0b0010_1010);
        // example 2: [u] = 01011001 = alpha^152 gives [z] = 11101000
        assert_eq!(f.pow(152), 0b0101_1001);
        assert_eq!(to_dual_basis(0b0101_1001), 0b1110_1000);
    }

    #[test]
    fn the_generators_are_those_of_annex_g() {
        let f = Field::new(8, RS_FIELD).unwrap();
        // the exponents of G_0 ... G_16 (E = 16) and G_0 ... G_8 (E = 8), symmetric
        let g16 = [0, 249, 59, 66, 4, 43, 126, 251, 97, 30, 3, 213, 50, 66, 170, 5, 24];
        let g8 = [0, 30, 230, 49, 235, 129, 81, 76, 173];
        for (e, half) in [(16, &g16[..]), (8, &g8[..])] {
            let rs = CcsdsRs::new(e, 1, 0).unwrap();
            let g = rs.code().generator();
            assert_eq!(g.len(), 2 * e + 1);
            for (i, &x) in half.iter().enumerate() {
                assert_eq!(u32::from(g[i]), f.pow(x), "E = {e}, G{i}");
                assert_eq!(g[i], g[2 * e - i], "E = {e}, G{i} = G{}", 2 * e - i);
            }
        }
    }

    #[test]
    fn a_codeblock_corrects_e_symbols_per_codeword() {
        let mut rng = Rng::new(1);
        for (e, depth, fill) in [(16, 5, 0), (8, 2, 11), (16, 8, 3)] {
            let rs = CcsdsRs::new(e, depth, fill).unwrap();
            assert_eq!(rs.codeblock_len() - rs.frame_len(), 2 * e * depth);
            let frame: Vec<u8> = (0..rs.frame_len()).map(|_| rng.below(256) as u8).collect();
            let mut block = rs.encode_bytes(&frame).unwrap();
            assert_eq!(&block[..frame.len()], &frame[..], "systematic");
            // E errors in every codeword: positions l = t I + i
            for i in 0..depth {
                for t in 0..e {
                    block[(3 * t + 1) * depth + i] ^= 0x5A;
                }
            }
            let (out, fixed) = rs.decode_bytes(&block).unwrap();
            assert_eq!(out, frame, "E = {e}, I = {depth}, q = {fill}");
            assert_eq!(fixed, Some(e * depth));
        }
        assert!(CcsdsRs::new(16, 6, 0).is_err() && CcsdsRs::new(12, 1, 0).is_err());
    }

    #[test]
    fn the_rate_7_8_code_is_that_of_table_c1() {
        let code = CcsdsLdpc::c2().unwrap();
        assert_eq!((code.k(), code.n()), (7136, 8160));
        let mut rng = Rng::new(2);
        let info = random_bits(7136, &mut rng);
        // G H^T = 0: the codeword of Table C-1 satisfies the checks of Table 7-1
        assert!(code.ldpc().is_codeword(&code.full_codeword(&info).unwrap()));
        let c = code.encode_block(&info).unwrap();
        assert_eq!((c.len(), c[..7136] == info[..], &c[8158..]), (8160, true, &[0u8, 0][..]));
        // decoding a noisy codeword: BPSK at Es/N0 = 5 dB (sigma = 0.4), exact LLRs
        let llr: Vec<f64> = bpsk(&c, 1.0).iter().map(|v| 2.0 / 0.16 * (v + 0.4 * rng.gaussian())).collect();
        assert_eq!(code.decode_block(&llr).unwrap(), info);
    }

    #[test]
    fn the_ar4ja_codes_have_the_lengths_of_table_7_5_and_encode_codewords() {
        let mut rng = Rng::new(3);
        for (k, rate, n) in
            [(1024, Ar4jaRate::R1_2, 2048), (1024, Ar4jaRate::R2_3, 1536), (1024, Ar4jaRate::R4_5, 1280)]
        {
            let code = CcsdsLdpc::ar4ja(k, rate).unwrap();
            assert_eq!((code.k(), code.n()), (k, n), "rate {}", rate.name());
            let info = random_bits(k, &mut rng);
            assert!(code.ldpc().is_codeword(&code.full_codeword(&info).unwrap()), "rate {}", rate.name());
            let c = code.encode_block(&info).unwrap();
            assert_eq!(&c[..k], &info[..]);
            let llr: Vec<f64> = bpsk(&c, 1.0).iter().map(|v| 2.0 / 0.16 * (v + 0.4 * rng.gaussian())).collect();
            assert_eq!(code.decode_block(&llr).unwrap(), info, "rate {}", rate.name());
        }
        assert!(CcsdsLdpc::ar4ja(2048, Ar4jaRate::R1_2).is_err());
    }

    #[test]
    fn the_permutations_of_the_ar4ja_codes_are_permutations() {
        for m in [128, 256, 512, 1024, 2048, 4096, 8192] {
            for k in 1..=26 {
                let mut seen = vec![false; m];
                for i in 0..m {
                    seen[ar4ja_pi(k, i, m)] = true;
                }
                assert!(seen.iter().all(|&s| s), "Pi_{k}, M = {m}");
            }
        }
    }

    #[test]
    fn the_turbo_codes_have_the_lengths_of_table_6_2() {
        let table = [
            (1784, [3576, 5364, 7152, 10728]),
            (3568, [7144, 10716, 14288, 21432]),
            (7136, [14280, 21420, 28560, 42840]),
            (8920, [17848, 26772, 35696, 53544]),
        ];
        for (k, ns) in table {
            for (r, n) in [TurboRate::R1_2, TurboRate::R1_3, TurboRate::R1_4, TurboRate::R1_6].into_iter().zip(ns) {
                assert_eq!(CcsdsTurbo::new(k, r).unwrap().n(), n, "k = {k}, rate {}", r.name());
            }
        }
        assert!(CcsdsTurbo::new(1000, TurboRate::R1_2).is_err());
    }

    #[test]
    fn the_turbo_permutation_starts_as_computed_from_sec_6_3_g() {
        // pi(1..10) for every k (computed from the algorithm, a regression vector)
        for k in TURBO_LENGTHS {
            let p = turbo_permutation(k).unwrap();
            let first: Vec<usize> = p.perm()[..10].iter().map(|v| v + 1).collect();
            assert_eq!(first, [4, 171, 300, 467, 596, 763, 892, 1059, 1188, 1355], "k = {k}");
        }
    }

    #[test]
    fn the_turbo_codes_terminate_and_decode() {
        let mut rng = Rng::new(4);
        for rate in [TurboRate::R1_2, TurboRate::R1_3, TurboRate::R1_4, TurboRate::R1_6] {
            let code = CcsdsTurbo::new(1784, rate).unwrap();
            let info = random_bits(1784, &mut rng);
            let c = code.encode_block(&info).unwrap();
            assert_eq!(c.len(), code.n());
            // the systematic output 0a leads every bit time: the information, then the 4
            // tail bits of encoder a
            let per = c.len() / (1784 + 4);
            if rate != TurboRate::R1_2 {
                let sys: Vec<u8> = (0..1784).map(|t| c[t * per]).collect();
                assert_eq!(sys, info);
            }
            let llr: Vec<f64> = bpsk(&c, 1.0).iter().map(|v| 2.0 * (v + 0.5 * rng.gaussian())).collect();
            assert_eq!(code.decode_block(&llr).unwrap(), info, "rate {}", rate.name());
        }
    }
}
