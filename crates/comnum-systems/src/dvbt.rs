//! DVB-T (ETSI EN 300 744 V1.6.2): the non-hierarchical transmitter and a receiver that
//! knows the channel, the constellations of every hierarchy, the inner interleaver and
//! the transmission parameter signalling (TPS).
//!
//! # Model
//!
//! The transmitter of Sec. 4 (Figure 1), for one transport stream:
//!
//! 1. the energy dispersal, RS(204, 188) and the outer interleaver of [`crate::dvb`],
//!    identical to DVB-S (Secs. 4.3.1 and 4.3.2);
//! 2. the inner code (171, 133) punctured by Table 2, $X$ before $Y$ ([`InnerRate`],
//!    Sec. 4.3.3), continuous across the symbols;
//! 3. the inner interleaver ([`InnerInterleaver`], Sec. 4.3.4): the punctured stream is
//!    demultiplexed into $v = 2, 4, 6$ sub-streams (QPSK, 16-QAM, 64-QAM; $x_{d} \to
//!    b_{e, d \,\mathrm{div}\, v}$, $e = [d \bmod v] \,\mathrm{div}\, (v/2) + 2 [d \bmod
//!    (v/2)]$), sub-stream $e$ is interleaved in blocks of 126 bits by $H_{e}(w) = (w +
//!    s_{e}) \bmod 126$, $s = (0, 63, 105, 42, 21, 84)$, the words $y'_{w} = (a_{0,w},
//!    \dots, a_{v-1,w})$ of 12 (2K) or 48 (8K) blocks fill a symbol, and the symbol
//!    interleaver ([`symbol_permutation`]) places word $q$ on data cell $H(q)$ in the even
//!    symbols and takes data cell $q$ from word $H(q)$ in the odd ones;
//! 4. the mapping on the constellations of Figure 9 ([`constellation`], Sec. 4.3.5): $y_{0},
//!    y_{2}, y_{4}$ on the real axis, $y_{1}, y_{3}, y_{5}$ on the imaginary axis, the
//!    first of each the sign, normalised by Table 6;
//! 5. the OFDM frame of [`comnum_waveform::ofdm::dvbt`] (pilots, TPS, data cells) and the
//!    OFDM modulation with its guard interval (Secs. 4.4 to 4.6); the TPS of every frame
//!    ([`Tps`], Sec. 4.6) signal the mode, the guard, the constellation and the rate, and
//!    are protected by the shortened BCH(67, 53, $t = 2$) code.
//!
//! As [`crate::dvbs::Dvbs`], the chain transmits a burst of whole transport packets: the
//! outer interleaver starts empty and is flushed by 11 packets of zeros, the inner code
//! starts in state 0 and returns to it within the zero padding that fills the last OFDM
//! symbol; the burst starts with the first symbol of a super-frame. The receiver
//! ([`Dvbt::inner_decode`]) takes the cells of the received symbols, the noise variance of
//! a carrier and, if any, the gain of every carrier (a static channel known to the
//! receiver, the perfect channel estimation of Annex A), computes the exact LLRs of the
//! data cells, deinterleaves them, decodes the inner code by Viterbi (soft decisions) and
//! the outer coding.
//!
//! Annex A gives the C/N of a BER of $2 \cdot 10^{-4}$ after the Viterbi decoder, "calculated
//! over the transmitted 6 817 OFDM carriers": the mean power of the carriers (pilots
//! boosted) over the noise of a carrier ([`Dvbt::carrier_noise_var`]); a data cell then
//! has $E_{s}/N_{0} = C/N - 10 \log_{10} 1.08$.
//!
//! Hierarchical transmission (two streams of their own rates on the non-uniform
//! constellations of $\alpha = 2, 4$) is not implemented by [`Dvbt`]; its constellations
//! are ([`constellation`]).
//!
//! # Examples
//! ```
//! use comnum_core::rng::Rng;
//! use comnum_systems::dvb::{self, InnerRate};
//! use comnum_systems::dvbt::{Dvbt, Modulation};
//! use comnum_waveform::ofdm::dvbt::{Guard, Mode};
//! let link = Dvbt::new(Mode::K2, Guard::G1_4, Modulation::Qam16, InnerRate::R2_3)?;
//! assert_eq!(link.packets_per_superframe(), 672);              // Table 16
//! let packets = dvb::random_packets(30, &mut Rng::new(1));
//! let x = link.modulate(&packets)?;                            // samples at 1/T
//! let cells = link.demodulate(&x)?;
//! let d = link.decode_cells(&cells, 30, 1e-3, None)?;
//! assert_eq!(d.packets, packets);
//! // Table 17: 16-QAM 2/3, guard 1/4, 8 MHz: 13.27 Mbit/s
//! assert!((link.useful_bit_rate(8e6) / 1e6 - 13.27).abs() < 0.005);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. ETSI EN 300 744 V1.6.2, *Digital Video Broadcasting (DVB); Framing structure,
//!    channel coding and modulation for digital terrestrial television*, 2015, Secs. 4.3
//!    to 4.7, Tables 2, 3a, 3b, 6, 9 to 17, Annexes A and C.

use crate::dvb::{self, CODED_PACKET_LEN, InnerRate, OuterDecoded, PACKET_LEN};
use comnum_core::{Error, Result, ensure, ensure_shape};
use comnum_fec::bch::Bch;
use comnum_fec::conv::{Trellis, viterbi};
use comnum_fec::puncture::Puncturer;
use comnum_mapping::constellation::Constellation;
use comnum_math::bits::gray_inverse;
use comnum_math::{C64, Jacobian};
use comnum_waveform::ofdm::Ofdm;
use comnum_waveform::ofdm::dvbt::{DvbtFrame, FRAMES_PER_SUPERFRAME, Guard, Mode, SYMBOLS_PER_FRAME, tps_signs};
use std::str::FromStr;

/// The constraint length $K$ of the inner code.
const CONSTRAINT: usize = 7;
/// Bits of a block of the bit interleaver.
const BIT_BLOCK: usize = 126;
/// The shifts of the six bit interleavers, $H_{e}(w) = (w + s_{e}) \bmod 126$ (Sec. 4.3.4.1).
const BIT_SHIFTS: [usize; 6] = [0, 63, 105, 42, 21, 84];
/// The TPS synchronisation word of frames 1 and 3, $s_{1} \dots s_{16}$ (Sec. 4.6.2.2);
/// frames 2 and 4 carry its complement.
const TPS_SYNC: [u8; 16] = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0];

/// The constellation of the data cells (Sec. 4.3.5, TPS Table 11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modulation {
    /// QPSK, $v = 2$.
    Qpsk,
    /// 16-QAM, $v = 4$.
    Qam16,
    /// 64-QAM, $v = 6$.
    Qam64,
}

impl Modulation {
    /// The bits of a cell, $v$.
    pub fn bits(self) -> usize {
        match self {
            Modulation::Qpsk => 2,
            Modulation::Qam16 => 4,
            Modulation::Qam64 => 6,
        }
    }

    /// `QPSK`, `16-QAM` or `64-QAM`.
    pub fn name(self) -> &'static str {
        match self {
            Modulation::Qpsk => "QPSK",
            Modulation::Qam16 => "16-QAM",
            Modulation::Qam64 => "64-QAM",
        }
    }

    /// The code of TPS Table 11.
    fn tps_code(self) -> u8 {
        match self {
            Modulation::Qpsk => 0,
            Modulation::Qam16 => 1,
            Modulation::Qam64 => 2,
        }
    }
}

impl FromStr for Modulation {
    type Err = Error;

    /// `QPSK`, `16-QAM` (or `16QAM`) or `64-QAM` (or `64QAM`), any case.
    fn from_str(s: &str) -> Result<Modulation> {
        match s.trim().to_ascii_uppercase().replace('-', "").as_str() {
            "QPSK" => Ok(Modulation::Qpsk),
            "16QAM" => Ok(Modulation::Qam16),
            "64QAM" => Ok(Modulation::Qam64),
            _ => Err(Error::Config(format!("the DVB-T constellation must be QPSK, 16-QAM or 64-QAM, got {s:?}"))),
        }
    }
}

/// The hierarchy of TPS Table 12: non-hierarchical, or the $\alpha$ of the hierarchical
/// constellation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hierarchy {
    /// Non-hierarchical (uniform constellations).
    None,
    /// Hierarchical, $\alpha = 1$.
    Alpha1,
    /// Hierarchical, $\alpha = 2$.
    Alpha2,
    /// Hierarchical, $\alpha = 4$.
    Alpha4,
}

impl Hierarchy {
    /// The $\alpha$ of the constellation (1 when non-hierarchical).
    pub fn alpha(self) -> u32 {
        match self {
            Hierarchy::None | Hierarchy::Alpha1 => 1,
            Hierarchy::Alpha2 => 2,
            Hierarchy::Alpha4 => 4,
        }
    }

    fn tps_code(self) -> u8 {
        match self {
            Hierarchy::None => 0,
            Hierarchy::Alpha1 => 1,
            Hierarchy::Alpha2 => 2,
            Hierarchy::Alpha4 => 3,
        }
    }
}

/// The constellation of `modulation` with the parameter $\alpha$ = `alpha` (1, 2 or 4;
/// 1 for the non-hierarchical modes), Figure 9 and Table 6: the label
/// $y_{0} y_{1} \dots y_{v-1}$ (most significant first) is the point $z / \sqrt{E|z|^{2}}$
/// whose real part carries $y_{0}, y_{2}, y_{4}$ and imaginary part $y_{1}, y_{3}, y_{5}$:
/// on each axis the first bit is the sign (1 negative) and the others select the
/// magnitude $\alpha + 2 (L - 1 - m)$, $m$ their Gray index, $L$ the magnitudes of an axis
/// (16-QAM, $\alpha = 1$: 00 gives 3, 01 gives 1, 11 gives −1, 10 gives −3).
///
/// # Errors
/// [`Error::Config`] for $\alpha \notin \{1, 2, 4\}$, or $\alpha \ne 1$ with QPSK.
pub fn constellation(modulation: Modulation, alpha: u32) -> Result<Constellation> {
    ensure(matches!(alpha, 1 | 2 | 4), || format!("alpha must be 1, 2 or 4, got {alpha}"))?;
    ensure(modulation != Modulation::Qpsk || alpha == 1, || "QPSK has no hierarchical alpha".into())?;
    let v = modulation.bits();
    let per_axis = v / 2;
    let magnitudes = 1usize << (per_axis - 1);
    let level = |bits: &[usize]| -> f64 {
        let sign = if bits[0] == 1 { -1.0 } else { 1.0 };
        let rest = bits[1..].iter().fold(0usize, |acc, &b| (acc << 1) | b);
        let m = gray_inverse(rest);
        sign * (f64::from(alpha) + 2.0 * (magnitudes - 1 - m) as f64)
    };
    let points: Vec<C64> = (0..1usize << v)
        .map(|label| {
            let y: Vec<usize> = (0..v).map(|j| (label >> (v - 1 - j)) & 1).collect();
            let re: Vec<usize> = y.iter().step_by(2).copied().collect();
            let im: Vec<usize> = y.iter().skip(1).step_by(2).copied().collect();
            C64::new(level(&re), level(&im))
        })
        .collect();
    Ok(Constellation::from_points(points)?.normalized())
}

/// The symbol interleaver $H(q)$ of the mode (Sec. 4.3.4.2, Tables 3a and 3b): the
/// $(N_{r} - 1)$-bit register $R'_{i}$ ($N_{r} = 11$ or 13; zero for $i = 0, 1$, one for
/// $i = 2$, then shifted down with the feedback $R'_{i-1}\[0\] \oplus R'_{i-1}\[3\]$ in the 2K
/// mode or $R'_{i-1}\[0\] \oplus R'_{i-1}\[1\] \oplus R'_{i-1}\[4\] \oplus R'_{i-1}\[6\]$ in the 8K
/// mode), its bits permuted into $R_{i}$, $H = (i \bmod 2)\, 2^{N_{r}-1} + R_{i}$, the
/// values below $N_{\max}$ kept in order.
pub fn symbol_permutation(mode: Mode) -> Vec<usize> {
    // the bit of R_i that bit p of R'_i goes to, Tables 3a and 3b (p = 0 first)
    let (nr, to): (u32, &[u32]) = match mode {
        Mode::K2 => (11, &[4, 3, 9, 6, 2, 8, 1, 5, 7, 0]),
        Mode::K8 => (13, &[7, 1, 4, 2, 9, 6, 8, 10, 0, 3, 11, 5]),
    };
    let nmax = mode.data_cells();
    let mut h = Vec::with_capacity(nmax);
    let mut rp = 0u32;
    for i in 0..1u32 << nr {
        rp = match i {
            0 | 1 => 0,
            2 => 1,
            _ => {
                let bit = |b: u32| (rp >> b) & 1;
                let fb = match mode {
                    Mode::K2 => bit(0) ^ bit(3),
                    Mode::K8 => bit(0) ^ bit(1) ^ bit(4) ^ bit(6),
                };
                (rp >> 1) | (fb << (nr - 2))
            }
        };
        let r: u32 = to.iter().enumerate().map(|(p, &q)| ((rp >> p) & 1) << q).sum();
        let value = ((i % 2) << (nr - 1)) | r;
        if (value as usize) < nmax {
            h.push(value as usize);
        }
    }
    h
}

/// The inner interleaver of a mode and a constellation (see the module): for every bit
/// $y_{j}$ of data cell $q$ of an even or odd symbol, the index of the punctured bit it
/// carries among the $v N_{\max}$ bits of the symbol.
#[derive(Clone, Debug)]
pub struct InnerInterleaver {
    v: usize,
    /// `even[q v + j]`: the punctured bit carried by $y_{j}$ of data cell $q$.
    even: Vec<usize>,
    odd: Vec<usize>,
}

impl InnerInterleaver {
    /// The interleaver of the mode `mode` and the constellation `modulation`
    /// (non-hierarchical).
    pub fn new(mode: Mode, modulation: Modulation) -> InnerInterleaver {
        let v = modulation.bits();
        let h = symbol_permutation(mode);
        let mut h_inv = vec![0; h.len()];
        for (q, &hq) in h.iter().enumerate() {
            h_inv[hq] = q;
        }
        // the input bit d = d_o v + r of sub-stream e(r) = (r mod v) div (v/2) + 2 (r mod (v/2))
        let mut r_of = vec![0; v];
        for r in 0..v {
            r_of[(r % v) / (v / 2) + 2 * (r % (v / 2))] = r;
        }
        let source = |word: usize, j: usize| -> usize {
            let (block, w) = (word / BIT_BLOCK, word % BIT_BLOCK);
            let d_o = block * BIT_BLOCK + (w + BIT_SHIFTS[j]) % BIT_BLOCK;
            d_o * v + r_of[j]
        };
        let build = |word_of: &dyn Fn(usize) -> usize| -> Vec<usize> {
            (0..h.len()).flat_map(|q| (0..v).map(move |j| (q, j))).map(|(q, j)| source(word_of(q), j)).collect()
        };
        // even: y_{H(q)} = y'_q, so cell q holds word H^{-1}(q); odd: y_q = y'_{H(q)}
        let even = build(&|q| h_inv[q]);
        let odd = build(&|q| h[q]);
        InnerInterleaver { v, even, odd }
    }

    /// The bits of a cell, $v$.
    pub fn bits_per_cell(&self) -> usize {
        self.v
    }

    /// The punctured bit carried by bit $y_{j}$ of data cell `q` of symbol `l`.
    pub fn source(&self, l: usize, q: usize, j: usize) -> usize {
        let perm = if l % 2 == 0 { &self.even } else { &self.odd };
        perm[q * self.v + j]
    }

    fn perm(&self, l: usize) -> &[usize] {
        if l % 2 == 0 { &self.even } else { &self.odd }
    }

    /// The labels of the data cells of symbol `l` from its $v N_{\max}$ punctured bits.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of bits.
    pub fn labels(&self, l: usize, bits: &[u8]) -> Result<Vec<usize>> {
        let perm = self.perm(l);
        ensure_shape(bits.len() == perm.len(), || format!("{} bits for a symbol of {}", bits.len(), perm.len()))?;
        Ok(perm
            .chunks(self.v)
            .map(|c| c.iter().fold(0usize, |acc, &d| (acc << 1) | usize::from(bits[d] & 1)))
            .collect())
    }

    /// The LLRs of the punctured bits of symbol `l` from the LLRs of its cells ($v$ per
    /// cell, $y_{0}$ first).
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of LLRs.
    pub fn deinterleave(&self, l: usize, llr: &[f64]) -> Result<Vec<f64>> {
        let perm = self.perm(l);
        ensure_shape(llr.len() == perm.len(), || format!("{} LLRs for a symbol of {}", llr.len(), perm.len()))?;
        let mut out = vec![0.0; perm.len()];
        for (&d, &v) in perm.iter().zip(llr) {
            out[d] = v;
        }
        Ok(out)
    }
}

/// The transmission parameter signalling of one frame (Sec. 4.6, Tables 9 to 15a).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tps {
    /// The frame of the super-frame, 0 to 3 (Table 10: frames 1 to 4).
    pub frame: usize,
    /// The constellation (Table 11).
    pub modulation: Modulation,
    /// The hierarchy (Table 12).
    pub hierarchy: Hierarchy,
    /// The code rate of the high-priority stream, or of the stream (Table 13).
    pub rate_hp: InnerRate,
    /// The code rate of the low-priority stream (`None` when non-hierarchical: `000`).
    pub rate_lp: Option<InnerRate>,
    /// The guard interval (Table 14).
    pub guard: Guard,
    /// The transmission mode (Table 15).
    pub mode: Mode,
    /// The cell identifier (Table 15a), `None` if not transmitted.
    pub cell_id: Option<u16>,
}

/// The shortened BCH(67, 53, $t = 2$) of the TPS, from BCH(127, 113) of the primitive
/// polynomial $x^{7} + x^{3} + 1$: its generator is the $h(x) = x^{14} + x^{9} + x^{8} +
/// x^{6} + x^{5} + x^{4} + x^{2} + x + 1$ of Sec. 4.6.2.11.
///
/// # Errors
/// Never: the parameters are those of the standard.
pub fn tps_code() -> Result<Bch> {
    Bch::new(7, 0x89, 2, 67)
}

fn rate_code(r: InnerRate) -> u8 {
    match r {
        InnerRate::R1_2 => 0,
        InnerRate::R2_3 => 1,
        InnerRate::R3_4 => 2,
        InnerRate::R5_6 => 3,
        InnerRate::R7_8 => 4,
    }
}

fn push_bits(out: &mut Vec<u8>, value: u32, n: u32) {
    out.extend((0..n).rev().map(|k| ((value >> k) & 1) as u8));
}

fn field(bits: &[u8], from: usize, n: usize) -> u32 {
    bits[from..from + n].iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
}

impl Tps {
    /// The 68 bits $s_{0} \dots s_{67}$ of the frame (Table 9): $s_{0} = 0$ (it only
    /// initialises the differential BPSK), the synchronisation word (complemented in
    /// frames 2 and 4), the length indicator (23 or 31 bits in use), the parameters, the
    /// cell identifier (its high byte in frames 1 and 3, its low byte in frames 2 and 4),
    /// six zeros and the 14 BCH parity bits of $s_{1} \dots s_{53}$.
    ///
    /// # Errors
    /// [`Error::Config`] for a frame outside 0 to 3.
    pub fn bits(&self) -> Result<Vec<u8>> {
        ensure(self.frame < FRAMES_PER_SUPERFRAME, || format!("the TPS frame must be 0 to 3, got {}", self.frame))?;
        let mut s = vec![0u8];
        let complement = u8::from(self.frame % 2 == 1);
        s.extend(TPS_SYNC.iter().map(|&b| b ^ complement));
        push_bits(&mut s, if self.cell_id.is_some() { 0b01_1111 } else { 0b01_0111 }, 6);
        push_bits(&mut s, self.frame as u32, 2);
        push_bits(&mut s, u32::from(self.modulation.tps_code()), 2);
        push_bits(&mut s, u32::from(self.hierarchy.tps_code()), 3);
        push_bits(&mut s, u32::from(rate_code(self.rate_hp)), 3);
        push_bits(&mut s, self.rate_lp.map_or(0, |r| u32::from(rate_code(r))), 3);
        push_bits(
            &mut s,
            match self.guard {
                Guard::G1_32 => 0,
                Guard::G1_16 => 1,
                Guard::G1_8 => 2,
                Guard::G1_4 => 3,
            },
            2,
        );
        push_bits(&mut s, u32::from(self.mode == Mode::K8), 2);
        let id = u32::from(self.cell_id.unwrap_or(0));
        push_bits(&mut s, if self.frame % 2 == 0 { id >> 8 } else { id & 0xFF }, 8);
        s.extend([0u8; 6]);
        let coded = tps_code()?.encode(&s[1..54])?;
        s.extend_from_slice(&coded[53..]);
        Ok(s)
    }

    /// The signalling of the 68 received TPS bits of a frame: the BCH decoding (up to two
    /// errors), the synchronisation word, then the fields. With a cell identifier, only
    /// the byte of this frame is known: the other is returned as zero.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of bits; [`Error::Numerical`] if the BCH
    /// decoding fails or the synchronisation word is wrong; [`Error::Config`] for a
    /// reserved value.
    pub fn decode(bits: &[u8]) -> Result<Tps> {
        ensure_shape(bits.len() == SYMBOLS_PER_FRAME, || format!("{} TPS bits for a frame of 68", bits.len()))?;
        let (msg, fixed) = tps_code()?.decode(&bits[1..])?;
        if fixed.is_none() {
            return Err(Error::Numerical("the TPS BCH decoding failed".into()));
        }
        let mut s = vec![0u8];
        s.extend(msg);
        let sync = &s[1..17];
        let frame_bit = if sync == TPS_SYNC {
            0
        } else if sync.iter().zip(TPS_SYNC).all(|(&a, b)| a != b) {
            1
        } else {
            return Err(Error::Numerical("the TPS synchronisation word is wrong".into()));
        };
        let frame = field(&s, 23, 2) as usize;
        ensure(frame % 2 == frame_bit, || "the TPS frame number contradicts its synchronisation word".into())?;
        let modulation = match field(&s, 25, 2) {
            0 => Modulation::Qpsk,
            1 => Modulation::Qam16,
            2 => Modulation::Qam64,
            c => return Err(Error::Config(format!("reserved TPS constellation {c:02b}"))),
        };
        let hierarchy = match field(&s, 27, 3) {
            0 => Hierarchy::None,
            1 => Hierarchy::Alpha1,
            2 => Hierarchy::Alpha2,
            3 => Hierarchy::Alpha4,
            c => return Err(Error::Config(format!("TPS hierarchy {c:03b} (Annex F) is not supported"))),
        };
        let rate = |code: u32| -> Result<InnerRate> {
            usize::try_from(code)
                .ok()
                .and_then(|c| InnerRate::ALL.get(c).copied())
                .ok_or_else(|| Error::Config(format!("reserved TPS code rate {code:03b}")))
        };
        let rate_hp = rate(field(&s, 30, 3))?;
        let rate_lp = if hierarchy == Hierarchy::None { None } else { Some(rate(field(&s, 33, 3))?) };
        let guard = [Guard::G1_32, Guard::G1_16, Guard::G1_8, Guard::G1_4][field(&s, 36, 2) as usize];
        let mode = match field(&s, 38, 2) {
            0 => Mode::K2,
            1 => Mode::K8,
            c => return Err(Error::Config(format!("TPS mode {c:02b} (Annex F) is not supported"))),
        };
        let cell_id = match field(&s, 17, 6) {
            0b01_1111 => {
                let byte = field(&s, 40, 8) as u16;
                Some(if frame % 2 == 0 { byte << 8 } else { byte })
            }
            _ => None,
        };
        Ok(Tps { frame, modulation, hierarchy, rate_hp, rate_lp, guard, mode, cell_id })
    }
}

/// A DVB-T transmitter and receiver of one non-hierarchical configuration (see the
/// module).
#[derive(Clone, Debug)]
pub struct Dvbt {
    mode: Mode,
    guard: Guard,
    modulation: Modulation,
    rate: InnerRate,
    frame: DvbtFrame,
    interleaver: InnerInterleaver,
    trellis: Trellis,
    puncturer: Puncturer,
    constellation: Constellation,
    ofdm: Ofdm,
    /// The cell identifier signalled by the TPS.
    pub cell_id: Option<u16>,
}

/// The output of [`Dvbt::decode_cells`].
pub type DvbtDecoded = OuterDecoded;

impl Dvbt {
    /// The link of a mode, a guard interval, a constellation and a code rate.
    ///
    /// # Errors
    /// Never: every combination is defined.
    pub fn new(mode: Mode, guard: Guard, modulation: Modulation, rate: InnerRate) -> Result<Dvbt> {
        let frame = DvbtFrame::new(mode);
        let ofdm = frame.ofdm(guard)?;
        Ok(Dvbt {
            mode,
            guard,
            modulation,
            rate,
            interleaver: InnerInterleaver::new(mode, modulation),
            frame,
            trellis: dvb::inner_trellis()?,
            puncturer: rate.puncturer()?,
            constellation: constellation(modulation, 1)?,
            ofdm,
            cell_id: None,
        })
    }

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The guard interval.
    pub fn guard(&self) -> Guard {
        self.guard
    }

    /// The constellation of the data cells.
    pub fn modulation(&self) -> Modulation {
        self.modulation
    }

    /// The code rate.
    pub fn rate(&self) -> InnerRate {
        self.rate
    }

    /// The frame structure.
    pub fn frame(&self) -> &DvbtFrame {
        &self.frame
    }

    /// The points of the data cells (label $y_{0} \dots y_{v-1}$).
    pub fn constellation(&self) -> &Constellation {
        &self.constellation
    }

    /// The OFDM modulator (one sample per $T$, the guard interval as cyclic prefix).
    pub fn ofdm(&self) -> &Ofdm {
        &self.ofdm
    }

    /// The TPS of frame `frame` (0 to 3) of a super-frame.
    pub fn tps(&self, frame: usize) -> Tps {
        Tps {
            frame,
            modulation: self.modulation,
            hierarchy: Hierarchy::None,
            rate_hp: self.rate,
            rate_lp: None,
            guard: self.guard,
            mode: self.mode,
            cell_id: self.cell_id,
        }
    }

    /// Punctured bits of a symbol, $v N_{\max}$.
    pub fn coded_bits_per_symbol(&self) -> usize {
        self.modulation.bits() * self.mode.data_cells()
    }

    /// Input bits of the inner code per symbol, $v N_{\max} r$.
    pub fn input_bits_per_symbol(&self) -> usize {
        let (p, w) = self.rate.fraction();
        self.coded_bits_per_symbol() * p / w
    }

    /// The RS packets of a super-frame, $4 \cdot 68 \cdot v N_{\max} r / (8 \cdot 204)$
    /// (Table 16).
    pub fn packets_per_superframe(&self) -> usize {
        FRAMES_PER_SUPERFRAME * SYMBOLS_PER_FRAME * self.input_bits_per_symbol() / (8 * CODED_PACKET_LEN)
    }

    /// The useful bit rate (bit/s) in a channel of bandwidth `bandwidth_hz`,
    /// $v N_{\max} r \cdot (188/204) / T_{S}$, $T_{S} = (T_{U}/T + \Delta/T)\, T$ (Table 17
    /// and the bit-rate columns of Tables A.1 to A.3).
    pub fn useful_bit_rate(&self, bandwidth_hz: f64) -> f64 {
        let t = comnum_waveform::ofdm::dvbt::elementary_period(bandwidth_hz);
        let ts = (self.mode.fft_len() + self.guard.samples(self.mode)) as f64 * t;
        self.input_bits_per_symbol() as f64 * PACKET_LEN as f64 / CODED_PACKET_LEN as f64 / ts
    }

    /// The noise variance of a carrier at a C/N (dB) calculated over the $K$ carriers
    /// (Annex A): the mean power of the carriers, pilots boosted, over $10^{C/N / 10}$.
    pub fn carrier_noise_var(&self, cn_db: f64) -> f64 {
        self.frame.mean_carrier_power() / 10f64.powf(cn_db / 10.0)
    }

    /// The noise variance of a sample of [`Dvbt::modulate`] that gives `carrier_noise_var`
    /// on every carrier: $\sigma^{2} N / K$ ($N$ the FFT size; [`Ofdm`] keeps the power of
    /// a carrier).
    pub fn sample_noise_var(&self, carrier_noise_var: f64) -> f64 {
        carrier_noise_var * self.mode.fft_len() as f64 / self.mode.carriers() as f64
    }

    /// The baseband frequency (Hz) of every carrier in a channel of bandwidth
    /// `bandwidth_hz`, $k' / T_{U}$, $k' = k - (K - 1)/2$ (Sec. 4.4).
    pub fn carrier_frequencies(&self, bandwidth_hz: f64) -> Vec<f64> {
        let tu = self.mode.fft_len() as f64 * comnum_waveform::ofdm::dvbt::elementary_period(bandwidth_hz);
        let centre = (self.mode.carriers() - 1) as f64 / 2.0;
        (0..self.mode.carriers()).map(|k| (k as f64 - centre) / tu).collect()
    }

    /// OFDM symbols of a burst of `n_packets` packets.
    pub fn symbols_len(&self, n_packets: usize) -> usize {
        (8 * dvb::burst_bytes(n_packets) + CONSTRAINT - 1).div_ceil(self.input_bits_per_symbol())
    }

    /// The cells (all $K$ carriers of every symbol) of the bytes of
    /// [`dvb::burst_encode`]: the inner code from state 0, padded with zeros to whole
    /// symbols, the inner interleaver, the mapping, the pilots and the TPS of the
    /// super-frames.
    ///
    /// # Errors
    /// As [`Tps::bits`] (never).
    pub fn cells(&self, bytes: &[u8]) -> Result<Vec<Vec<C64>>> {
        let per_symbol = self.input_bits_per_symbol();
        let n_symbols = (8 * bytes.len() + CONSTRAINT - 1).div_ceil(per_symbol);
        let mut bits = dvb::bytes_to_bits(bytes);
        bits.resize(n_symbols * per_symbol, 0);
        let coded = self.puncturer.puncture(&self.trellis.encode(&bits, false))?;
        let signs: Vec<Vec<f64>> =
            (0..FRAMES_PER_SUPERFRAME).map(|f| tps_signs(&self.tps(f).bits()?)).collect::<Result<_>>()?;
        let points = self.constellation.points();
        coded
            .chunks(self.coded_bits_per_symbol())
            .enumerate()
            .map(|(n, c)| {
                let (l, f) = (n % SYMBOLS_PER_FRAME, (n / SYMBOLS_PER_FRAME) % FRAMES_PER_SUPERFRAME);
                let data: Vec<C64> = self.interleaver.labels(l, c)?.into_iter().map(|i| points[i]).collect();
                self.frame.symbol(l, &data, signs[f][l])
            })
            .collect()
    }

    /// The samples of a burst of whole transport packets: [`Dvbt::cells`] of
    /// [`dvb::burst_encode`], OFDM modulated with the guard interval, one sample per $T$.
    ///
    /// # Errors
    /// As [`dvb::burst_encode`].
    pub fn modulate(&self, packets: &[u8]) -> Result<Vec<C64>> {
        let cells = self.cells(&dvb::burst_encode(packets)?)?;
        self.ofdm.modulate(&cells.concat())
    }

    /// The $K$ cells of every received symbol (the guard intervals removed).
    ///
    /// # Errors
    /// [`Error::Shape`] unless `samples` holds whole symbols.
    pub fn demodulate(&self, samples: &[C64]) -> Result<Vec<Vec<C64>>> {
        let cells = self.ofdm.demodulate(samples)?;
        Ok(cells.chunks(self.mode.carriers()).map(<[C64]>::to_vec).collect())
    }

    /// The bytes decided by the inner decoder from the cells of the symbols of a burst of
    /// `n_packets` packets: the exact LLRs of the data cells for the noise variance
    /// `noise_var` of a carrier and the gains `channel` of the carriers (1 if `None`), the
    /// inner deinterleaver and the soft Viterbi decoder.
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of symbols is not that of the burst or a symbol (or
    /// the channel) does not have $K$ values.
    pub fn inner_decode(
        &self,
        cells: &[Vec<C64>],
        n_packets: usize,
        noise_var: f64,
        channel: Option<&[C64]>,
    ) -> Result<Vec<u8>> {
        let n_symbols = self.symbols_len(n_packets);
        ensure_shape(cells.len() == n_symbols, || {
            format!("{} symbols for a burst of {n_packets} packets ({n_symbols} symbols)", cells.len())
        })?;
        let k_total = self.mode.carriers();
        ensure_shape(channel.is_none_or(|h| h.len() == k_total), || format!("the channel has {k_total} gains"))?;
        let mut llr = Vec::with_capacity(n_symbols * self.coded_bits_per_symbol());
        for (n, symbol) in cells.iter().enumerate() {
            let l = n % SYMBOLS_PER_FRAME;
            let mut z = self.frame.data_of(l, symbol)?;
            let mut var = vec![noise_var; z.len()];
            if let Some(h) = channel {
                for ((zi, vi), &k) in z.iter_mut().zip(&mut var).zip(self.frame.data_carriers(l)) {
                    *zi /= h[k];
                    *vi /= h[k].norm_sqr();
                }
            }
            let cell_llr = self.constellation.llr_with_variances(&z, &var, None, Jacobian::Exact)?;
            llr.extend(self.interleaver.deinterleave(l, &cell_llr)?);
        }
        let steps = n_symbols * self.input_bits_per_symbol();
        let mut bits = viterbi(&self.trellis, &self.puncturer.depuncture(&llr, 2 * steps)?, true)?;
        bits.truncate(8 * dvb::burst_bytes(n_packets));
        Ok(dvb::bits_to_bytes(&bits))
    }

    /// The packets of the cells of a burst of `n_packets` packets ([`dvb::burst_decode`]
    /// of [`Dvbt::inner_decode`]).
    ///
    /// # Errors
    /// As [`Dvbt::inner_decode`].
    pub fn decode_cells(
        &self,
        cells: &[Vec<C64>],
        n_packets: usize,
        noise_var: f64,
        channel: Option<&[C64]>,
    ) -> Result<DvbtDecoded> {
        dvb::burst_decode(&self.inner_decode(cells, n_packets, noise_var, channel)?)
    }

    /// The TPS of the frame whose 68 symbols are `cells` (their first symbol the first of
    /// a frame): the differential BPSK of the TPS carriers decided symbol to symbol, then
    /// [`Tps::decode`].
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of symbols; as [`Tps::decode`].
    pub fn decode_tps(&self, cells: &[Vec<C64>], channel: Option<&[C64]>) -> Result<Tps> {
        ensure_shape(cells.len() == SYMBOLS_PER_FRAME, || format!("{} symbols for a frame of 68", cells.len()))?;
        let stats: Vec<f64> = cells.iter().map(|c| self.frame.tps_statistic(c, channel)).collect::<Result<_>>()?;
        let mut bits = vec![0u8];
        bits.extend(stats.windows(2).map(|w| u8::from(w[0] * w[1] < 0.0)));
        Tps::decode(&bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    #[test]
    fn the_symbol_interleaver_starts_as_computed_from_tables_3a_and_3b() {
        let h2 = symbol_permutation(Mode::K2);
        assert_eq!(&h2[..16], &[0, 1024, 16, 1025, 128, 1056, 2, 1280, 4, 1088, 513, 1160, 48, 1027, 384, 1060]);
        let h8 = symbol_permutation(Mode::K8);
        assert_eq!(&h8[..16], &[0, 4096, 128, 4128, 2048, 4104, 1, 5120, 256, 4192, 2560, 4140, 2065, 5130, 417, 776]);
        for (h, n) in [(h2, 1512), (h8, 6048)] {
            let mut s = h.clone();
            s.sort_unstable();
            assert_eq!(s, (0..n).collect::<Vec<_>>(), "a permutation");
        }
    }

    #[test]
    fn the_inner_interleaver_reproduces_table_c1() {
        // Annex C, Table C.1: 2K, 64-QAM, the first (even) symbol of a super-frame; for a
        // carrier k, the input bit numbers of y0 ... y5
        let rows: [(usize, [usize; 6]); 24] = [
            (1, [0, 381, 631, 256, 128, 509]),
            (2, [4602, 4983, 5233, 4858, 4730, 5111]),
            (3, [36, 417, 667, 292, 164, 545]),
            (4, [4656, 5037, 5287, 4912, 4784, 5165]),
            (5, [48, 429, 679, 304, 176, 557]),
            (6, [2376, 2757, 3007, 2632, 2504, 2885]),
            (7, [780, 1161, 1411, 1036, 908, 1289]),
            (8, [6906, 7287, 7537, 7162, 7034, 7415]),
            (9, [4590, 4971, 5221, 4846, 4718, 5099]),
            (10, [5286, 4911, 5161, 4786, 4658, 5039]),
            (11, [2364, 2745, 2995, 2620, 2492, 2873]),
            (13, [4788, 5169, 4663, 5044, 4916, 4541]),
            (1691, [4194, 3819, 4069, 4450, 4322, 3947]),
            (1693, [7782, 8163, 7657, 8038, 7910, 8291]),
            (1694, [6624, 6249, 6499, 6124, 6752, 6377]),
            (1695, [3402, 3027, 3277, 3658, 3530, 3155]),
            (1696, [546, 171, 421, 46, 674, 299]),
            (1697, [8574, 8955, 8449, 8830, 8702, 8327]),
            (1698, [8376, 8757, 9007, 8632, 8504, 8885]),
            (1699, [1680, 2061, 1555, 1936, 1808, 2189]),
            (1700, [7620, 8001, 8251, 7876, 7748, 8129]),
            (1701, [5700, 5325, 5575, 5956, 5828, 5453]),
            (1702, [8826, 8451, 8701, 8326, 8954, 8579]),
            (1703, [8724, 8349, 8599, 8980, 8852, 8477]),
        ];
        let frame = DvbtFrame::new(Mode::K2);
        let ii = InnerInterleaver::new(Mode::K2, Modulation::Qam64);
        for (k, bits) in rows {
            let q = frame.data_carriers(0).iter().position(|&c| c == k).unwrap();
            let got: Vec<usize> = (0..6).map(|j| ii.source(0, q, j)).collect();
            assert_eq!(got, bits, "carrier {k}");
        }
        // carriers 0, 12, 1692 and 1704 are pilots in this symbol
        for k in [0, 12, 1692, 1704] {
            assert!(!frame.data_carriers(0).contains(&k));
        }
    }

    #[test]
    fn the_constellations_follow_figure_9_and_table_6() {
        // 16-QAM, alpha = 1: labels of the row Im = 3, columns Re = -3, -1, 1, 3
        let c = constellation(Modulation::Qam16, 1).unwrap();
        let s = 10f64.sqrt();
        for (label, re) in [(0b1000, -3.0), (0b1010, -1.0), (0b0010, 1.0), (0b0000, 3.0)] {
            assert!((c.points()[label] * s - C64::new(re, 3.0)).norm() < 1e-12, "label {label:04b}");
        }
        // 64-QAM row Im = -7: 110000 at Re = -7, 010000 at Re = 7
        let c = constellation(Modulation::Qam64, 1).unwrap();
        let s = 42f64.sqrt();
        assert!((c.points()[0b11_0000] * s - C64::new(-7.0, -7.0)).norm() < 1e-12);
        assert!((c.points()[0b01_0000] * s - C64::new(7.0, -7.0)).norm() < 1e-12);
        assert!((c.points()[0b00_1101] * s - C64::new(1.0, 3.0)).norm() < 1e-12);
        // Table 6: the normalisation factors are the mean energies of the sets
        for (m, alpha, energy) in [
            (Modulation::Qpsk, 1, 2.0_f64),
            (Modulation::Qam16, 2, 20.0),
            (Modulation::Qam16, 4, 52.0),
            (Modulation::Qam64, 2, 60.0),
            (Modulation::Qam64, 4, 108.0),
        ] {
            let c = constellation(m, alpha).unwrap();
            let max_re = c.points().iter().map(|p| p.re).fold(0.0, f64::max);
            let expected = (f64::from(alpha) + 2.0 * ((1usize << (m.bits() / 2 - 1)) - 1) as f64) / energy.sqrt();
            assert!((max_re - expected).abs() < 1e-12, "{} alpha {alpha}", m.name());
        }
        assert!(constellation(Modulation::Qpsk, 2).is_err());
    }

    #[test]
    fn the_tps_code_has_the_generator_of_sec_4_6_2_11() {
        // h(x) = x^14 + x^9 + x^8 + x^6 + x^5 + x^4 + x^2 + x + 1, lowest degree first
        let h = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1];
        let code = tps_code().unwrap();
        assert_eq!(code.generator(), h);
        assert_eq!((code.n(), code.k()), (67, 53));
    }

    #[test]
    fn the_tps_round_trip_and_carry_the_synchronisation_words() {
        let link = Dvbt::new(Mode::K8, Guard::G1_8, Modulation::Qam64, InnerRate::R5_6).unwrap();
        for f in 0..4 {
            let mut tps = link.tps(f);
            tps.cell_id = Some(0xAB12);
            let bits = tps.bits().unwrap();
            assert_eq!(bits.len(), 68);
            let sync: Vec<u8> = TPS_SYNC.iter().map(|&b| b ^ u8::from(f % 2 == 1)).collect();
            assert_eq!(&bits[1..17], &sync[..], "frame {f}");
            let mut noisy = bits.clone();
            noisy[5] ^= 1;
            noisy[60] ^= 1; // two errors
            let back = Tps::decode(&noisy).unwrap();
            assert_eq!(back.cell_id, Some(if f % 2 == 0 { 0xAB00 } else { 0x0012 }));
            assert_eq!(Tps { cell_id: tps.cell_id, ..back }, tps);
        }
    }

    #[test]
    fn the_packets_and_bit_rates_of_tables_16_and_17() {
        // Table 16, 2K and 8K: packets per super-frame
        let table16 = [
            (Modulation::Qpsk, [252, 336, 378, 420, 441]),
            (Modulation::Qam16, [504, 672, 756, 840, 882]),
            (Modulation::Qam64, [756, 1008, 1134, 1260, 1323]),
        ];
        for (m, row) in table16 {
            for (r, n) in InnerRate::ALL.into_iter().zip(row) {
                let l2 = Dvbt::new(Mode::K2, Guard::G1_4, m, r).unwrap();
                let l8 = Dvbt::new(Mode::K8, Guard::G1_4, m, r).unwrap();
                assert_eq!((l2.packets_per_superframe(), l8.packets_per_superframe()), (n, 4 * n));
            }
        }
        // Table 17 (8 MHz, Mbit/s), 64-QAM 7/8: 26.13, 29.03, 30.74, 31.67
        for (g, ru) in Guard::ALL.into_iter().zip([26.13, 29.03, 30.74, 31.67]) {
            let link = Dvbt::new(Mode::K8, g, Modulation::Qam64, InnerRate::R7_8).unwrap();
            assert!((link.useful_bit_rate(8e6) / 1e6 - ru).abs() < 0.005, "guard {}", g.name());
        }
    }

    #[test]
    fn a_burst_round_trips_through_the_ofdm_signal() {
        let packets = dvb::random_packets(20, &mut Rng::new(2));
        for (m, r) in [(Modulation::Qpsk, InnerRate::R1_2), (Modulation::Qam64, InnerRate::R7_8)] {
            let link = Dvbt::new(Mode::K2, Guard::G1_32, m, r).unwrap();
            let x = link.modulate(&packets).unwrap();
            let n = link.symbols_len(20);
            assert_eq!(x.len(), n * (2048 + 64));
            let cells = link.demodulate(&x).unwrap();
            assert_eq!(cells.len(), n);
            let d = link.decode_cells(&cells, 20, 1e-4, None).unwrap();
            assert_eq!(d.packets, packets, "{} {}", m.name(), r.name());
        }
    }

    #[test]
    fn the_tps_decode_from_the_cells_of_a_frame() {
        let link = Dvbt::new(Mode::K2, Guard::G1_4, Modulation::Qam16, InnerRate::R3_4).unwrap();
        let n_packets = 400; // enough for one frame of 68 symbols
        let cells = link.cells(&dvb::burst_encode(&dvb::random_packets(n_packets, &mut Rng::new(3))).unwrap()).unwrap();
        assert!(cells.len() >= 68);
        let tps = link.decode_tps(&cells[..68], None).unwrap();
        assert_eq!(tps, link.tps(0));
    }
}
