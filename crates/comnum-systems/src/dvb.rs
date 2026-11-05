//! The channel coding that DVB-S (ETSI EN 300 421) and DVB-T (ETSI EN 300 744) share:
//! the energy dispersal of the MPEG-2 transport stream, the outer Reed–Solomon code, the
//! outer convolutional interleaver and the punctured inner convolutional code.
//!
//! # Model
//!
//! The input is a stream of MPEG-2 transport packets of [`PACKET_LEN`] = 188 bytes, each
//! starting with the sync byte [`SYNC`] = `0x47`; every byte is processed most
//! significant bit first (EN 300 421, Sec. 4.4.1).
//!
//! 1. **Energy dispersal** ([`energy_dispersal`], EN 300 421 Sec. 4.4.1, Figure 2): the
//!    PRBS $1 + X^{14} + X^{15}$ ([`dispersal_prbs`]), loaded with
//!    `100101010000000` in its stages 1 to 15 at the start of every group of
//!    [`GROUP`] = 8 packets, is added to every byte but the sync bytes. The sync byte of
//!    the first packet of a group becomes [`INVERTED_SYNC`] = `0xB8`; the generator is
//!    not clocked during it, so that its first bit goes to the most significant bit of
//!    the next byte, and it keeps running, its output disabled, during the sync bytes of
//!    packets 2 to 8: a period of $8 \times 188 - 1 = 1503$ bytes.
//! 2. **Outer code** ([`outer_code`], Sec. 4.4.2): the shortened RS(204, 188, $T = 8$)
//!    code, the (255, 239) code over $\mathrm{GF}(2^{8})$ of field polynomial
//!    $p(x) = x^{8} + x^{4} + x^{3} + x^{2} + 1$ and generator
//!    $g(x) = (x + \lambda^{0})(x + \lambda^{1}) \cdots (x + \lambda^{15})$,
//!    $\lambda = 02_{16}$, with 51 leading zero bytes removed; 16 parity bytes follow the
//!    188 bytes of every packet ([`CODED_PACKET_LEN`] = 204).
//! 3. **Outer interleaver** ([`outer_interleaver`], Sec. 4.4.2, Figure 4): the
//!    convolutional byte interleaver of $I = 12$ branches of unit depth $M = 17$
//!    ([`ConvolutionalInterleaver`]); since $204 = 12 \times 17$, the sync bytes always
//!    take branch 0, without delay.
//! 4. **Inner code** ([`inner_trellis`], [`InnerRate`], Sec. 4.4.3, Table 2): the
//!    rate-1/2 code of constraint length 7, output $X$ from $G_{1} = 171_{8}$ and $Y$
//!    from $G_{2} = 133_{8}$, $X$ before $Y$ at every step, punctured to 2/3, 3/4, 5/6 or
//!    7/8 by the patterns of Table 2.
//!
//! The DVB-S chain ([`crate::dvbs`]) adds the QPSK mapping; DVB-T adds the inner
//! interleaver, its constellations and the OFDM frame.
//!
//! # Examples
//! ```
//! use comnum_systems::dvb;
//! // eight null packets: the randomised payload is the PRBS itself
//! let mut packets = vec![0u8; 8 * dvb::PACKET_LEN];
//! packets.iter_mut().step_by(dvb::PACKET_LEN).for_each(|b| *b = dvb::SYNC);
//! let r = dvb::energy_dispersal(&packets)?;
//! assert_eq!(r[0], dvb::INVERTED_SYNC);
//! assert_eq!(r[1], 0x03);                    // "PRBS sequence: 0 0 0 0 0 0 1 1", Figure 2
//! assert_eq!(r[188], dvb::SYNC);             // packet 2: its sync byte is not randomised
//! assert_eq!(dvb::energy_dispersal_removal(&r)?, packets);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. ETSI EN 300 421 V1.1.2, *Digital Video Broadcasting (DVB); Framing structure,
//!    channel coding and modulation for 11/12 GHz satellite services*, 1997, Sec. 4.4.
//! 2. ETSI EN 300 744 V1.6.2, *Digital Video Broadcasting (DVB); Framing structure,
//!    channel coding and modulation for digital terrestrial television*, 2015, Sec. 4.3.

use comnum_core::rng::Rng;
use comnum_core::{Error, Result, ensure};
use comnum_fec::conv::Trellis;
use comnum_fec::interleaving::ConvolutionalInterleaver;
use comnum_fec::puncture::Puncturer;
use comnum_fec::rs::ReedSolomon;
use comnum_math::sequences::lfsr;
use std::str::FromStr;

/// Bytes of an MPEG-2 transport packet.
pub const PACKET_LEN: usize = 188;
/// Bytes of a packet after the outer code RS(204, 188).
pub const CODED_PACKET_LEN: usize = 204;
/// The sync byte of a transport packet.
pub const SYNC: u8 = 0x47;
/// The sync byte of the first packet of a group, $\overline{47_{16}} = B8_{16}$.
pub const INVERTED_SYNC: u8 = 0xB8;
/// Packets of a group of the energy dispersal (the period of the PRBS reset).
pub const GROUP: usize = 8;
/// Branches $I$ of the outer convolutional interleaver.
pub const INTERLEAVER_BRANCHES: usize = 12;
/// Unit depth $M = 204 / I$ of the outer convolutional interleaver.
pub const INTERLEAVER_UNIT: usize = 17;

/// Bytes of PRBS between two resets: $8 \times 188 - 1$.
const DISPERSAL_PERIOD: usize = GROUP * PACKET_LEN - 1;

/// `n` bits of the PRBS $1 + X^{14} + X^{15}$ loaded with `100101010000000` in its
/// stages 1 to 15 (EN 300 421, Sec. 4.4.1, Figure 2; also the BBFRAME scrambler of
/// DVB-S2, EN 302 307-1 Sec. 5.2.2, and the energy dispersal of DVB-T, EN 300 744
/// Sec. 4.3.1).
///
/// Stage 1 takes the sum of stages 14 and 15, which is also the output bit:
/// $p_{n} = p_{n-14} \oplus p_{n-15}$.
///
/// # Examples
/// ```
/// // the first byte, 0000 0011, printed under Figure 2 of EN 300 421
/// assert_eq!(&comnum_systems::dvb::dispersal_prbs(8), &[0, 0, 0, 0, 0, 0, 1, 1]);
/// ```
pub fn dispersal_prbs(n: usize) -> Vec<u8> {
    // stages 15 … 1 hold the last 15 outputs, the oldest (stage 15) first
    const INIT: [u8; 15] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1];
    lfsr(&[0, 1], 15, &INIT, n + 15).split_off(15)
}

/// The PRBS of one group, as the bytes it adds to bytes 1 to 1503 of the group.
fn dispersal_bytes() -> Vec<u8> {
    dispersal_prbs(8 * DISPERSAL_PERIOD).chunks(8).map(|b| b.iter().fold(0u8, |acc, &v| (acc << 1) | v)).collect()
}

/// [`Error::Shape`] unless `len` bytes make whole transport packets.
fn whole_packets(len: usize, packet: usize) -> Result<()> {
    ensure(len % packet == 0, || format!("{len} bytes is not a multiple of the packet length {packet}"))
        .map_err(|e| Error::Shape(e.to_string()))
}

/// The energy dispersal of a stream of whole transport packets (see the module): the
/// PRBS added to every byte but the sync bytes, reset every 8 packets, the sync byte of
/// the first packet of every group inverted.
///
/// # Errors
/// [`Error::Shape`] if the length is not a multiple of [`PACKET_LEN`];
/// [`Error::Config`] if a packet does not start with [`SYNC`].
pub fn energy_dispersal(packets: &[u8]) -> Result<Vec<u8>> {
    whole_packets(packets.len(), PACKET_LEN)?;
    if let Some(p) = packets.chunks(PACKET_LEN).position(|p| p[0] != SYNC) {
        return Err(Error::Config(format!("packet {p} does not start with the sync byte 0x47")));
    }
    Ok(disperse(packets, INVERTED_SYNC))
}

/// The inverse of [`energy_dispersal`]: the PRBS removed and the inverted sync bytes
/// restored to [`SYNC`] (the de-randomiser of Annex B of EN 300 421, the same circuit).
/// A byte that should be a sync byte is left as received.
///
/// # Errors
/// [`Error::Shape`] if the length is not a multiple of [`PACKET_LEN`].
pub fn energy_dispersal_removal(stream: &[u8]) -> Result<Vec<u8>> {
    whole_packets(stream.len(), PACKET_LEN)?;
    let mut out = disperse(stream, SYNC);
    // a received inverted sync byte goes back to 0x47; another value is left as is
    for (g, group) in stream.chunks(GROUP * PACKET_LEN).enumerate() {
        if group[0] != INVERTED_SYNC {
            out[g * GROUP * PACKET_LEN] = group[0];
        }
    }
    Ok(out)
}

/// The PRBS added to every byte of `x` but the sync bytes; the first sync byte of every
/// group becomes `first_sync`.
fn disperse(x: &[u8], first_sync: u8) -> Vec<u8> {
    let prbs = dispersal_bytes();
    x.chunks(GROUP * PACKET_LEN)
        .flat_map(|group| {
            group.iter().enumerate().map(|(k, &b)| match k {
                0 => first_sync,
                _ if k % PACKET_LEN == 0 => b,
                _ => b ^ prbs[k - 1],
            })
        })
        .collect()
}

/// `n` MPEG-2 transport packets for a simulation: the sync byte [`SYNC`] and 187 random
/// bytes each.
pub fn random_packets(n: usize, rng: &mut Rng) -> Vec<u8> {
    (0..n * PACKET_LEN).map(|i| if i % PACKET_LEN == 0 { SYNC } else { rng.below(256) as u8 }).collect()
}

/// The outer code, the shortened RS(204, 188, $T = 8$) of EN 300 421 Sec. 4.4.2 and
/// EN 300 744 Sec. 4.3.2.
///
/// # Errors
/// Never: the parameters are those of the standard.
pub fn outer_code() -> Result<ReedSolomon> {
    ReedSolomon::with_field(8, 0x11D, 0, 1, 8, CODED_PACKET_LEN)
}

/// Encode a stream of whole 188-byte packets with RS(204, 188): 16 parity bytes after
/// every packet.
///
/// # Errors
/// [`Error::Shape`] if the length is not a multiple of [`PACKET_LEN`].
pub fn outer_encode(stream: &[u8]) -> Result<Vec<u8>> {
    whole_packets(stream.len(), PACKET_LEN)?;
    let rs = outer_code()?;
    let mut out = Vec::with_capacity(stream.len() / PACKET_LEN * CODED_PACKET_LEN);
    for p in stream.chunks(PACKET_LEN) {
        let msg: Vec<u16> = p.iter().map(|&b| u16::from(b)).collect();
        out.extend(rs.encode_symbols(&msg)?.into_iter().map(|s| s as u8));
    }
    Ok(out)
}

/// The decoding of [`outer_decode`]: the packets, then the counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterDecoded {
    /// The 188 bytes of every packet (as received when its codeword was not decoded).
    pub packets: Vec<u8>,
    /// Bytes corrected in all.
    pub corrected: usize,
    /// Packets whose errors exceeded the code (detected failures).
    pub failed: usize,
}

/// Decode a stream of whole 204-byte codewords of RS(204, 188).
///
/// # Errors
/// [`Error::Shape`] if the length is not a multiple of [`CODED_PACKET_LEN`].
pub fn outer_decode(coded: &[u8]) -> Result<OuterDecoded> {
    whole_packets(coded.len(), CODED_PACKET_LEN)?;
    let rs = outer_code()?;
    let mut d = OuterDecoded { packets: Vec::with_capacity(coded.len()), corrected: 0, failed: 0 };
    for c in coded.chunks(CODED_PACKET_LEN) {
        let r: Vec<u16> = c.iter().map(|&b| u16::from(b)).collect();
        let (msg, fixed) = rs.decode_symbols(&r)?;
        match fixed {
            Some(n) => d.corrected += n,
            None => d.failed += 1,
        }
        d.packets.extend(msg.into_iter().map(|s| s as u8));
    }
    Ok(d)
}

/// The outer convolutional interleaver, $I = 12$, $M = 17$ (EN 300 421 Sec. 4.4.2).
///
/// # Errors
/// Never: the parameters are those of the standard.
pub fn outer_interleaver() -> Result<ConvolutionalInterleaver<u8>> {
    ConvolutionalInterleaver::new(INTERLEAVER_BRANCHES, INTERLEAVER_UNIT)
}

/// The outer convolutional deinterleaver, $I = 12$, $M = 17$.
///
/// # Errors
/// Never: the parameters are those of the standard.
pub fn outer_deinterleaver() -> Result<ConvolutionalInterleaver<u8>> {
    ConvolutionalInterleaver::deinterleaver(INTERLEAVER_BRANCHES, INTERLEAVER_UNIT)
}

/// Bytes of zeros that flush the outer interleaver: $I - 1$ packets of 204 bytes, its
/// latency $I M (I - 1) = 2244$ bytes.
pub const FLUSH_BYTES: usize = (INTERLEAVER_BRANCHES - 1) * CODED_PACKET_LEN;

/// Bytes of [`burst_encode`] for a burst of `n_packets` packets: the coded packets and
/// the flush of the interleaver.
pub fn burst_bytes(n_packets: usize) -> usize {
    n_packets * CODED_PACKET_LEN + FLUSH_BYTES
}

/// The outer coding of a burst of whole transport packets: the energy dispersal,
/// RS(204, 188), then the outer interleaver from its initial state, flushed by the 11
/// packets of zeros of [`FLUSH_BYTES`], so that a receiver decodes every packet of the
/// burst; the bytes that enter the inner code ([`burst_bytes`] of them).
///
/// # Errors
/// As [`energy_dispersal`].
pub fn burst_encode(packets: &[u8]) -> Result<Vec<u8>> {
    let mut coded = outer_encode(&energy_dispersal(packets)?)?;
    coded.resize(coded.len() + FLUSH_BYTES, 0);
    Ok(outer_interleaver()?.process(&coded))
}

/// The packets of the bytes of a burst ([`burst_encode`]): the outer deinterleaver, the
/// flush dropped, RS(204, 188) and the removal of the energy dispersal.
///
/// # Errors
/// [`Error::Shape`] unless `bytes` is a burst of at least one packet.
pub fn burst_decode(bytes: &[u8]) -> Result<OuterDecoded> {
    let n = bytes.len().checked_sub(FLUSH_BYTES).filter(|&c| c > 0 && c % CODED_PACKET_LEN == 0);
    let Some(coded) = n else {
        return Err(Error::Shape(format!("{} bytes is not a burst of whole packets", bytes.len())));
    };
    let deinterleaved = outer_deinterleaver()?.process(bytes);
    let mut d = outer_decode(&deinterleaved[FLUSH_BYTES..FLUSH_BYTES + coded])?;
    d.packets = energy_dispersal_removal(&d.packets)?;
    Ok(d)
}

/// The inner code before puncturing: constraint length 7, the output $X$ from
/// $G_{1} = 171_{8}$ then the output $Y$ from $G_{2} = 133_{8}$ at every step (EN 300 421,
/// Sec. 4.4.3, Table 2; EN 300 744, Sec. 4.3.3).
///
/// [`comnum_fec::conv::ConvolutionalCode::k7`] is the same code with the outputs in the
/// other order ($133_{8}$ first, IEEE 802.11).
///
/// # Errors
/// Never: the parameters are those of the standard.
pub fn inner_trellis() -> Result<Trellis> {
    Trellis::feedforward(7, &[0o171, 0o133])
}

/// The rates of the punctured inner code (EN 300 421 Table 2, EN 300 744 Table 2).
///
/// | Rate | $X$ | $Y$ | Transmitted order | $d_{\mathrm{free}}$ |
/// | --- | --- | --- | --- | --- |
/// | 1/2 | 1 | 1 | $X_{1} Y_{1}$ | 10 |
/// | 2/3 | 1 0 | 1 1 | $X_{1} Y_{1} Y_{2}$ | 6 |
/// | 3/4 | 1 0 1 | 1 1 0 | $X_{1} Y_{1} Y_{2} X_{3}$ | 5 |
/// | 5/6 | 1 0 1 0 1 | 1 1 0 1 0 | $X_{1} Y_{1} Y_{2} X_{3} Y_{4} X_{5}$ | 4 |
/// | 7/8 | 1 0 0 0 1 0 1 | 1 1 1 1 0 1 0 | $X_{1} Y_{1} Y_{2} Y_{3} Y_{4} X_{5} Y_{6} X_{7}$ | 3 |
///
/// The kept bits leave in time order, $X$ before $Y$ at every step: DVB-S then sends
/// them alternately on I and Q, which gives the I and Q rows of its Table 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InnerRate {
    /// 1/2, the mother code.
    R1_2,
    /// 2/3.
    R2_3,
    /// 3/4.
    R3_4,
    /// 5/6.
    R5_6,
    /// 7/8.
    R7_8,
}

impl InnerRate {
    /// The five rates, in the order of Table 2.
    pub const ALL: [InnerRate; 5] =
        [InnerRate::R1_2, InnerRate::R2_3, InnerRate::R3_4, InnerRate::R5_6, InnerRate::R7_8];

    /// The rate as a number, $P / w$.
    pub fn value(self) -> f64 {
        let (num, den) = self.fraction();
        num as f64 / den as f64
    }

    /// The rate as a fraction: input bits and transmitted bits of a period.
    pub fn fraction(self) -> (usize, usize) {
        match self {
            InnerRate::R1_2 => (1, 2),
            InnerRate::R2_3 => (2, 3),
            InnerRate::R3_4 => (3, 4),
            InnerRate::R5_6 => (5, 6),
            InnerRate::R7_8 => (7, 8),
        }
    }

    /// The rows $X$ and $Y$ of the puncturing pattern of Table 2 (1 transmitted).
    pub fn pattern(self) -> [&'static [u8]; 2] {
        match self {
            InnerRate::R1_2 => [&[1], &[1]],
            InnerRate::R2_3 => [&[1, 0], &[1, 1]],
            InnerRate::R3_4 => [&[1, 0, 1], &[1, 1, 0]],
            InnerRate::R5_6 => [&[1, 0, 1, 0, 1], &[1, 1, 0, 1, 0]],
            InnerRate::R7_8 => [&[1, 0, 0, 0, 1, 0, 1], &[1, 1, 1, 1, 0, 1, 0]],
        }
    }

    /// The puncturer of the pattern, for the outputs of [`inner_trellis`] ($X$ first).
    ///
    /// # Errors
    /// Never: the patterns are those of the standard.
    pub fn puncturer(self) -> Result<Puncturer> {
        Puncturer::new(&self.pattern())
    }

    /// The name of the rate, `1/2` to `7/8`.
    pub fn name(self) -> &'static str {
        match self {
            InnerRate::R1_2 => "1/2",
            InnerRate::R2_3 => "2/3",
            InnerRate::R3_4 => "3/4",
            InnerRate::R5_6 => "5/6",
            InnerRate::R7_8 => "7/8",
        }
    }
}

impl FromStr for InnerRate {
    type Err = Error;

    /// `1/2`, `2/3`, `3/4`, `5/6` or `7/8`.
    fn from_str(s: &str) -> Result<InnerRate> {
        InnerRate::ALL
            .into_iter()
            .find(|r| r.name() == s.trim())
            .ok_or_else(|| Error::Config(format!("the inner code rate must be 1/2, 2/3, 3/4, 5/6 or 7/8, got {s:?}")))
    }
}

/// The bits of bytes, the most significant first.
pub(crate) fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |k| (b >> k) & 1)).collect()
}

/// The bytes of bits (a multiple of 8), the first bit the most significant.
pub(crate) fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8).map(|b| b.iter().fold(0u8, |acc, &v| (acc << 1) | (v & 1))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_fec::puncture::free_distance;

    fn null_packets(n: usize) -> Vec<u8> {
        (0..n * PACKET_LEN).map(|i| if i % PACKET_LEN == 0 { SYNC } else { 0 }).collect()
    }

    #[test]
    fn the_prbs_starts_with_the_byte_printed_under_figure_2() {
        // EN 300 421, Figure 2: "PRBS sequence: 0 0 0 0 0 0 1 1 ...."
        assert_eq!(dispersal_bytes()[0], 0x03);
        // the sequence has the period 2^15 - 1 of a maximal-length register
        let p = dispersal_prbs(2 * 32_767);
        assert_eq!(&p[..32_767], &p[32_767..]);
    }

    #[test]
    fn the_dispersal_resets_every_eight_packets() {
        let r = energy_dispersal(&null_packets(16)).unwrap();
        let g = GROUP * PACKET_LEN;
        assert_eq!(&r[..g], &r[g..2 * g], "two groups randomised alike");
        for p in 0..16 {
            let expected = if p % GROUP == 0 { INVERTED_SYNC } else { SYNC };
            assert_eq!(r[p * PACKET_LEN], expected, "sync byte of packet {p}");
        }
        // the generator runs, output disabled, during the sync bytes of packets 2 to 8:
        // the payload of packet 2 starts with PRBS byte 188, not 187
        assert_eq!(r[PACKET_LEN + 1], dispersal_bytes()[PACKET_LEN]);
    }

    #[test]
    fn the_removal_undoes_the_dispersal() {
        let packets = random_packets(20, &mut Rng::new(1));
        assert_eq!(energy_dispersal_removal(&energy_dispersal(&packets).unwrap()).unwrap(), packets);
        let mut bad = packets.clone();
        bad[PACKET_LEN] = 0x48;
        assert!(energy_dispersal(&bad).is_err());
        assert!(energy_dispersal(&packets[..100]).is_err());
    }

    #[test]
    fn the_outer_code_corrects_eight_bytes_per_packet() {
        let packets = random_packets(3, &mut Rng::new(2));
        let mut coded = outer_encode(&packets).unwrap();
        assert_eq!(coded.len(), 3 * CODED_PACKET_LEN);
        assert_eq!(&coded[..PACKET_LEN], &packets[..PACKET_LEN], "systematic");
        for k in 0..8 {
            coded[k * 20] ^= 0x5A; // eight errors in the first packet
        }
        for k in 0..9 {
            coded[2 * CODED_PACKET_LEN + k * 20] ^= 0xFF; // nine in the third
        }
        let d = outer_decode(&coded).unwrap();
        assert_eq!(&d.packets[..2 * PACKET_LEN], &packets[..2 * PACKET_LEN]);
        assert_eq!((d.corrected, d.failed), (8, 1));
    }

    #[test]
    fn the_sync_bytes_take_the_branch_without_delay() {
        // 204 = 12 x 17: byte k of a codeword enters branch k mod 12
        let stream = outer_encode(&energy_dispersal(&null_packets(40)).unwrap()).unwrap();
        let y = outer_interleaver().unwrap().process(&stream);
        for p in 0..40 {
            assert_eq!(y[p * CODED_PACKET_LEN], stream[p * CODED_PACKET_LEN]);
        }
    }

    #[test]
    fn a_burst_round_trips_through_the_outer_coding() {
        let packets = random_packets(10, &mut Rng::new(5));
        let bytes = burst_encode(&packets).unwrap();
        assert_eq!(bytes.len(), burst_bytes(10));
        assert_eq!(burst_decode(&bytes).unwrap().packets, packets);
        assert!(burst_decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(burst_decode(&bytes[..FLUSH_BYTES]).is_err());
    }

    #[test]
    fn the_punctured_rates_have_the_free_distances_of_table_2() {
        let t = inner_trellis().unwrap();
        // Table 2 of EN 300 421: d_free = 10, 6, 5, 4, 3
        for (r, dfree) in InnerRate::ALL.into_iter().zip([10, 6, 5, 4, 3]) {
            let p = r.puncturer().unwrap();
            let (k, n) = r.fraction();
            assert_eq!((p.period(), p.kept_per_period()), (k, n), "rate {}", r.name());
            assert_eq!(free_distance(&t, &p).unwrap(), dfree, "rate {}", r.name());
            assert_eq!(r.name().parse::<InnerRate>().unwrap(), r);
        }
        assert!("4/5".parse::<InnerRate>().is_err());
    }

    #[test]
    fn the_transmitted_order_is_that_of_table_2() {
        // label the coded bits X_t, Y_t and read the kept ones in order
        let labels = |r: InnerRate| -> Vec<String> {
            let p = r.puncturer().unwrap();
            let steps = p.period();
            let names: Vec<String> = (1..=steps).flat_map(|t| [format!("X{t}"), format!("Y{t}")]).collect();
            let kept = p.selection(2 * steps).unwrap();
            kept.positions().iter().map(|&i| names[i].clone()).collect()
        };
        assert_eq!(labels(InnerRate::R3_4), ["X1", "Y1", "Y2", "X3"]);
        assert_eq!(labels(InnerRate::R7_8), ["X1", "Y1", "Y2", "Y3", "Y4", "X5", "Y6", "X7"]);
        assert_eq!(bits_to_bytes(&bytes_to_bits(&[0x47, 0xB8])), [0x47, 0xB8]);
        assert_eq!(&bytes_to_bits(&[0x47])[..], &[0, 1, 0, 0, 0, 1, 1, 1]);
    }
}
