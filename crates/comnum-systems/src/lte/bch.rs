//! The broadcast channel of LTE: the 24 bits of the master information block, their
//! CRC masked by the number of antenna ports, the tail-biting convolutional code and the
//! rate matching to the 1920 bits of a PBCH period of 40 ms (TS 36.212, Sec. 5.3.1).
//!
//! # Transmitter
//!
//! 1. CRC attachment (Sec. 5.3.1.1): the 16 parity bits of $g_{\mathrm{CRC16}}(D) = D^{16} +
//!    D^{12} + D^{5} + 1$ ([`comnum_fec::crc::Crc::crc16`]) are added modulo 2 to the mask of
//!    the number of ports (Table 5.3.1.1-1): all zeros for one port, all ones for two,
//!    $\langle 0, 1, 0, 1, \dots \rangle$ for four, so that the receiver learns the number of
//!    ports from the CRC that passes; $K = 40$ bits.
//! 2. Channel coding (Sec. 5.1.3.1): the tail-biting code of constraint length 7 and
//!    generators $G_{0} = 133$, $G_{1} = 171$, $G_{2} = 165$ (octal), three streams
//!    $d^{(0)}, d^{(1)}, d^{(2)}$ of $D = 40$ bits
//!    ([`comnum_fec::conv::Trellis::encode_tail_biting`]).
//! 3. Rate matching (Sec. 5.1.4.2): every stream is written row by row in a matrix of 32
//!    columns and $R = \lceil D/32 \rceil$ rows, $32R - D$ dummy bits first, read column by
//!    column in the order $P = \langle 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15,
//!    31, 0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30 \rangle$ (Table
//!    5.1.4-2); the three outputs are concatenated into a circular buffer of $3 \cdot 32R$
//!    bits, read from the start, the dummy bits skipped, until $E$ bits are sent: $E =
//!    1920$ with the normal prefix, sixteen repetitions of the 120 coded bits
//!    ([`circular_buffer`]).
//!
//! The PBCH sends a quarter of the $E$ bits in every frame (480 bits, four repetitions):
//! one frame suffices to decode, and the quarter that decodes tells the position of the
//! frame in the 40 ms period.
//!
//! # Receiver
//!
//! [`decode`] adds the LLRs of the received bits onto the 120 coded bits they repeat,
//! decodes by the wrap-around Viterbi algorithm ([`comnum_fec::conv::viterbi_tail_biting`])
//! and checks the CRC under the three masks.
//!
//! # Examples
//! ```
//! use comnum_systems::lte::bch::{decode, encode, BCH_BITS};
//! let mib: Vec<u8> = (0..24).map(|i| (i % 3 == 0) as u8).collect();
//! let e = encode(&mib, 2)?;
//! assert_eq!(e.len(), BCH_BITS);
//! // the 480 bits of the third frame of the period, noiseless
//! let llr: Vec<f64> = e[960..1440].iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
//! let d = decode(&llr, 960, 4)?.expect("the CRC passes");
//! assert_eq!((d.bits, d.ports), (mib, 2));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 36.212 V18.1.0, *E-UTRA; Multiplexing and channel coding*, Sec. 5.1.1,
//!    5.1.3.1, 5.1.4.2, 5.3.1.

use comnum_core::{Error, Result, ensure_shape};
use comnum_fec::conv::{Trellis, viterbi_tail_biting};
use comnum_fec::crc::Crc;

/// Bits of the master information block, $A$.
pub const MIB_BITS: usize = 24;
/// Bits after the CRC attachment, $K = A + 16$.
pub const CODED_INPUT_BITS: usize = MIB_BITS + 16;
/// Bits of a PBCH period of 40 ms with the normal cyclic prefix, $E$.
pub const BCH_BITS: usize = 1920;
/// Bits of the PBCH in one frame, $E/4$.
pub const FRAME_BITS: usize = BCH_BITS / 4;

/// The inter-column permutation of the sub-block interleaver of convolutional codes,
/// Table 5.1.4-2.
pub const COLUMN_PERMUTATION: [usize; 32] = [
    1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14,
    30,
];

/// The convolutional code of LTE, $(133, 171, 165)_{8}$, constraint length 7.
///
/// # Errors
/// Never in practice (the generators are fixed).
pub fn trellis() -> Result<Trellis> {
    Trellis::feedforward(7, &[0o133, 0o171, 0o165])
}

/// The CRC mask of `ports` antenna ports (Table 5.3.1.1-1).
///
/// # Errors
/// [`Error::Config`] for another number of ports than 1, 2 or 4.
pub fn crc_mask(ports: usize) -> Result<[u8; 16]> {
    match ports {
        1 => Ok([0; 16]),
        2 => Ok([1; 16]),
        4 => Ok(std::array::from_fn(|i| (i % 2) as u8)),
        _ => Err(Error::Config(format!("an eNodeB has 1, 2 or 4 antenna ports, got {ports}"))),
    }
}

/// The input bit read by every position of the output of the sub-block interleaver of
/// a stream of `d` bits (`None` for a dummy bit), Sec. 5.1.4.2.1.
pub fn subblock_interleaver(d: usize) -> Vec<Option<usize>> {
    let rows = d.div_ceil(32);
    let dummies = 32 * rows - d;
    (0..32 * rows)
        .map(|out| {
            let (col, row) = (out / rows, out % rows);
            (row * 32 + COLUMN_PERMUTATION[col]).checked_sub(dummies)
        })
        .collect()
}

/// The circular buffer of the rate matching of a code of $D$ = `d` bits per stream, its
/// dummy bits skipped: the index, in the output of the encoder ($n$ bits per step, the
/// streams interleaved, $3q + i$ for the bit $q$ of the stream $i$), of every bit that
/// the buffer sends in turn. The bit $e_{k}$ is the encoder output at
/// `circular_buffer(d)[k % (3 d)]`.
pub fn circular_buffer(d: usize) -> Vec<usize> {
    let sub = subblock_interleaver(d);
    (0..3).flat_map(|i| sub.iter().flatten().map(move |&q| 3 * q + i)).collect()
}

/// The $E$ = [`BCH_BITS`] bits of a PBCH period that carry the master information block
/// `mib` (24 bits) from an eNodeB of `ports` antenna ports (see the module).
///
/// # Errors
/// [`Error::Shape`] for another number of bits than 24; [`Error::Config`] for a number
/// of ports other than 1, 2 or 4.
pub fn encode(mib: &[u8], ports: usize) -> Result<Vec<u8>> {
    ensure_shape(mib.len() == MIB_BITS, || format!("a MIB has {MIB_BITS} bits, got {}", mib.len()))?;
    let mask = crc_mask(ports)?;
    let mut c = Crc::crc16().attach(mib);
    for (b, m) in c[MIB_BITS..].iter_mut().zip(mask) {
        *b ^= m;
    }
    let coded = trellis()?.encode_tail_biting(&c)?;
    let buffer = circular_buffer(CODED_INPUT_BITS);
    Ok((0..BCH_BITS).map(|k| coded[buffer[k % buffer.len()]]).collect())
}

/// A decoded broadcast channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BchDecoded {
    /// The 24 bits of the master information block.
    pub bits: Vec<u8>,
    /// The number of antenna ports whose CRC mask passed.
    pub ports: usize,
}

/// The master information block of the LLRs `llr` of the rate-matched bits $e_{k}$,
/// $k = $ `offset`, `offset + 1`, …: their LLRs added onto the 120 coded bits, decoded by
/// `passes` passes of the wrap-around Viterbi algorithm, and checked under the CRC masks
/// of 1, 2 and 4 ports; `None` when no CRC passes.
///
/// # Errors
/// [`Error::Config`] for `passes = 0`.
pub fn decode(llr: &[f64], offset: usize, passes: usize) -> Result<Option<BchDecoded>> {
    let buffer = circular_buffer(CODED_INPUT_BITS);
    let mut acc = vec![0.0; 3 * CODED_INPUT_BITS];
    for (k, l) in llr.iter().enumerate() {
        acc[buffer[(offset + k) % buffer.len()]] += l;
    }
    let c = viterbi_tail_biting(&trellis()?, &acc, passes)?;
    let crc = Crc::crc16();
    for ports in [1, 2, 4] {
        let mut word = c.clone();
        for (b, m) in word[MIB_BITS..].iter_mut().zip(crc_mask(ports)?) {
            *b ^= m;
        }
        if crc.check(&word) {
            word.truncate(MIB_BITS);
            return Ok(Some(BchDecoded { bits: word, ports }));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_interleaver_of_forty_bits_has_twenty_four_dummies_read_first() {
        // D = 40: R = 2 rows, 24 dummy bits; column P(0) = 1 holds y_1 and y_33
        let sub = subblock_interleaver(40);
        assert_eq!(sub.len(), 64);
        assert_eq!(sub.iter().filter(|v| v.is_none()).count(), 24);
        assert_eq!(&sub[..2], &[None, Some(9)]); // y_1 is a dummy, y_33 = d_9
        let buffer = circular_buffer(40);
        assert_eq!(buffer.len(), 120);
        let mut sorted = buffer.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..120).collect::<Vec<_>>()); // every coded bit once
    }

    #[test]
    fn the_crc_mask_tells_the_number_of_ports() {
        let mib: Vec<u8> = (0..24).map(|i| ((i * 7) % 5 < 2) as u8).collect();
        for ports in [1, 2, 4] {
            let e = encode(&mib, ports).unwrap();
            // the 16 repetitions of the 120 coded bits
            assert!(e[..120] == e[120..240] && e[..120] == e[1800..]);
            let llr: Vec<f64> = e[480..960].iter().map(|&b| 2.0 * (1.0 - 2.0 * f64::from(b))).collect();
            assert_eq!(decode(&llr, 480, 2).unwrap(), Some(BchDecoded { bits: mib.clone(), ports }));
            // every quarter of the period starts at the same place of the buffer (480 is
            // a multiple of 120): the scrambling of the PBCH tells the frames apart
            assert_eq!(decode(&llr, 0, 2).unwrap().map(|d| d.ports), Some(ports));
        }
        assert!(crc_mask(3).is_err() && encode(&mib[1..], 1).is_err() && decode(&[0.0; 4], 0, 0).is_err());
    }
}
