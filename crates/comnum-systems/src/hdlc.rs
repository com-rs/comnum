//! HDLC framing (ISO/IEC 13239), the link layer of AIS, AX.25 and X.25: the flag
//! sequence, bit stuffing, the 16-bit frame check sequence and NRZI line coding.
//!
//! # Model
//!
//! A frame is the flag `01111110`, the bytes of the payload each sent least significant
//! bit first, the two bytes of the frame check sequence (low byte first) and the flag.
//! Between the flags, a 0 is inserted after every five consecutive 1 bits (bit stuffing),
//! so that the flag never occurs in the data; the receiver removes it, and six or more 1
//! bits in a row outside a flag abort the frame.
//!
//! The frame check sequence is the CRC-16 of X.25 (the "FCS-16" of ISO/IEC 13239
//! Sec. 4.2.7.2.1): generator $x^{16} + x^{12} + x^{5} + 1$, the bits of every byte
//! taken least significant first, the register initialised to all ones and the
//! result complemented; in the catalogue of Cook it is CRC-16/X-25, with the check
//! value `0x906E` for the ASCII string "123456789" and the residue `0xF0B8` of a
//! correct frame. [`fcs`] computes it and [`frames`] checks it; the frames that fail
//! are returned with [`Frame::fcs_ok`] false, so that a caller can count them.
//!
//! NRZI (non-return to zero, inverted) codes a 0 as a transition of the line level and a
//! 1 as no transition ([`nrzi_encode`], [`nrzi_decode`]): the data does not depend on the
//! polarity of the line, which AIS relies on (ITU-R M.1371-5, Annex 2, Sec. 3.3.2.3).
//!
//! # Examples
//! ```
//! use comnum_systems::hdlc::{frame, frames, nrzi_decode, nrzi_encode};
//! let payload = [0x12u8, 0x7F, 0xFF, 0x00];
//! let line = nrzi_encode(&frame(&payload), 0);
//! let received = frames(&nrzi_decode(&line));
//! assert_eq!(received.len(), 1);
//! assert!(received[0].fcs_ok);
//! assert_eq!(received[0].payload, payload);
//! ```
//!
//! # References
//!
//! 1. ISO/IEC 13239:2002, *High-level data link control (HDLC) procedures*, Secs. 4.2.2
//!    (flag), 4.2.7 (frame check sequence), 4.3.2 (transparency, bit stuffing).
//! 2. G. Cook, *Catalogue of parametrised CRC algorithms*, CRC-16/X-25 (check `0x906E`,
//!    residue `0xF0B8`).
//! 3. Recommendation ITU-R M.1371-5, *Technical characteristics for an automatic
//!    identification system using time division multiple access in the VHF maritime
//!    mobile band*, 2014, Annex 2, Sec. 3.3.2 (NRZI, bit stuffing, the packet format).

/// The flag sequence `01111110`.
pub const FLAG: [u8; 8] = [0, 1, 1, 1, 1, 1, 1, 0];

/// The frame check sequence of `bytes`: CRC-16/X-25 (ISO/IEC 13239 FCS-16).
///
/// # Examples
/// ```
/// use comnum_systems::hdlc::fcs;
/// assert_eq!(fcs(b"123456789"), 0x906E);
/// ```
pub fn fcs(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in bytes {
        crc ^= u16::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0x8408 } else { crc >> 1 };
        }
    }
    !crc
}

/// The bits of `bytes`, every byte least significant bit first (the order of the line).
pub fn bits_lsb_first(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&b| (0..8).map(move |i| (b >> i) & 1)).collect()
}

/// The bytes of `bits` taken eight at a time, the first bit of each group the least
/// significant; a last group of fewer than eight bits is dropped.
pub fn bytes_lsb_first(bits: &[u8]) -> Vec<u8> {
    bits.chunks_exact(8).map(|c| c.iter().enumerate().fold(0u8, |b, (i, &v)| b | ((v & 1) << i))).collect()
}

/// Bit stuffing: a 0 inserted after every five consecutive 1 bits.
pub fn stuff(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() + bits.len() / 5);
    let mut ones = 0;
    for &b in bits {
        out.push(b);
        if b == 1 {
            ones += 1;
            if ones == 5 {
                out.push(0);
                ones = 0;
            }
        } else {
            ones = 0;
        }
    }
    out
}

/// The inverse of [`stuff`]: the 0 after five consecutive 1 bits is removed; `None` if
/// six 1 bits occur in a row (an abort, or a flag).
pub fn unstuff(bits: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(bits.len());
    let mut ones = 0;
    let mut skip = false;
    for &b in bits {
        if skip {
            skip = false;
            if b != 0 {
                return None;
            }
            continue;
        }
        out.push(b);
        if b == 1 {
            ones += 1;
            if ones == 5 {
                skip = true;
                ones = 0;
            }
        } else {
            ones = 0;
        }
    }
    Some(out)
}

/// NRZI: the line levels of `bits` from the level `start` (0 or 1), a 0 toggling the
/// level and a 1 keeping it; one more level than bits.
pub fn nrzi_encode(bits: &[u8], start: u8) -> Vec<u8> {
    let mut level = start & 1;
    let mut out = Vec::with_capacity(bits.len() + 1);
    out.push(level);
    for &b in bits {
        if b == 0 {
            level ^= 1;
        }
        out.push(level);
    }
    out
}

/// The inverse of [`nrzi_encode`]: a 1 where two consecutive levels are equal, a 0
/// where they differ; one bit fewer than levels.
pub fn nrzi_decode(levels: &[u8]) -> Vec<u8> {
    levels.windows(2).map(|w| u8::from((w[0] & 1) == (w[1] & 1))).collect()
}

/// The bits of one frame of `payload`: the flag, the stuffed bits of the payload and
/// of its frame check sequence (low byte first), and the flag.
pub fn frame(payload: &[u8]) -> Vec<u8> {
    let crc = fcs(payload);
    let mut bytes = payload.to_vec();
    bytes.push((crc & 0xFF) as u8);
    bytes.push((crc >> 8) as u8);
    let mut out = FLAG.to_vec();
    out.extend(stuff(&bits_lsb_first(&bytes)));
    out.extend(FLAG);
    out
}

/// A frame found by [`frames`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// The bytes between the flags, the frame check sequence removed.
    pub payload: Vec<u8>,
    /// The position of the first bit of the opening flag in the stream.
    pub start: usize,
    /// Whether the frame check sequence was correct.
    pub fcs_ok: bool,
}

/// The frames of a stream of (NRZI-decoded) bits: the data between two flags, unstuffed,
/// a whole number of bytes of at least the two of the frame check sequence; `min_bytes`
/// payload bytes at least.
pub fn frames(bits: &[u8]) -> Vec<Frame> {
    frames_of_at_least(bits, 1)
}

/// [`frames`] with a smallest payload of `min_bytes` bytes.
pub fn frames_of_at_least(bits: &[u8], min_bytes: usize) -> Vec<Frame> {
    let flags: Vec<usize> = (0..bits.len().saturating_sub(7)).filter(|&i| bits[i..i + 8] == FLAG).collect();
    let mut out = Vec::new();
    for w in flags.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b < a + 8 {
            continue;
        }
        let Some(data) = unstuff(&bits[a + 8..b]) else { continue };
        if data.len() % 8 != 0 || data.len() < 8 * (min_bytes + 2) {
            continue;
        }
        let bytes = bytes_lsb_first(&data);
        let n = bytes.len();
        let received = u16::from(bytes[n - 2]) | (u16::from(bytes[n - 1]) << 8);
        out.push(Frame { payload: bytes[..n - 2].to_vec(), start: a, fcs_ok: fcs(&bytes[..n - 2]) == received });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stuffing_round_trips_and_never_leaves_six_ones() {
        let bits: Vec<u8> =
            [1u8; 17].iter().chain([0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1].iter()).copied().collect();
        let s = stuff(&bits);
        assert!(s.windows(6).all(|w| w != [1, 1, 1, 1, 1, 1]));
        assert_eq!(unstuff(&s).unwrap(), bits);
        assert_eq!(unstuff(&[1, 1, 1, 1, 1, 1]), None);
    }

    #[test]
    fn nrzi_round_trips_from_either_polarity() {
        let bits = [0u8, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1];
        for start in [0, 1] {
            assert_eq!(nrzi_decode(&nrzi_encode(&bits, start)), bits);
        }
        let inverted: Vec<u8> = nrzi_encode(&bits, 0).iter().map(|v| v ^ 1).collect();
        assert_eq!(nrzi_decode(&inverted), bits);
    }

    #[test]
    fn a_frame_is_found_between_its_flags_and_a_corrupted_one_fails_the_fcs() {
        let payload = [0xC0u8, 0x3E, 0xFF, 0xFF, 0x7E, 0x01];
        let mut stream = vec![0, 1, 0, 1, 0, 1];
        stream.extend(frame(&payload));
        stream.extend([1, 1, 0, 0]);
        stream.extend(frame(&[0x55]));
        let f = frames(&stream);
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].payload, payload);
        assert!(f[0].fcs_ok && f[1].fcs_ok);
        assert_eq!(f[0].start, 6);
        let mut corrupted = stream.clone();
        corrupted[6 + 8 + 3] ^= 1;
        let f = frames(&corrupted);
        assert_eq!(f.len(), 2);
        assert!(!f[0].fcs_ok && f[1].fcs_ok);
        // two frames sharing a flag
        let mut shared = frame(&payload);
        shared.extend(&frame(&[0x55])[8..]);
        assert_eq!(frames(&shared).len(), 2);
        assert_eq!(frames_of_at_least(&shared, 2).len(), 1);
    }

    #[test]
    fn the_bytes_of_the_line_are_least_significant_bit_first() {
        assert_eq!(bits_lsb_first(&[0x01, 0x80]), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(bytes_lsb_first(&bits_lsb_first(&[0x3A, 0xF0, 0x81])), [0x3A, 0xF0, 0x81]);
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    /// G. Cook, *Catalogue of parametrised CRC algorithms*, CRC-16/X-25: check 0x906E,
    /// residue 0xF0B8 (the register before the final complement, of a frame followed by
    /// its FCS, low byte first); ISO/IEC 13239 Sec. 4.2.7.2.1.
    #[test]
    fn cook_crc16_x25_check_and_residue() {
        assert_eq!(fcs(b"123456789"), 0x906E);
        let crc = fcs(b"123456789");
        let mut with_fcs = b"123456789".to_vec();
        with_fcs.push((crc & 0xFF) as u8);
        with_fcs.push((crc >> 8) as u8);
        assert_eq!(!fcs(&with_fcs), 0xF0B8);
    }
}
