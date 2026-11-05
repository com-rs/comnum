//! The `!AIVDM` sentences of NMEA 0183 (IEC 61162-1) that carry AIS messages: the
//! six-bit armouring of the payload, the fill bits, the checksum and the fragments.
//!
//! Six bits $v$ become the character of code $v + 48$ for $v < 40$ and $v + 56$ otherwise
//! (`0`–`W` then `` ` ``–`w`); the payload is padded to a multiple of six bits and the
//! number of padding bits is the fill-bits field. A sentence is
//! `!AIVDM,<fragments>,<fragment>,<sequence>,<channel>,<payload>,<fill>*<checksum>`,
//! the checksum the exclusive or of the characters between `!` and `*`, in two
//! hexadecimal digits; a message longer than 60 characters of payload is split into
//! fragments that share a sequence number.
//!
//! # References
//!
//! 1. IEC 61162-1:2016, *Maritime navigation and radiocommunication equipment and
//!    systems — Digital interfaces — Part 1*, the `VDM` sentence (Sec. 8.3.93) and the
//!    six-bit field (Sec. 7.3.3).
//! 2. Recommendation ITU-R M.1371-5, Annex 8, Table 46 (the six-bit ASCII of the text
//!    fields).

use comnum_core::{Result, ensure};

/// The longest payload of one sentence (characters).
pub const MAX_PAYLOAD_CHARS: usize = 60;

/// The six-bit armouring of `bits`: the characters and the number of fill bits.
pub fn armour(bits: &[u8]) -> (String, u8) {
    let fill = (6 - bits.len() % 6) % 6;
    let padded: Vec<u8> = bits.iter().map(|b| b & 1).chain(std::iter::repeat_n(0, fill)).collect();
    let s = padded
        .chunks_exact(6)
        .map(|c| {
            let v = c.iter().fold(0u8, |a, &b| (a << 1) | b);
            char::from(if v < 40 { v + 48 } else { v + 56 })
        })
        .collect();
    (s, fill as u8)
}

/// The inverse of [`armour`]: the bits of `payload`, the last `fill` bits removed.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a character outside the six-bit alphabet or a fill
/// above 5.
pub fn dearmour(payload: &str, fill: u8) -> Result<Vec<u8>> {
    ensure(fill <= 5, || format!("the fill bits are 0 to 5, got {fill}"))?;
    let mut bits = Vec::with_capacity(6 * payload.len());
    for c in payload.chars() {
        let code = u32::from(c);
        let v = match code {
            48..=87 => code - 48,
            96..=119 => code - 56,
            _ => return Err(comnum_core::Error::Config(format!("{c:?} is not a six-bit character"))),
        };
        bits.extend((0..6).rev().map(|i| ((v >> i) & 1) as u8));
    }
    ensure(bits.len() >= usize::from(fill), || "fewer bits than fill bits".into())?;
    bits.truncate(bits.len() - usize::from(fill));
    Ok(bits)
}

/// The checksum of the characters of `body` (between `!` and `*`).
pub fn checksum(body: &str) -> u8 {
    body.bytes().fold(0, |a, b| a ^ b)
}

/// The `!AIVDM` sentences of the message `bits` on the channel `channel` (`'A'` or
/// `'B'`), one or several fragments; `sequence` is the sequential message identifier
/// of a multi-fragment message (0 to 9).
pub fn aivdm(bits: &[u8], channel: char, sequence: u8) -> Vec<String> {
    let (payload, fill) = armour(bits);
    let chars: Vec<char> = payload.chars().collect();
    let chunks: Vec<String> = if chars.is_empty() {
        vec![String::new()]
    } else {
        chars.chunks(MAX_PAYLOAD_CHARS).map(|c| c.iter().collect()).collect()
    };
    let n = chunks.len();
    chunks
        .iter()
        .enumerate()
        .map(|(k, chunk)| {
            let seq = if n > 1 { (sequence % 10).to_string() } else { String::new() };
            let fill_k = if k + 1 == n { fill } else { 0 };
            let body = format!("AIVDM,{n},{},{seq},{channel},{chunk},{fill_k}", k + 1);
            format!("!{body}*{:02X}", checksum(&body))
        })
        .collect()
}

/// One parsed sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sentence {
    /// Number of fragments of the message.
    pub fragments: u8,
    /// This fragment, from 1.
    pub fragment: u8,
    /// The sequential message identifier (`None` for a single-fragment message).
    pub sequence: Option<u8>,
    /// The channel (`'A'`, `'B'`, or another character of the field; `' '` if empty).
    pub channel: char,
    /// The payload characters.
    pub payload: String,
    /// The fill bits of the payload.
    pub fill: u8,
}

/// Parse one `!AIVDM` (or `!AIVDO`) sentence and check its checksum.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a sentence that is not `!AIVDM` or `!AIVDO`, has
/// fewer than six fields, or whose checksum does not match.
pub fn parse(sentence: &str) -> Result<Sentence> {
    let s = sentence.trim();
    let body = s.strip_prefix('!').ok_or_else(|| comnum_core::Error::Config(format!("not a sentence: {s:?}")))?;
    let (body, cs) = body.split_once('*').ok_or_else(|| comnum_core::Error::Config("no checksum".into()))?;
    let expected =
        u8::from_str_radix(cs.trim(), 16).map_err(|_| comnum_core::Error::Config(format!("checksum {cs:?}")))?;
    ensure(checksum(body) == expected, || format!("checksum {:02X} for *{cs}", checksum(body)))?;
    let f: Vec<&str> = body.split(',').collect();
    ensure(f.len() >= 7 && (f[0] == "AIVDM" || f[0] == "AIVDO"), || format!("not an AIVDM sentence: {s:?}"))?;
    let num = |x: &str, what: &str| x.parse::<u8>().map_err(|_| comnum_core::Error::Config(format!("{what} {x:?}")));
    Ok(Sentence {
        fragments: num(f[1], "fragments")?,
        fragment: num(f[2], "fragment")?,
        sequence: if f[3].is_empty() { None } else { Some(num(f[3], "sequence")?) },
        channel: f[4].chars().next().unwrap_or(' '),
        payload: f[5].to_string(),
        fill: num(f[6], "fill")?,
    })
}

/// The messages of a list of sentences: the bits and the channel of every complete
/// message, the fragments of one sequence identifier assembled in order (a fragment
/// missing drops the message).
///
/// # Errors
/// The errors of [`parse`] and [`dearmour`].
pub fn messages(sentences: &[&str]) -> Result<Vec<(Vec<u8>, char)>> {
    let mut out = Vec::new();
    let mut pending: Vec<Sentence> = Vec::new();
    for s in sentences {
        let p = parse(s)?;
        if p.fragments <= 1 {
            out.push((dearmour(&p.payload, p.fill)?, p.channel));
            continue;
        }
        if p.fragment == 1 {
            pending.retain(|q| q.sequence != p.sequence);
        }
        let complete = pending.iter().filter(|q| q.sequence == p.sequence).count() + 1 == usize::from(p.fragments)
            && (1..p.fragment).all(|k| pending.iter().any(|q| q.sequence == p.sequence && q.fragment == k));
        if complete {
            let mut parts: Vec<Sentence> = pending.iter().filter(|q| q.sequence == p.sequence).cloned().collect();
            parts.push(p.clone());
            parts.sort_by_key(|q| q.fragment);
            let payload: String = parts.iter().map(|q| q.payload.as_str()).collect();
            out.push((dearmour(&payload, parts.last().map_or(0, |q| q.fill))?, p.channel));
            pending.retain(|q| q.sequence != p.sequence);
        } else {
            pending.push(p);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn armouring_round_trips_with_fill_bits() {
        for n in [6, 7, 11, 168, 424] {
            let bits: Vec<u8> = (0..n).map(|i| ((i * 7 + i / 3) % 3 == 0) as u8).collect();
            let (s, fill) = armour(&bits);
            assert_eq!(usize::from(fill), (6 - n % 6) % 6);
            assert_eq!(dearmour(&s, fill).unwrap(), bits);
        }
        assert!(dearmour("x", 0).is_err());
    }

    #[test]
    fn sentences_carry_their_checksum_and_fragments_reassemble() {
        let bits: Vec<u8> = (0..424).map(|i| (i % 5 == 1) as u8).collect();
        let s = aivdm(&bits, 'B', 3);
        assert_eq!(s.len(), 2);
        assert!(s[0].starts_with("!AIVDM,2,1,3,B,"));
        for x in &s {
            parse(x).unwrap();
        }
        let m = messages(&[s[0].as_str(), s[1].as_str()]).unwrap();
        assert_eq!(m, vec![(bits.clone(), 'B')]);
        // the second fragment alone is incomplete; a wrong checksum is an error
        assert_eq!(messages(&[s[1].as_str()]).unwrap(), vec![]);
        let bad = format!("{}0", &s[0][..s[0].len() - 1]);
        assert!(parse(&bad).is_err());
        let short = aivdm(&bits[..168], 'A', 0);
        assert_eq!(short.len(), 1);
        assert!(short[0].starts_with("!AIVDM,1,1,,A,"));
    }
}
