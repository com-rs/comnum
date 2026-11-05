//! The radio data system (RDS, RBDS in the United States): 1187.5 bit/s on a 57 kHz
//! subcarrier of the FM multiplex, in groups of four blocks protected by a shortened
//! cyclic code (NRSC-4, the U.S. RBDS standard of April 1998, which reproduces the
//! European specification EN 50067 and adds the U.S. annexes).
//!
//! # Physical layer (Sec. 1)
//!
//! The subcarrier, at three times the 19 kHz pilot (57 kHz, Sec. 1.1), is suppressed and
//! amplitude-modulated by the data, a two-phase PSK of $\pm 90$° (Sec. 1.4), at $57/48$
//! kHz $= 1187.5$ bit/s (Sec. 1.5). The source bits are differentially encoded, $d_{k} =
//! d_{k-1} \oplus b_{k}$ (Table 1), so that the receiver decodes them whatever the sign of
//! its carrier ($b_{k} = d_{k} \oplus d_{k-1}$, Table 2), and every bit is sent as a
//! biphase symbol: an odd pair of impulses $\pm[\delta(t) - \delta(t - t_{d}/2)]$, $t_{d} =
//! 1/1187.5$ s, the sign set by the bit (eqs. (1)–(2)), shaped by
//!
//! $$
//! H_{T}(f) = \cos\frac{\pi f t_{d}}{4}, \quad 0 \le f \le \frac{2}{t_{d}}, \qquad 0 \text{ above}
//! $$
//!
//! (eq. (3)), whose impulse response is $h(t) \propto \cos(4\pi t/t_{d}) / (1 - 64 t^{2} /
//! t_{d}^{2})$ ([`shaping_pulse`], truncated to $\pm 4 t_{d}$ under a Dolph–Chebyshev
//! window). The receiver applies the same filter (the shaping is split between the two
//! ends, Sec. 1.7), so that the whole channel has a cosine roll-off of 100 %.
//!
//! [`demodulate`] mixes the multiplex down from 57 kHz, filters it by $h$, combines the
//! two halves of every symbol, $z(t) = y(t) - y(t - t_{d}/2)$, at two samples per bit,
//! recovers the clock by the Gardner loop ([`comnum_dsp::sync::timing::GardnerLoop`])
//! and the phase of the BPSK by the Viterbi–Viterbi estimator of order 2
//! ([`comnum_dsp::sync::carrier::viterbi_viterbi_for`]): no pilot is needed (a mono station
//! has none), the subcarrier being within a few hertz of 57 kHz.
//!
//! # Data-link layer (Sec. 2)
//!
//! A group has 104 bits, four blocks of a 16-bit information word and a 10-bit
//! checkword, most significant bit first (Sec. 2.1–2.2). The checkword is the remainder
//! of $m(x) x^{10}$ by $g(x) = x^{10} + x^{8} + x^{7} + x^{5} + x^{4} + x^{3} + 1$
//! ([`checkword`], [`comnum_fec::crc::Crc`]) plus the offset word of the block (Sec.
//! 2.3, Table A.1): A = 0011111100, B = 0110011000, C = 0101101000, C′ = 1101010000, D =
//! 0110110100, C′ in the third block of the groups of version B. The offsets identify
//! the blocks and synchronise the receiver (Sec. 2.4): [`groups`] looks for two blocks
//! whose offsets follow each other, then reads the groups at that phase.
//!
//! # Messages (Sec. 3)
//!
//! The first block of every group is the programme identification (PI); the second
//! holds the group type (4 bits), its version B$_0$, the traffic programme flag (TP) and
//! the programme type (PTY, 5 bits) (Figure 9). [`Station::from_groups`] assembles:
//!
//! - the programme service name (PS), 8 characters, two per group of type 0A or 0B at the
//!   address $C_{1} C_{0}$ of block 2 (Sec. 3.1.5.1, Figures 12–13);
//! - the radiotext (RT), 64 characters, four per group 2A (blocks 3 and 4) or 32, two per
//!   group 2B (block 4), at the address $C_{3} \dots C_{0}$; a change of the text A/B
//!   flag clears it (Sec. 3.1.5.3, Figures 16–17);
//! - the clock time (CT) of the groups 4A: the modified Julian day (17 bits), the UTC
//!   hour and minute, the local offset in half hours (Sec. 3.1.5.6, Figure 20), the date
//!   from the formulas of Annex G.
//!
//! [`receive`] chains the demodulator, the differential decoding, the synchronisation and
//! the assembly.
//!
//! The characters follow the code table of Figure E.1 (Annex E, the EBU Latin-based
//! repertoire; [`character`]).
//!
//! # Examples
//! ```
//! use comnum_systems::fm::rds::{block_offset, checkword, encode_block, Offset};
//! let block = encode_block(0xF201, Offset::A);
//! assert_eq!(block >> 10, 0xF201);
//! assert_eq!(block_offset(block), Some(Offset::A));
//! assert_eq!(block_offset(block ^ 0x40), None);       // an error in the checkword
//! assert_eq!(checkword(0) , 0);
//! ```
//!
//! # References
//!
//! 1. NRSC-4, *United States RBDS Standard: Specification of the radio broadcast data
//!    system (RBDS)*, National Radio Systems Committee, April 1998, Sec. 1–3, Annexes A,
//!    E, G (the European specification is CENELEC EN 50067, now IEC 62106).

use comnum_core::{Result, ensure, ensure_shape};
use comnum_dsp::filter::fir_filter;
use comnum_dsp::sync::carrier::{derotate, viterbi_viterbi_for};
use comnum_dsp::sync::timing::GardnerLoop;
use comnum_dsp::window::chebyshev;
use comnum_fec::crc::Crc;
use comnum_math::C64;
use std::f64::consts::PI;

/// The frequency of the subcarrier, three times the pilot (Hz).
pub const SUBCARRIER_HZ: f64 = 57e3;
/// The bit rate, $57\\,000/48$ (bit/s).
pub const BIT_RATE: f64 = 1187.5;
/// Bits of a block.
pub const BLOCK_BITS: usize = 26;
/// Bits of a group.
pub const GROUP_BITS: usize = 4 * BLOCK_BITS;

/// The offset words of Table A.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Offset {
    /// Block 1.
    A,
    /// Block 2.
    B,
    /// Block 3 of a group of version A.
    C,
    /// Block 3 of a group of version B.
    CPrime,
    /// Block 4.
    D,
}

impl Offset {
    /// The 10 bits $d_{9} \dots d_{0}$ of the offset word (Table A.1).
    pub fn word(self) -> u16 {
        match self {
            Offset::A => 0b00_1111_1100,
            Offset::B => 0b01_1001_1000,
            Offset::C => 0b01_0110_1000,
            Offset::CPrime => 0b11_0101_0000,
            Offset::D => 0b01_1011_0100,
        }
    }

    /// The position of the block in its group (0 to 3).
    pub fn position(self) -> usize {
        match self {
            Offset::A => 0,
            Offset::B => 1,
            Offset::C | Offset::CPrime => 2,
            Offset::D => 3,
        }
    }

    const ALL: [Offset; 5] = [Offset::A, Offset::B, Offset::C, Offset::CPrime, Offset::D];
}

/// The checkword of an information word without its offset: the remainder of
/// $m(x) x^{10}$ by $g(x)$ (Sec. 2.3).
pub fn checkword(info: u16) -> u16 {
    let crc = Crc::new("RDS", 10, 0x1B9).unwrap_or_else(|_| unreachable!("a 10-bit CRC is valid"));
    let bits: Vec<u8> = (0..16).rev().map(|i| ((info >> i) & 1) as u8).collect();
    crc.checksum(&bits).iter().fold(0, |v, &b| 2 * v + u16::from(b))
}

/// The 26 bits of a block, information word then checkword plus offset, as an integer
/// whose bit 25 is sent first.
pub fn encode_block(info: u16, offset: Offset) -> u32 {
    (u32::from(info) << 10) | u32::from(checkword(info) ^ offset.word())
}

/// The offset whose word the checkword of the 26-bit `block` carries, `None` when the
/// block has errors or another offset (the syndrome of Annex B).
pub fn block_offset(block: u32) -> Option<Offset> {
    let s = checkword((block >> 10) as u16) ^ (block & 0x3FF) as u16;
    Offset::ALL.into_iter().find(|o| o.word() == s)
}

/// The 104 bits of the group of the four information words `blocks`, version B when
/// `version_b` (the offset C′ in block 3).
pub fn group_bits(blocks: [u16; 4], version_b: bool) -> Vec<u8> {
    let offsets = [Offset::A, Offset::B, if version_b { Offset::CPrime } else { Offset::C }, Offset::D];
    blocks
        .iter()
        .zip(offsets)
        .flat_map(|(&b, o)| {
            let w = encode_block(b, o);
            (0..BLOCK_BITS).rev().map(move |i| ((w >> i) & 1) as u8)
        })
        .collect()
}

/// The differential encoding $d_{k} = d_{k-1} \oplus b_{k}$, $d_{-1} = 0$ (Table 1).
pub fn differential_encode(bits: &[u8]) -> Vec<u8> {
    let mut d = 0;
    bits.iter()
        .map(|b| {
            d ^= b & 1;
            d
        })
        .collect()
}

/// The differential decoding $b_{k} = d_{k} \oplus d_{k-1}$, $d_{-1} = 0$ (Table 2): the
/// same bits for the inverted sequence but the first.
pub fn differential_decode(bits: &[u8]) -> Vec<u8> {
    let mut prev = 0;
    bits.iter()
        .map(|d| {
            let b = (d & 1) ^ prev;
            prev = d & 1;
            b
        })
        .collect()
}

/// Samples of a bit at `fs`, which must be a multiple of $2 \cdot 1187.5$ Hz (so that the
/// two halves of a biphase symbol start on samples).
fn samples_per_bit(fs: f64) -> Result<usize> {
    let half = fs / (2.0 * BIT_RATE);
    ensure(fs.is_finite() && half >= 2.0 && (half - half.round()).abs() < 1e-9, || {
        format!("the RDS needs a rate that is a multiple of 2375 Hz, at least 4750 Hz, got {fs}")
    })?;
    Ok(2 * half.round() as usize)
}

/// The data-shaping filter $h(t) \propto \cos(4\pi t/t_{d})/(1 - 64 t^{2}/t_{d}^{2})$ of
/// $H_{T}(f)$ (eq. (3)) sampled at `fs` over $\pm 4 t_{d}$, under a Dolph–Chebyshev
/// window of 60 dB, peak 1.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a rate that is not a multiple of 2375 Hz.
pub fn shaping_pulse(fs: f64) -> Result<Vec<f64>> {
    let n = samples_per_bit(fs)?;
    let half = 4 * n;
    let w = chebyshev(2 * half + 1, 60.0);
    Ok((0..=2 * half)
        .map(|i| {
            let u = 8.0 * (i as f64 - half as f64) / n as f64; // 8 t / t_d
            let h = if (u.abs() - 1.0).abs() < 1e-9 { PI / 4.0 } else { (PI * u / 2.0).cos() / (1.0 - u * u) };
            h * w[i]
        })
        .collect())
}

/// The shaped biphase baseband of the transmitted bits `d` (already differentially
/// encoded) at `fs`: the impulse pair $(2d_{k} - 1)[\delta(t - k t_{d}) - \delta(t - k t_{d} -
/// t_{d}/2)]$ of every bit through $h$, `d.len()` bits of samples, peak about 1.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a rate that is not a multiple of 2375 Hz.
pub fn biphase_baseband(d: &[u8], fs: f64) -> Result<Vec<f64>> {
    let n = samples_per_bit(fs)?;
    let mut impulses = vec![C64::new(0.0, 0.0); d.len() * n];
    for (k, &b) in d.iter().enumerate() {
        let a = if b & 1 == 1 { 1.0 } else { -1.0 };
        impulses[k * n] = C64::new(a, 0.0);
        impulses[k * n + n / 2] = C64::new(-a, 0.0);
    }
    Ok(fir_filter(&shaping_pulse(fs)?, &impulses).iter().map(|v| v.re).collect())
}

/// The soft decisions on the transmitted bits $d_{k}$ of the RDS of the multiplex `mpx`
/// (at `fs`, a multiple of 2375 Hz above 120 kHz), one per bit, their common sign
/// unknown (see the module): differential decoding removes it.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a rate that is not a multiple of 2375 Hz or below
/// 120 kHz; [`comnum_core::Error::Shape`] for less than 0.1 s of multiplex.
pub fn demodulate(mpx: &[f64], fs: f64) -> Result<Vec<f64>> {
    let n = samples_per_bit(fs)?;
    ensure(fs >= 120e3, || format!("the multiplex must be sampled at 120 kHz at least, got {fs}"))?;
    ensure_shape(mpx.len() as f64 >= 0.1 * fs, || format!("{} samples is less than 0.1 s at {fs} Hz", mpx.len()))?;
    let w = 2.0 * PI * SUBCARRIER_HZ / fs;
    let mixed: Vec<C64> = mpx.iter().enumerate().map(|(i, v)| C64::from_polar(*v, -w * i as f64)).collect();
    let y = fir_filter(&shaping_pulse(fs)?, &mixed);
    // the two halves of every symbol, at two samples per bit
    let h = n / 2;
    let z: Vec<C64> = (h..y.len()).step_by(h).map(|i| y[i] - y[i - h]).collect();
    // unit power: the gains of the clock loop assume it
    let power = z.iter().map(C64::norm_sqr).sum::<f64>() / z.len().max(1) as f64;
    let z: Vec<C64> = z.iter().map(|v| v / power.sqrt().max(1e-300)).collect();
    let clock = GardnerLoop { nyquist: false, ..GardnerLoop::default() }.run(&z);
    let s = &clock.signal;
    // the phase of the samples that are the symbols: the larger energy
    let energy = |p: usize| s.iter().skip(p).step_by(2).map(C64::norm_sqr).sum::<f64>();
    let phase = usize::from(energy(1) > energy(0));
    let symbols: Vec<C64> = s.iter().skip(phase).step_by(2).copied().collect();
    if symbols.is_empty() {
        return Ok(Vec::new());
    }
    let bpsk = [C64::new(1.0, 0.0), C64::new(-1.0, 0.0)];
    let theta = viterbi_viterbi_for(&symbols, 16, 2, &bpsk)?;
    Ok(derotate(&symbols, &theta, 1.0)?.iter().map(|v| v.re).collect())
}

/// The groups and the station of the RDS of the multiplex `mpx` (at `fs`): [`demodulate`],
/// hard decisions, [`differential_decode`], [`groups`], [`Station::from_groups`].
///
/// # Errors
/// As [`demodulate`].
pub fn receive(mpx: &[f64], fs: f64) -> Result<(Vec<Group>, Station)> {
    let soft = demodulate(mpx, fs)?;
    let hard: Vec<u8> = soft.iter().map(|v| u8::from(*v > 0.0)).collect();
    let g = groups(&differential_decode(&hard));
    let station = Station::from_groups(&g);
    Ok((g, station))
}

/// A received group: its four information words, `None` for a block whose checkword
/// failed, and the bit where it starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    /// The information words of blocks 1 to 4.
    pub blocks: [Option<u16>; 4],
    /// Version B (the offset C′ in block 3).
    pub version_b: bool,
    /// The first bit of the group in the decoded stream.
    pub start: usize,
}

/// The 26 bits from `i` as an integer.
fn word_at(bits: &[u8], i: usize) -> u32 {
    bits[i..i + BLOCK_BITS].iter().fold(0, |v, &b| 2 * v + u32::from(b & 1))
}

/// The groups of the source bits `bits` (after differential decoding): the receiver
/// synchronises on two consecutive blocks whose offsets follow each other (Sec. 2.4),
/// reads the groups at that phase, and searches again after five groups without a
/// valid block.
pub fn groups(bits: &[u8]) -> Vec<Group> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 2 * BLOCK_BITS <= bits.len() {
        // look for the start of a group from i
        let Some(start) = (i..=bits.len() - 2 * BLOCK_BITS).find_map(|j| {
            let a = block_offset(word_at(bits, j))?;
            let b = block_offset(word_at(bits, j + BLOCK_BITS))?;
            (b.position() == (a.position() + 1) % 4).then(|| j.checked_sub(a.position() * BLOCK_BITS)).flatten()
        }) else {
            break;
        };
        let mut g = start;
        let mut misses = 0;
        while g + GROUP_BITS <= bits.len() && misses < 5 {
            let mut blocks = [None; 4];
            let mut version_b = false;
            for (k, block) in blocks.iter_mut().enumerate() {
                let w = word_at(bits, g + k * BLOCK_BITS);
                // (two `if`s rather than a `let` chain: the MSRV is 1.85)
                if let Some(o) = block_offset(w) {
                    if o.position() == k {
                        *block = Some((w >> 10) as u16);
                        version_b |= o == Offset::CPrime;
                    }
                }
            }
            // the version is also in block 2
            if let Some(b2) = blocks[1] {
                version_b = (b2 >> 11) & 1 == 1;
            }
            misses = if blocks.iter().all(Option::is_none) { misses + 1 } else { 0 };
            out.push(Group { blocks, version_b, start: g });
            g += GROUP_BITS;
        }
        // drop the groups of the misses, search after the last valid one
        while out.last().is_some_and(|gr| gr.blocks.iter().all(Option::is_none)) {
            out.pop();
        }
        i = out.last().map_or(g, |gr| gr.start + GROUP_BITS).max(i + 1);
    }
    out
}

/// The character of a code of Figure E.1 (the complete EBU Latin-based repertoire),
/// `None` for the control codes 0/0 to 1/15 and 7/15.
pub fn character(code: u8) -> Option<char> {
    const UPPER: [&str; 8] = [
        "áàéèíìóòúùÑÇŞβ¡Ĳ",
        "âäêëîïôöûüñçşğıĳ",
        "ªα©‰Ğěňőπ₠£$←↑→↓",
        "º¹²³±İńűµ¿÷°¼½¾§",
        "ÁÀÉÈÍÌÓÒÚÙŘČŠŽÐĿ",
        "ÂÄÊËÎÏÔÖÛÜřčšžđŀ",
        "ÃÅÆŒŶÝÕØÞŊŔĆŚŹŦð",
        "ãåæœŵýõøþŋŕćśźŧ",
    ];
    match code {
        0x20..=0x7E => Some(match code {
            0x24 => '¤',
            0x5E => '―',
            0x60 => '‖',
            0x7E => '¯',
            c => char::from(c),
        }),
        0x80..=0xFE => UPPER[usize::from((code - 0x80) / 16)].chars().nth(usize::from(code % 16)),
        _ => None,
    }
}

/// A date and a time of the clock-time groups (UTC, and the local offset).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockTime {
    /// The modified Julian day.
    pub mjd: u32,
    /// The year (e.g. 2026), from Annex G.
    pub year: u32,
    /// The month, 1 to 12.
    pub month: u32,
    /// The day of the month, 1 to 31.
    pub day: u32,
    /// The UTC hour, 0 to 23.
    pub hour: u32,
    /// The UTC minute, 0 to 59.
    pub minute: u32,
    /// The local offset in half hours, $-24$ to 24.
    pub offset_half_hours: i32,
}

impl ClockTime {
    /// The clock time of blocks 2 to 4 of a group 4A (Figure 20; the date by the
    /// formulas a) of Annex G).
    pub fn from_blocks(b2: u16, b3: u16, b4: u16) -> ClockTime {
        let mjd = (u32::from(b2 & 3) << 15) | u32::from(b3 >> 1);
        let m = f64::from(mjd);
        let yp = ((m - 15_078.2) / 365.25).floor();
        let mp = ((m - 14_956.1 - (yp * 365.25).floor()) / 30.6001).floor();
        let day = m - 14_956.0 - (yp * 365.25).floor() - (mp * 30.6001).floor();
        let k = if mp == 14.0 || mp == 15.0 { 1.0 } else { 0.0 };
        let offset = i32::from(b4 & 0x1F);
        ClockTime {
            mjd,
            year: (1900.0 + yp + k) as u32,
            month: (mp - 1.0 - 12.0 * k) as u32,
            day: day as u32,
            hour: (u32::from(b3 & 1) << 4) | u32::from(b4 >> 12),
            minute: u32::from((b4 >> 6) & 0x3F),
            offset_half_hours: if (b4 >> 5) & 1 == 1 { -offset } else { offset },
        }
    }
}

/// What the groups of a station tell (see the module).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Station {
    /// The programme identification, the most frequent PI of the groups.
    pub pi: Option<u16>,
    /// The programme type (0 to 31) of the last group.
    pub pty: Option<u8>,
    /// The traffic programme flag of the last group.
    pub tp: Option<bool>,
    /// The programme service name, 8 characters (a space where none was received).
    pub ps: String,
    /// The radiotext up to its carriage return (spaces where none was received).
    pub radiotext: String,
    /// The last clock time.
    pub clock: Option<ClockTime>,
    /// The number of groups of each type, `[type][version]` (0 for A).
    pub group_types: [[usize; 2]; 16],
    /// The groups whose four blocks passed their checkword.
    pub complete_groups: usize,
    /// The blocks that passed their checkword, and all the blocks read.
    pub valid_blocks: (usize, usize),
}

impl Station {
    /// The station of the groups `groups` (see the module).
    pub fn from_groups(groups: &[Group]) -> Station {
        let mut st = Station::default();
        let mut ps = [None::<char>; 8];
        let mut rt = [None::<char>; 64];
        let mut rt_flag = None;
        let mut pi_count: Vec<(u16, usize)> = Vec::new();
        let ch = |c: u16| character(c as u8);
        for g in groups {
            let valid = g.blocks.iter().filter(|b| b.is_some()).count();
            st.valid_blocks.0 += valid;
            st.valid_blocks.1 += 4;
            st.complete_groups += usize::from(valid == 4);
            if let Some(pi) = g.blocks[0] {
                match pi_count.iter_mut().find(|(p, _)| *p == pi) {
                    Some(e) => e.1 += 1,
                    None => pi_count.push((pi, 1)),
                }
            }
            let Some(b2) = g.blocks[1] else { continue };
            let (kind, version_b) = (usize::from(b2 >> 12), (b2 >> 11) & 1 == 1);
            st.group_types[kind][usize::from(version_b)] += 1;
            st.tp = Some((b2 >> 10) & 1 == 1);
            st.pty = Some(((b2 >> 5) & 0x1F) as u8);
            match (kind, version_b, g.blocks[2], g.blocks[3]) {
                (0, _, _, Some(b4)) => {
                    let seg = usize::from(b2 & 3);
                    ps[2 * seg] = ch(b4 >> 8);
                    ps[2 * seg + 1] = ch(b4 & 0xFF);
                }
                (2, _, b3, Some(b4)) => {
                    let flag = (b2 >> 4) & 1;
                    if rt_flag.is_some_and(|f| f != flag) {
                        rt = [None; 64];
                    }
                    rt_flag = Some(flag);
                    let seg = usize::from(b2 & 0xF);
                    if version_b {
                        rt[2 * seg] = ch(b4 >> 8);
                        rt[2 * seg + 1] = ch(b4 & 0xFF);
                    } else if let Some(b3) = b3 {
                        for (i, c) in [b3 >> 8, b3 & 0xFF, b4 >> 8, b4 & 0xFF].into_iter().enumerate() {
                            rt[4 * seg + i] = if c == 0x0D { Some('\r') } else { ch(c) };
                        }
                    }
                }
                (4, false, Some(b3), Some(b4)) => st.clock = Some(ClockTime::from_blocks(b2, b3, b4)),
                _ => {}
            }
        }
        st.pi = pi_count.iter().max_by_key(|(_, n)| *n).map(|(p, _)| *p);
        st.ps = ps.iter().map(|c| c.unwrap_or(' ')).collect();
        let text: String = rt.iter().map(|c| c.unwrap_or(' ')).collect();
        st.radiotext = text.split('\r').next().unwrap_or("").trim_end().to_string();
        st
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_checkword_detects_every_single_and_double_error_of_a_block() {
        let block = encode_block(0x1234, Offset::B);
        assert_eq!(block_offset(block), Some(Offset::B));
        for i in 0..26 {
            assert_eq!(block_offset(block ^ (1 << i)), None, "bit {i}");
            for j in 0..i {
                let e = block ^ (1 << i) ^ (1 << j);
                assert!(block_offset(e).is_none_or(|o| o != Offset::B), "bits {i} {j}");
            }
        }
        // g(x) divides the codewords without offset
        assert_eq!(checkword(0x8000) ^ checkword(0x4000) ^ checkword(0xC000), 0);
    }

    #[test]
    fn the_offsets_are_table_a1() {
        let w: Vec<u16> = Offset::ALL.iter().map(|o| o.word()).collect();
        assert_eq!(w, [0x0FC, 0x198, 0x168, 0x350, 0x1B4]);
        // d1 = d0 = 0 (annex A)
        assert!(w.iter().all(|v| v.trailing_zeros() >= 2));
    }

    #[test]
    fn differential_coding_ignores_an_inversion() {
        let b = [1u8, 0, 0, 1, 1, 1, 0, 1, 0];
        let d = differential_encode(&b);
        assert_eq!(differential_decode(&d), b);
        let inverted: Vec<u8> = d.iter().map(|x| 1 - x).collect();
        assert_eq!(differential_decode(&inverted)[1..], b[1..]);
    }

    #[test]
    fn the_groups_and_the_station_are_read_back_from_the_bits() {
        let pi = 0xF20C;
        let ps = b"COMNUM  ";
        let mut bits = vec![0u8, 1, 1, 0, 1]; // an arbitrary phase
        for seg in 0..4u16 {
            let b2 = (1 << 10) | (10 << 5) | seg; // group 0A, TP, PTY 10
            let b4 = (u16::from(ps[2 * seg as usize]) << 8) | u16::from(ps[2 * seg as usize + 1]);
            bits.extend(group_bits([pi, b2, 0xE0CD, b4], false));
        }
        // a 4A group: MJD 61347 (3 November 2026), 21:07 UTC, offset +1 h
        let mjd: u16 = (0xEFA3_u32 & 0x7FFF) as u16;
        let b2 = (4 << 12) | (61_347u32 >> 15) as u16;
        let b3 = (mjd << 1) | 1; // hour bit 4 = 1 (21 = 10101)
        let b4 = (0b0101 << 12) | (7 << 6) | 2;
        bits.extend(group_bits([pi, b2, b3, b4], false));
        // a 2B group: "Hi" at address 0
        bits.extend(group_bits([pi, (2 << 12) | (1 << 11), pi, u16::from_be_bytes(*b"Hi")], true));
        let gs = groups(&bits);
        assert_eq!(gs.len(), 6);
        assert_eq!(gs[0].start, 5);
        assert!(gs.iter().all(|g| g.blocks.iter().all(Option::is_some)));
        let st = Station::from_groups(&gs);
        assert_eq!((st.pi, st.ps.as_str(), st.pty, st.tp), (Some(pi), "COMNUM  ", Some(0), Some(false)));
        let ct = st.clock.unwrap();
        assert_eq!((ct.year, ct.month, ct.day, ct.hour, ct.minute, ct.offset_half_hours), (2026, 11, 3, 21, 7, 2));
        assert!(st.radiotext.starts_with("Hi"));
        assert_eq!((st.group_types[0][0], st.group_types[4][0], st.group_types[2][1]), (4, 1, 1));
        // a corrupted block is lost alone
        let mut bad = bits.clone();
        bad[5 + 30] ^= 1;
        let gs = groups(&bad);
        assert_eq!(gs[0].blocks[1], None);
        assert_eq!(Station::from_groups(&gs).valid_blocks, (23, 24));
    }

    #[test]
    fn the_annex_g_example_and_the_characters() {
        // Annex G b): MJD of 3 November 2026 (Y = 126, M = 11, D = 3, L = 0)
        let mjd = 14_956 + 3 + (126.0f64 * 365.25).floor() as u32 + (12.0f64 * 30.6001).floor() as u32;
        assert_eq!(mjd, 61_347);
        assert_eq!(character(b'A'), Some('A'));
        assert_eq!(character(0x82), Some('é'));
        assert_eq!(character(0x9B), Some('ç'));
        assert_eq!(character(0x24), Some('¤'));
        assert_eq!((character(0x0D), character(0xFF)), (None, None));
        assert!((0x80..=0xFE).all(|c| character(c).is_some()));
    }

    #[test]
    fn the_biphase_symbols_are_odd_and_band_limited() {
        let fs = 228e3;
        let h = shaping_pulse(fs).unwrap();
        assert_eq!(h.len(), 8 * 192 + 1);
        let peak = h.iter().copied().fold(0.0, f64::max);
        assert!((h[4 * 192] - peak).abs() < 1e-12);
        // its spectrum vanishes above 2/t_d = 2375 Hz (the truncation leaves -40 dB)
        let resp = |f: f64| -> f64 {
            h.iter().enumerate().map(|(i, v)| v * (2.0 * PI * f * i as f64 / fs).cos()).sum::<f64>().abs()
        };
        assert!(20.0 * (resp(4e3) / resp(0.0)).log10() < -40.0, "{}", 20.0 * (resp(4e3) / resp(0.0)).log10());
        assert!(shaping_pulse(240e3).is_err());
        let x = biphase_baseband(&[1, 0, 1], fs).unwrap();
        assert_eq!(x.len(), 3 * 192);
    }
}
