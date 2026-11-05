//! The messages of AIS (ITU-R M.1371-5, Annex 8): the fields of the position reports
//! (messages 1, 2, 3 and 18), the base station report (4), the static and voyage data
//! (5) and the class B static data (24), read from the bit string of a message.
//!
//! A message is a string of bits, most significant first; a field of $n$ bits at
//! position $p$ is the unsigned integer of those bits, a signed field its two's
//! complement, and a text field a sequence of six-bit characters (Table 46: codes 0 to
//! 31 are `@` to `_`, 32 to 63 are space to `?`; `@` ends a text). The special values
//! of Annex 8 ("not available": longitude 181°, latitude 91°, speed 1023, course 3600,
//! heading 511) are `None`.
//!
//! # References
//!
//! 1. Recommendation ITU-R M.1371-5, *Technical characteristics for an automatic
//!    identification system using time division multiple access in the VHF maritime
//!    mobile band*, 2014, Annex 8: Sec. 3.3.8.2.1 (messages 1, 2, 3, Table 47),
//!    Sec. 3.3.8.2.2 (message 4, Table 48), Sec. 3.3.8.2.3 (message 5, Table 49),
//!    Sec. 3.3.8.2.17 (message 18, Table 68), Sec. 3.3.8.2.22 (message 24, Tables 76–77),
//!    Table 46 (the six-bit ASCII).

use comnum_core::{Result, ensure};

/// The bits of a message, with its field accessors.
#[derive(Clone, Debug)]
pub struct Fields<'a>(pub &'a [u8]);

impl Fields<'_> {
    /// The unsigned integer of `len` bits from `start` (most significant first); the bits
    /// beyond the message are 0.
    pub fn uint(&self, start: usize, len: usize) -> u64 {
        (start..start + len).fold(0u64, |a, i| (a << 1) | u64::from(self.0.get(i).copied().unwrap_or(0) & 1))
    }

    /// The two's complement integer of `len` bits from `start`.
    pub fn int(&self, start: usize, len: usize) -> i64 {
        let v = self.uint(start, len);
        if len < 64 && v >> (len - 1) & 1 == 1 { v as i64 - (1i64 << len) } else { v as i64 }
    }

    /// The text of `len` bits from `start` (six-bit characters, Table 46), cut at the
    /// first `@` and trimmed of trailing spaces.
    pub fn text(&self, start: usize, len: usize) -> String {
        let mut s = String::new();
        for k in 0..len / 6 {
            let v = self.uint(start + 6 * k, 6) as u8;
            if v == 0 {
                break;
            }
            s.push(char::from(if v < 32 { v + 64 } else { v }));
        }
        s.trim_end().to_string()
    }

    /// The number of bits.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there is no bit.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A position, speed and course with the "not available" values of Annex 8 as `None`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    /// Longitude (degrees, east positive), `None` for 181°.
    pub lon_deg: Option<f64>,
    /// Latitude (degrees, north positive), `None` for 91°.
    pub lat_deg: Option<f64>,
    /// Position accuracy: `true` for a high accuracy (below 10 m).
    pub accurate: bool,
    /// Speed over ground (knots), `None` for 1023.
    pub sog_knots: Option<f64>,
    /// Course over ground (degrees), `None` for 3600.
    pub cog_deg: Option<f64>,
    /// True heading (degrees), `None` for 511.
    pub heading_deg: Option<u16>,
    /// The UTC second of the position (0 to 59; 60 and more: not available or a
    /// positioning device in a special state).
    pub timestamp: u8,
}

impl Position {
    /// The fields of a position report: the accuracy bit at `accuracy`, then `lon` (28
    /// bits), `lat` (27), `cog` (12), `heading` (9), `timestamp` (6) at the given starts,
    /// and the speed (10 bits) at `sog`.
    fn read(
        f: &Fields<'_>,
        sog: usize,
        accuracy: usize,
        lon: usize,
        lat: usize,
        cog: usize,
        heading: usize,
        timestamp: usize,
    ) -> Position {
        let lon_raw = f.int(lon, 28);
        let lat_raw = f.int(lat, 27);
        let sog_raw = f.uint(sog, 10);
        let cog_raw = f.uint(cog, 12);
        let hdg = f.uint(heading, 9) as u16;
        Position {
            lon_deg: (lon_raw != 181 * 600_000).then(|| lon_raw as f64 / 600_000.0),
            lat_deg: (lat_raw != 91 * 600_000).then(|| lat_raw as f64 / 600_000.0),
            accurate: f.uint(accuracy, 1) == 1,
            sog_knots: (sog_raw != 1023).then(|| sog_raw as f64 / 10.0),
            cog_deg: (cog_raw != 3600).then(|| cog_raw as f64 / 10.0),
            heading_deg: (hdg != 511).then_some(hdg),
            timestamp: f.uint(timestamp, 6) as u8,
        }
    }
}

/// The dimensions of a ship from its reference point (metres), Annex 8, Fig. 42.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dimensions {
    /// Distance to the bow.
    pub to_bow: u16,
    /// Distance to the stern.
    pub to_stern: u16,
    /// Distance to the port side.
    pub to_port: u8,
    /// Distance to the starboard side.
    pub to_starboard: u8,
}

impl Dimensions {
    fn read(f: &Fields<'_>, start: usize) -> Dimensions {
        Dimensions {
            to_bow: f.uint(start, 9) as u16,
            to_stern: f.uint(start + 9, 9) as u16,
            to_port: f.uint(start + 18, 6) as u8,
            to_starboard: f.uint(start + 24, 6) as u8,
        }
    }
}

/// A decoded message.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    /// Messages 1, 2 and 3: the position report of a class A station (Table 47).
    PositionReport {
        /// 1 (scheduled), 2 (assigned) or 3 (in response to an interrogation).
        msg_type: u8,
        /// Repeat indicator.
        repeat: u8,
        /// The maritime mobile service identity.
        mmsi: u32,
        /// Navigational status (0 under way using engine, 1 at anchor, 5 moored, …;
        /// 15 not defined).
        nav_status: u8,
        /// Rate of turn, the raw field (−128 not available, ±127 turning faster than
        /// 5°/30 s; otherwise $4.733\sqrt{\mathrm{ROT}}$ in degrees per minute).
        rot_raw: i8,
        /// The position, speed and course.
        position: Position,
        /// Special manoeuvre indicator (0 not available).
        manoeuvre: u8,
        /// RAIM flag.
        raim: bool,
    },
    /// Message 4: the base station report (Table 48).
    BaseStation {
        /// Repeat indicator.
        repeat: u8,
        /// MMSI of the base station.
        mmsi: u32,
        /// UTC year (0 not available), month, day, hour, minute, second.
        utc: (u16, u8, u8, u8, u8, u8),
        /// The position (its speed, course and heading fields do not exist: `None`).
        position: Position,
        /// Type of electronic position fixing device (1 GPS, 2 GLONASS, 3 both, …).
        epfd: u8,
        /// RAIM flag.
        raim: bool,
    },
    /// Message 5: the static and voyage related data of a class A station (Table 49).
    StaticAndVoyage {
        /// Repeat indicator.
        repeat: u8,
        /// MMSI.
        mmsi: u32,
        /// AIS version indicator.
        ais_version: u8,
        /// IMO number (0 not available).
        imo: u32,
        /// Call sign.
        callsign: String,
        /// Name.
        shipname: String,
        /// Type of ship and cargo (Table 53).
        ship_type: u8,
        /// Dimensions.
        dimensions: Dimensions,
        /// Type of electronic position fixing device.
        epfd: u8,
        /// Estimated time of arrival, UTC: month (0 not available), day, hour (24 not
        /// available), minute (60 not available).
        eta: (u8, u8, u8, u8),
        /// Maximum present static draught (metres, 0 not available).
        draught_m: f64,
        /// Destination.
        destination: String,
        /// Data terminal equipment ready (`false`: available).
        dte: bool,
    },
    /// Message 18: the position report of a class B station (Table 68).
    ClassBPosition {
        /// Repeat indicator.
        repeat: u8,
        /// MMSI.
        mmsi: u32,
        /// The position, speed and course.
        position: Position,
        /// Carrier-sense unit (`true`) or self-organising (`false`) class B.
        cs_unit: bool,
        /// RAIM flag.
        raim: bool,
    },
    /// Message 24, part A: the name of a class B station (Table 76).
    ClassBName {
        /// Repeat indicator.
        repeat: u8,
        /// MMSI.
        mmsi: u32,
        /// Name.
        shipname: String,
    },
    /// Message 24, part B: the type, vendor, call sign and dimensions of a class B
    /// station (Table 77).
    ClassBStatic {
        /// Repeat indicator.
        repeat: u8,
        /// MMSI.
        mmsi: u32,
        /// Type of ship and cargo.
        ship_type: u8,
        /// Vendor identifier (three six-bit characters).
        vendor: String,
        /// Unit model code (4 bits).
        model: u8,
        /// Unit serial number (20 bits).
        serial: u32,
        /// Call sign.
        callsign: String,
        /// Dimensions.
        dimensions: Dimensions,
    },
    /// Any other message: its type and the MMSI of its source.
    Other {
        /// Message type (1 to 27).
        msg_type: u8,
        /// Repeat indicator.
        repeat: u8,
        /// MMSI.
        mmsi: u32,
    },
}

impl Message {
    /// The MMSI of the source of the message.
    pub fn mmsi(&self) -> u32 {
        match self {
            Message::PositionReport { mmsi, .. }
            | Message::BaseStation { mmsi, .. }
            | Message::StaticAndVoyage { mmsi, .. }
            | Message::ClassBPosition { mmsi, .. }
            | Message::ClassBName { mmsi, .. }
            | Message::ClassBStatic { mmsi, .. }
            | Message::Other { mmsi, .. } => *mmsi,
        }
    }

    /// The message type (1 to 27).
    pub fn msg_type(&self) -> u8 {
        match self {
            Message::PositionReport { msg_type, .. } | Message::Other { msg_type, .. } => *msg_type,
            Message::BaseStation { .. } => 4,
            Message::StaticAndVoyage { .. } => 5,
            Message::ClassBPosition { .. } => 18,
            Message::ClassBName { .. } | Message::ClassBStatic { .. } => 24,
        }
    }

    /// The position of a position report (messages 1, 2, 3, 4 and 18).
    pub fn position(&self) -> Option<&Position> {
        match self {
            Message::PositionReport { position, .. }
            | Message::BaseStation { position, .. }
            | Message::ClassBPosition { position, .. } => Some(position),
            _ => None,
        }
    }
}

/// The lengths of the messages decoded here (bits), Annex 8.
fn expected_len(msg_type: u8) -> Option<usize> {
    match msg_type {
        1..=4 | 18 => Some(168),
        5 => Some(424),
        _ => None,
    }
}

/// Decode the bit string of a message (most significant bit first).
///
/// # Errors
/// [`comnum_core::Error::Config`] for fewer than 38 bits (the type, repeat indicator and
/// MMSI), a type outside 1 to 27, or a message shorter than its table.
pub fn decode(bits: &[u8]) -> Result<Message> {
    let f = Fields(bits);
    ensure(bits.len() >= 38, || format!("a message has 38 bits at least, got {}", bits.len()))?;
    let msg_type = f.uint(0, 6) as u8;
    ensure((1..=27).contains(&msg_type), || format!("message type {msg_type}"))?;
    let repeat = f.uint(6, 2) as u8;
    let mmsi = f.uint(8, 30) as u32;
    if let Some(n) = expected_len(msg_type) {
        ensure(bits.len() >= n, || format!("message {msg_type} has {n} bits, got {}", bits.len()))?;
    }
    Ok(match msg_type {
        1..=3 => Message::PositionReport {
            msg_type,
            repeat,
            mmsi,
            nav_status: f.uint(38, 4) as u8,
            rot_raw: f.int(42, 8) as i8,
            position: Position::read(&f, 50, 60, 61, 89, 116, 128, 137),
            manoeuvre: f.uint(143, 2) as u8,
            raim: f.uint(148, 1) == 1,
        },
        4 => Message::BaseStation {
            repeat,
            mmsi,
            utc: (
                f.uint(38, 14) as u16,
                f.uint(52, 4) as u8,
                f.uint(56, 5) as u8,
                f.uint(61, 5) as u8,
                f.uint(66, 6) as u8,
                f.uint(72, 6) as u8,
            ),
            position: Position {
                lon_deg: (f.int(79, 28) != 181 * 600_000).then(|| f.int(79, 28) as f64 / 600_000.0),
                lat_deg: (f.int(107, 27) != 91 * 600_000).then(|| f.int(107, 27) as f64 / 600_000.0),
                accurate: f.uint(78, 1) == 1,
                sog_knots: None,
                cog_deg: None,
                heading_deg: None,
                timestamp: f.uint(72, 6) as u8,
            },
            epfd: f.uint(134, 4) as u8,
            raim: f.uint(148, 1) == 1,
        },
        5 => Message::StaticAndVoyage {
            repeat,
            mmsi,
            ais_version: f.uint(38, 2) as u8,
            imo: f.uint(40, 30) as u32,
            callsign: f.text(70, 42),
            shipname: f.text(112, 120),
            ship_type: f.uint(232, 8) as u8,
            dimensions: Dimensions::read(&f, 240),
            epfd: f.uint(270, 4) as u8,
            eta: (f.uint(274, 4) as u8, f.uint(278, 5) as u8, f.uint(283, 5) as u8, f.uint(288, 6) as u8),
            draught_m: f.uint(294, 8) as f64 / 10.0,
            destination: f.text(302, 120),
            dte: f.uint(422, 1) == 1,
        },
        18 => Message::ClassBPosition {
            repeat,
            mmsi,
            position: Position::read(&f, 46, 56, 57, 85, 112, 124, 133),
            cs_unit: f.uint(141, 1) == 1,
            raim: f.uint(147, 1) == 1,
        },
        24 if bits.len() >= 160 => match f.uint(38, 2) {
            0 => Message::ClassBName { repeat, mmsi, shipname: f.text(40, 120) },
            _ => Message::ClassBStatic {
                repeat,
                mmsi,
                ship_type: f.uint(40, 8) as u8,
                vendor: f.text(48, 18),
                model: f.uint(66, 4) as u8,
                serial: f.uint(70, 20) as u32,
                callsign: f.text(90, 42),
                dimensions: Dimensions::read(&f, 132),
            },
        },
        _ => Message::Other { msg_type, repeat, mmsi },
    })
}

/// The bits of a position report (message 1) with the given fields, for tests and
/// simulations: the inverse of [`decode`] for that message (the other fields 0).
pub fn position_report_bits(mmsi: u32, nav_status: u8, position: &Position) -> Vec<u8> {
    let mut bits = vec![0u8; 168];
    let put = |bits: &mut [u8], start: usize, len: usize, v: i64| {
        for i in 0..len {
            bits[start + i] = ((v >> (len - 1 - i)) & 1) as u8;
        }
    };
    put(&mut bits, 0, 6, 1);
    put(&mut bits, 8, 30, i64::from(mmsi));
    put(&mut bits, 38, 4, i64::from(nav_status));
    put(&mut bits, 42, 8, -128);
    put(&mut bits, 50, 10, position.sog_knots.map_or(1023, |v| (v * 10.0).round() as i64));
    put(&mut bits, 60, 1, i64::from(position.accurate));
    put(&mut bits, 61, 28, position.lon_deg.map_or(181 * 600_000, |v| (v * 600_000.0).round() as i64));
    put(&mut bits, 89, 27, position.lat_deg.map_or(91 * 600_000, |v| (v * 600_000.0).round() as i64));
    put(&mut bits, 116, 12, position.cog_deg.map_or(3600, |v| (v * 10.0).round() as i64));
    put(&mut bits, 128, 9, position.heading_deg.map_or(511, i64::from));
    put(&mut bits, 137, 6, i64::from(position.timestamp));
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_position_report_round_trips_through_its_bits() {
        let p = Position {
            lon_deg: Some(-9.8765),
            lat_deg: Some(46.1234),
            accurate: true,
            sog_knots: Some(12.3),
            cog_deg: Some(271.5),
            heading_deg: Some(270),
            timestamp: 42,
        };
        let bits = position_report_bits(200_123_456, 0, &p);
        let m = decode(&bits).unwrap();
        match m {
            Message::PositionReport { msg_type, mmsi, nav_status, rot_raw, position, .. } => {
                assert_eq!((msg_type, mmsi, nav_status, rot_raw), (1, 200_123_456, 0, -128));
                assert!((position.lon_deg.unwrap() - p.lon_deg.unwrap()).abs() < 1e-6);
                assert!((position.lat_deg.unwrap() - p.lat_deg.unwrap()).abs() < 1e-6);
                assert_eq!(position.sog_knots, p.sog_knots);
                assert_eq!(position.cog_deg, p.cog_deg);
                assert_eq!((position.heading_deg, position.timestamp, position.accurate), (Some(270), 42, true));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(m.mmsi(), 200_123_456);
        assert_eq!(m.msg_type(), 1);
        // the "not available" values
        let none = Position {
            lon_deg: None,
            lat_deg: None,
            accurate: false,
            sog_knots: None,
            cog_deg: None,
            heading_deg: None,
            timestamp: 60,
        };
        let p2 = decode(&position_report_bits(1, 15, &none)).unwrap();
        assert_eq!(p2.position(), Some(&none));
    }

    #[test]
    fn texts_stop_at_the_at_sign_and_errors_name_the_problem() {
        // "AB@" in six-bit characters: 1, 2, 0
        let bits = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(Fields(&bits).text(0, 18), "AB");
        assert!(decode(&[0; 10]).is_err());
        let mut bits = vec![0u8; 168];
        bits[5] = 1; // type 1
        assert!(decode(&bits[..100]).is_err());
        bits[2] = 1; // type 9
        assert!(matches!(decode(&bits).unwrap(), Message::Other { msg_type: 9, .. }));
    }
}
