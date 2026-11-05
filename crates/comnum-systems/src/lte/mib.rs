//! The fields of the master information block of LTE (TS 36.331, Sec. 6.2.2,
//! `MasterInformationBlock`), in the order of their unaligned PER encoding: the downlink
//! bandwidth (3 bits, $N_{\mathrm{RB}} = 6, 15, 25, 50, 75, 100$), the PHICH duration (1
//! bit, normal or extended), the PHICH resource (2 bits, $N_{g} = 1/6, 1/2, 1, 2$), the 8
//! most significant bits of the 10-bit system frame number, then 10 bits that the later
//! releases fill (scheduling of SIB1-BR, system information unchanged, part of the
//! EARFCN) or leave spare, kept as they are received. The two least significant bits of
//! the frame number are the position of the frame in the 40 ms period of the PBCH, that
//! the receiver learns from the scrambling ([`super::search::decode_mib`]).
//!
//! # Examples
//! ```
//! use comnum_systems::lte::mib::{Mib, PhichDuration, PhichResource};
//! let mib = Mib { n_rb: 25, phich_duration: PhichDuration::Normal, phich_resource: PhichResource::One, sfn_msb: 0x9c, rest: [0; 10] };
//! let bits = mib.to_bits();
//! assert_eq!(&bits[..6], &[0, 1, 0, 0, 1, 0]);      // n25, normal, one
//! assert_eq!(Mib::from_bits(&bits)?, mib);
//! assert_eq!(mib.sfn(3), 4 * 0x9c + 3);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 36.331, *E-UTRA; Radio Resource Control (RRC); Protocol specification*,
//!    Sec. 6.2.2 (`MasterInformationBlock`).

use super::bch::MIB_BITS;
use comnum_core::{Error, Result, ensure_shape};

/// The PHICH duration of the MIB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhichDuration {
    /// Normal.
    Normal,
    /// Extended.
    Extended,
}

/// The PHICH resource $N_{g}$ of the MIB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhichResource {
    /// $N_{g} = 1/6$.
    OneSixth,
    /// $N_{g} = 1/2$.
    Half,
    /// $N_{g} = 1$.
    One,
    /// $N_{g} = 2$.
    Two,
}

impl PhichResource {
    /// $N_{g}$.
    pub fn ng(self) -> f64 {
        match self {
            PhichResource::OneSixth => 1.0 / 6.0,
            PhichResource::Half => 0.5,
            PhichResource::One => 1.0,
            PhichResource::Two => 2.0,
        }
    }
}

/// The downlink bandwidths of the field `dl-Bandwidth`, in resource blocks.
const BANDWIDTHS: [usize; 6] = [6, 15, 25, 50, 75, 100];

/// A master information block (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mib {
    /// The downlink bandwidth $N_{\mathrm{RB}}$.
    pub n_rb: usize,
    /// The PHICH duration.
    pub phich_duration: PhichDuration,
    /// The PHICH resource.
    pub phich_resource: PhichResource,
    /// The 8 most significant bits of the system frame number.
    pub sfn_msb: u8,
    /// The last 10 bits, as received.
    pub rest: [u8; 10],
}

impl Mib {
    /// The fields of the 24 bits of a MIB.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of bits than 24; [`Error::Config`] for a
    /// bandwidth code above 5.
    pub fn from_bits(bits: &[u8]) -> Result<Mib> {
        ensure_shape(bits.len() == MIB_BITS, || format!("a MIB has {MIB_BITS} bits, got {}", bits.len()))?;
        let value = |b: &[u8]| b.iter().fold(0usize, |v, &x| 2 * v + usize::from(x & 1));
        let bw = value(&bits[..3]);
        let n_rb =
            *BANDWIDTHS.get(bw).ok_or_else(|| Error::Config(format!("dl-Bandwidth {bw} is not a bandwidth of LTE")))?;
        let phich_duration = if bits[3] & 1 == 0 { PhichDuration::Normal } else { PhichDuration::Extended };
        let phich_resource =
            [PhichResource::OneSixth, PhichResource::Half, PhichResource::One, PhichResource::Two][value(&bits[4..6])];
        let mut rest = [0u8; 10];
        rest.copy_from_slice(&bits[14..]);
        Ok(Mib { n_rb, phich_duration, phich_resource, sfn_msb: value(&bits[6..14]) as u8, rest })
    }

    /// The 24 bits of the MIB.
    pub fn to_bits(&self) -> Vec<u8> {
        let bits_of = |v: usize, n: usize| (0..n).rev().map(move |i| ((v >> i) & 1) as u8);
        let bw = BANDWIDTHS.iter().position(|&n| n == self.n_rb).unwrap_or(0);
        let res = self.phich_resource as usize;
        bits_of(bw, 3)
            .chain([u8::from(self.phich_duration == PhichDuration::Extended)])
            .chain(bits_of(res, 2))
            .chain(bits_of(usize::from(self.sfn_msb), 8))
            .chain(self.rest.iter().map(|b| b & 1))
            .collect()
    }

    /// The system frame number of the frame `frame_in_period` (0 to 3) of the 40 ms
    /// period: $4 \cdot$ `sfn_msb` $+$ `frame_in_period`.
    pub fn sfn(&self, frame_in_period: usize) -> usize {
        4 * usize::from(self.sfn_msb) + frame_in_period % 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fields_round_trip_and_invalid_bandwidths_are_errors() {
        for (i, &n_rb) in BANDWIDTHS.iter().enumerate() {
            let mib = Mib {
                n_rb,
                phich_duration: if i % 2 == 0 { PhichDuration::Normal } else { PhichDuration::Extended },
                phich_resource: [PhichResource::OneSixth, PhichResource::Half, PhichResource::One, PhichResource::Two]
                    [i % 4],
                sfn_msb: (37 * i) as u8,
                rest: std::array::from_fn(|j| ((i + j) % 2) as u8),
            };
            assert_eq!(Mib::from_bits(&mib.to_bits()).unwrap(), mib);
        }
        let mut bits = vec![0u8; 24];
        bits[..3].copy_from_slice(&[1, 1, 0]);
        assert!(Mib::from_bits(&bits).is_err() && Mib::from_bits(&bits[1..]).is_err());
        assert_eq!(PhichResource::OneSixth.ng() * 6.0, 1.0);
    }
}
