//! DVB-S (ETSI EN 300 421 V1.1.2): the transmitter and a frame-aligned receiver of the
//! satellite link of the MPEG-2 transport stream.
//!
//! # Model
//!
//! The transmitter of Sec. 4 (Figure 1):
//!
//! 1. the energy dispersal, the outer code RS(204, 188) and the convolutional
//!    interleaver $I = 12$, $M = 17$ of [`crate::dvb`] (Secs. 4.4.1 and 4.4.2), over a
//!    burst ([`dvb::burst_encode`]);
//! 2. the inner code of constraint length 7, $X$ from $171_{8}$ and $Y$ from $133_{8}$,
//!    punctured by Table 2 ([`InnerRate`], Sec. 4.4.3);
//! 3. the transmitted bits sent alternately on I and Q, which gives the I and Q rows of
//!    Table 2 (rate 3/4: I = $X_{1} Y_{2}$, Q = $Y_{1} X_{3}$), mapped on the Gray QPSK of
//!    Figure 5 with absolute mapping (Sec. 4.5): $(b_{I}, b_{Q}) \mapsto ((1 - 2 b_{I}) +
//!    j (1 - 2 b_{Q})) / \sqrt{2}$, unit energy ([`Dvbs::constellation`]).
//!
//! The baseband shaping, a root-raised cosine of roll-off 0.35 (Sec. 4.5), is left to the
//! pulse shaping of `comnum_dsp` (`rrc_taps`): the symbols of [`Dvbs::modulate`] are at
//! one sample per symbol.
//!
//! The broadcast is continuous; the chain here transmits a **burst** of whole packets
//! that a receiver aligned on its first symbol decodes completely: the interleaver
//! starts empty and is flushed with $I - 1 = 11$ packets of zeros (its latency of
//! $I M (I - 1) = 2244$ bytes), and the inner code starts in state 0 and is returned to
//! it by at least 6 zero bits, padded to whole puncturing periods and to an even number
//! of transmitted bits. A long burst is the continuous link. The receiver
//! ([`Dvbs::decode`]) computes the exact LLRs of the QPSK bits, depunctures them, decodes
//! the inner code by Viterbi (soft decisions), deinterleaves, decodes RS(204, 188) and
//! removes the energy dispersal.
//!
//! The standard gives the required $E_{b}/N_{0}$ of each rate for a BER of $2 \cdot
//! 10^{-4}$ after the Viterbi decoder, "QEF after Reed-Solomon" (Table 3: 4.5, 5.0, 5.5,
//! 6.0 and 6.4 dB from 1/2 to 7/8), $E_{b}$ per useful bit before the outer code, a modem
//! implementation margin of 0.8 dB included. [`Dvbs::useful_bit_rate`] is the useful bit
//! rate of Annexes C and D, $R_{u} = 2 R_{s} r \cdot 188 / 204$.
//!
//! # Examples
//! ```
//! use comnum_core::rng::Rng;
//! use comnum_systems::dvb::{self, InnerRate};
//! use comnum_systems::dvbs::Dvbs;
//! let link = Dvbs::new(InnerRate::R3_4)?;
//! let packets = dvb::random_packets(8, &mut Rng::new(1));
//! let x = link.modulate(&packets)?;
//! assert_eq!(x.len(), link.symbols_len(8));
//! let rx = link.decode(&x, 1e-2)?;                    // a nearly noiseless channel
//! assert_eq!(rx.packets, packets);
//! assert_eq!((rx.corrected, rx.failed), (0, 0));
//! // Annex D, Table D.1: 25.776 Mbaud at rate 3/4 carry 35.631 Mbit/s of useful data
//! assert!((link.useful_bit_rate(25.776e6) - 35.631e6).abs() < 1e3);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. ETSI EN 300 421 V1.1.2, *Digital Video Broadcasting (DVB); Framing structure,
//!    channel coding and modulation for 11/12 GHz satellite services*, 1997, Secs. 4.4
//!    and 4.5, Tables 2 and 3, Figure 5, Annexes C and D.

use crate::dvb::{self, CODED_PACKET_LEN, InnerRate, OuterDecoded, PACKET_LEN};
use comnum_core::{Error, Result};
use comnum_fec::conv::{Trellis, viterbi};
use comnum_fec::puncture::Puncturer;
use comnum_mapping::constellation::Constellation;
use comnum_math::C64;
use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

/// The constraint length $K$ of the inner code.
const CONSTRAINT: usize = 7;

/// A DVB-S transmitter and frame-aligned receiver of one inner code rate (see the
/// module).
#[derive(Clone, Debug)]
pub struct Dvbs {
    rate: InnerRate,
    trellis: Trellis,
    puncturer: Puncturer,
}

/// The output of [`Dvbs::decode`].
pub type DvbsDecoded = OuterDecoded;

impl Dvbs {
    /// The link of the inner code rate `rate`.
    ///
    /// # Errors
    /// Never: the codes are those of the standard.
    pub fn new(rate: InnerRate) -> Result<Dvbs> {
        Ok(Dvbs { rate, trellis: dvb::inner_trellis()?, puncturer: rate.puncturer()? })
    }

    /// The inner code rate.
    pub fn rate(&self) -> InnerRate {
        self.rate
    }

    /// The Gray QPSK of Figure 5, unit energy: the label $2 b_{I} + b_{Q}$ is the point
    /// $((1 - 2 b_{I}) + j (1 - 2 b_{Q})) / \sqrt{2}$ (00 in the first quadrant, 10 in the
    /// second, 11 in the third, 01 in the fourth).
    ///
    /// # Errors
    /// Never: four points.
    pub fn constellation() -> Result<Constellation> {
        let points = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
            .map(|(i, q)| C64::new(i * FRAC_1_SQRT_2, q * FRAC_1_SQRT_2));
        Constellation::from_points(points.to_vec())
    }

    /// The useful bit rate (bit/s) at the symbol rate `symbol_rate` (baud),
    /// $R_{u} = 2 R_{s} r \cdot 188/204$ (Annexes C and D, Tables C.1 and D.1).
    pub fn useful_bit_rate(&self, symbol_rate: f64) -> f64 {
        2.0 * symbol_rate * self.rate.value() * PACKET_LEN as f64 / CODED_PACKET_LEN as f64
    }

    /// The $E_{s}/N_{0}$ (dB) of an $E_{b}/N_{0}$ (dB) per useful bit, the abscissa of
    /// Table 3: $E_{s}/N_{0} = E_{b}/N_{0} + 10 \log_{10}(2 r \cdot 188/204)$.
    pub fn esn0_db(&self, ebn0_db: f64) -> f64 {
        ebn0_db + 10.0 * (2.0 * self.rate.value() * PACKET_LEN as f64 / CODED_PACKET_LEN as f64).log10()
    }

    /// The input bits of the inner code for `n_bytes` bytes, padding included: at least
    /// $K - 1$ zero bits, up to whole puncturing periods carrying an even number of bits.
    fn padded_bits(&self, n_bytes: usize) -> usize {
        let (period, kept) = self.rate.fraction();
        let unit = if kept % 2 == 0 { period } else { 2 * period };
        (8 * n_bytes + CONSTRAINT - 1).div_ceil(unit) * unit
    }

    /// QPSK symbols of a burst of `n_packets` packets.
    pub fn symbols_len(&self, n_packets: usize) -> usize {
        let (period, kept) = self.rate.fraction();
        self.padded_bits(dvb::burst_bytes(n_packets)) / period * kept / 2
    }

    /// The inner coding of the bytes of [`dvb::burst_encode`]: the punctured inner code from
    /// state 0, padded (see the module), on the QPSK of Figure 5, I then Q.
    ///
    /// # Errors
    /// Never for whole bytes; the error of the puncturer otherwise.
    pub fn inner(&self, bytes: &[u8]) -> Result<Vec<C64>> {
        let mut bits = dvb::bytes_to_bits(bytes);
        bits.resize(self.padded_bits(bytes.len()), 0);
        let sent = self.puncturer.puncture(&self.trellis.encode(&bits, false))?;
        let level = |b: u8| if b == 0 { FRAC_1_SQRT_2 } else { -FRAC_1_SQRT_2 };
        Ok(sent.chunks_exact(2).map(|p| C64::new(level(p[0]), level(p[1]))).collect())
    }

    /// The symbols of a burst of whole packets, [`Dvbs::inner`] of [`dvb::burst_encode`].
    ///
    /// # Errors
    /// As [`dvb::burst_encode`].
    pub fn modulate(&self, packets: &[u8]) -> Result<Vec<C64>> {
        self.inner(&dvb::burst_encode(packets)?)
    }

    /// The exact LLRs (positive for a 0) of the transmitted bits, I then Q of every
    /// symbol, for a complex noise of variance `noise_var` per symbol:
    /// $L = 2 \sqrt{2}\, y / \sigma^{2}$ on each axis.
    pub fn llr(&self, rx: &[C64], noise_var: f64) -> Vec<f64> {
        let g = 2.0 * SQRT_2 / noise_var;
        rx.iter().flat_map(|z| [g * z.re, g * z.im]).collect()
    }

    /// The number of packets of a burst of `n_symbols` symbols.
    fn burst_packets(&self, n_symbols: usize) -> Result<usize> {
        let (period, kept) = self.rate.fraction();
        let bits = 2 * n_symbols * period;
        let max_bytes = (bits / kept).saturating_sub(CONSTRAINT - 1) / 8;
        let n = (max_bytes / CODED_PACKET_LEN).checked_sub(dvb::INTERLEAVER_BRANCHES - 1);
        match n {
            Some(n) if n > 0 && self.symbols_len(n) == n_symbols => Ok(n),
            _ => {
                Err(Error::Shape(format!("{n_symbols} symbols is not a DVB-S burst of the rate {}", self.rate.name())))
            }
        }
    }

    /// The bytes decided by the inner decoder (Viterbi, soft decisions) from the received
    /// symbols of a burst: the estimate of [`dvb::burst_encode`].
    ///
    /// # Errors
    /// [`Error::Shape`] if `rx` is not the length of a burst ([`Dvbs::symbols_len`]).
    pub fn inner_decode(&self, rx: &[C64], noise_var: f64) -> Result<Vec<u8>> {
        let n_bytes = dvb::burst_bytes(self.burst_packets(rx.len())?);
        let steps = self.padded_bits(n_bytes);
        let llr = self.puncturer.depuncture(&self.llr(rx, noise_var), 2 * steps)?;
        let mut bits = viterbi(&self.trellis, &llr, true)?;
        bits.truncate(8 * n_bytes);
        Ok(dvb::bits_to_bytes(&bits))
    }

    /// The packets of the received symbols of a burst, aligned on its first symbol, the
    /// carrier recovered: [`dvb::burst_decode`] of [`Dvbs::inner_decode`].
    ///
    /// # Errors
    /// As [`Dvbs::inner_decode`].
    pub fn decode(&self, rx: &[C64], noise_var: f64) -> Result<DvbsDecoded> {
        dvb::burst_decode(&self.inner_decode(rx, noise_var)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    #[test]
    fn every_rate_round_trips_a_burst() {
        let packets = dvb::random_packets(9, &mut Rng::new(3));
        for r in InnerRate::ALL {
            let link = Dvbs::new(r).unwrap();
            let x = link.modulate(&packets).unwrap();
            assert_eq!(x.len(), link.symbols_len(9), "rate {}", r.name());
            assert!(x.iter().all(|z| (z.norm_sqr() - 1.0).abs() < 1e-12));
            let d = link.decode(&x, 0.05).unwrap();
            assert_eq!(d.packets, packets, "rate {}", r.name());
        }
    }

    #[test]
    fn the_i_and_q_rows_are_those_of_table_2() {
        // a single 1 at input step t gives the impulse response; read which coded bit
        // lands on I or Q from the pattern and the alternation
        let rows = |r: InnerRate| -> (Vec<String>, Vec<String>) {
            let p = r.puncturer().unwrap();
            let (period, kept) = r.fraction();
            let steps = if kept % 2 == 0 { period } else { 2 * period };
            let names: Vec<String> = (1..=steps).flat_map(|t| [format!("X{t}"), format!("Y{t}")]).collect();
            let sel = p.selection(2 * steps).unwrap();
            let order: Vec<String> = sel.positions().iter().map(|&i| names[i].clone()).collect();
            (order.iter().step_by(2).cloned().collect(), order.iter().skip(1).step_by(2).cloned().collect())
        };
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_string()).collect::<Vec<String>>();
        assert_eq!(rows(InnerRate::R1_2), (s(&["X1"]), s(&["Y1"])));
        assert_eq!(rows(InnerRate::R2_3), (s(&["X1", "Y2", "Y3"]), s(&["Y1", "X3", "Y4"])));
        assert_eq!(rows(InnerRate::R3_4), (s(&["X1", "Y2"]), s(&["Y1", "X3"])));
        assert_eq!(rows(InnerRate::R5_6), (s(&["X1", "Y2", "Y4"]), s(&["Y1", "X3", "X5"])));
        assert_eq!(rows(InnerRate::R7_8), (s(&["X1", "Y2", "Y4", "Y6"]), s(&["Y1", "Y3", "X5", "X7"])));
    }

    #[test]
    fn the_constellation_is_that_of_figure_5() {
        let c = Dvbs::constellation().unwrap();
        let p = c.points();
        // labels 00, 01, 10, 11 (I bit first): first, fourth, second, third quadrants
        assert!(p[0].re > 0.0 && p[0].im > 0.0);
        assert!(p[1].re > 0.0 && p[1].im < 0.0);
        assert!(p[2].re < 0.0 && p[2].im > 0.0);
        assert!(p[3].re < 0.0 && p[3].im < 0.0);
    }

    #[test]
    fn a_burst_of_another_length_is_refused() {
        let link = Dvbs::new(InnerRate::R1_2).unwrap();
        let x = link.modulate(&dvb::random_packets(2, &mut Rng::new(4))).unwrap();
        assert!(link.decode(&x[..x.len() - 1], 0.1).is_err());
        assert!(dvb::burst_decode(&[0u8; 100]).is_err());
    }

    #[test]
    fn the_rates_of_tables_c1_and_d1() {
        // Table D.1 (33 MHz transponder, 25.776 Mbaud): Ru per rate, Mbit/s
        for (r, ru) in InnerRate::ALL.into_iter().zip([23.754, 31.672, 35.631, 39.590, 41.570]) {
            assert!((Dvbs::new(r).unwrap().useful_bit_rate(25.776e6) / 1e6 - ru).abs() < 6e-4, "rate {}", r.name());
        }
        // Table C.1, first row: 42.2 Mbaud, Ru 38.9 Mbit/s at 1/2 (printed digits)
        assert!((Dvbs::new(InnerRate::R1_2).unwrap().useful_bit_rate(42.2e6) / 1e6 - 38.9).abs() < 0.05);
    }
}
