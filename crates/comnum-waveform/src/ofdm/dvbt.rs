//! The OFDM frame of DVB-T (ETSI EN 300 744 V1.6.2, Secs. 4.4 to 4.6): the 2K and 8K
//! modes, the guard intervals, the scattered and continual pilots, the carriers of the
//! transmission parameter signalling (TPS) and the cells left to the data.
//!
//! # Frame
//!
//! A frame is [`SYMBOLS_PER_FRAME`] = 68 OFDM symbols $l = 0, \dots, 67$, a super-frame
//! four frames. A symbol has $K$ carriers $k = 0, \dots, K_{\max}$ (Table 4): $K = 1705$
//! in the 2K mode, $K = 6817$ in the 8K mode, spaced by $1/T_{U}$, $T_{U} = 2048\,T$ or
//! $8192\,T$, the elementary period $T = 7/(8B)$ of a channel of bandwidth $B$ (7/64 µs
//! at 8 MHz, Sec. 4.4); the guard interval $\Delta = T_{U}/4$, $T_{U}/8$, $T_{U}/16$ or
//! $T_{U}/32$ is the cyclic continuation of the useful part, in front of it (Table 5).
//! The carrier $k$ is emitted at $k' / T_{U}$, $k' = k - (K_{\max} + K_{\min})/2$, the
//! centring of [`Ofdm`] ([`DvbtFrame::ofdm`]).
//!
//! # Reference signals and cells
//!
//! - The pilots carry $\pm 4/3$ ([`PILOT_BOOST`], $E|c|^{2} = 16/9$, Sec. 4.5.5), the sign
//!   $2(1/2 - w_{k})$ of the PRBS $X^{11} + X^{2} + 1$ started at all ones on carrier
//!   $K_{\min}$, one value per carrier ([`pilot_prbs`], Sec. 4.5.2, Figure 10);
//! - the scattered pilots of symbol $l$ are the carriers $k = 3 (l \bmod 4) + 12 p$
//!   (Sec. 4.5.3), the continual pilots those of Table 7, in every symbol (45 in the 2K
//!   mode, 177 in the 8K mode, Sec. 4.5.4);
//! - the TPS carriers of Table 8 (17 and 68, Sec. 4.6) carry, all alike, the
//!   differential BPSK of one bit per symbol: $\pm 2(1/2 - w_{k})$, the sign $d_{0} = 1$ in
//!   the first symbol of a frame and $d_{l} = d_{l-1} (1 - 2 s_{l})$ after
//!   ([`tps_signs`], Sec. 4.6.3);
//! - the other carriers are the data cells, 1512 (2K) or 6048 (8K) in every symbol
//!   (Sec. 4.5.1), filled in the order of increasing $k$ (the order that reproduces the
//!   interleaving example of Annex C, Table C.1; the text does not state it).
//!
//! # Examples
//! ```
//! use comnum_waveform::ofdm::dvbt::{DvbtFrame, Mode};
//! let frame = DvbtFrame::new(Mode::K2);
//! for l in 0..4 {
//!     assert_eq!(frame.data_carriers(l).len(), 1512);      // Sec. 4.5.1
//! }
//! // "PRBS sequence starts: 1111111111100", Figure 10
//! assert_eq!(&comnum_waveform::ofdm::dvbt::pilot_prbs(13), &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
//! ```
//!
//! # References
//!
//! 1. ETSI EN 300 744 V1.6.2, *Digital Video Broadcasting (DVB); Framing structure,
//!    channel coding and modulation for digital terrestrial television*, 2015, Secs. 4.4
//!    to 4.6, Tables 4, 5, 7 and 8.

use super::Ofdm;
use comnum_core::{Error, Result, ensure_shape};
use comnum_math::C64;
use comnum_math::sequences::lfsr;
use std::str::FromStr;

/// OFDM symbols of a frame.
pub const SYMBOLS_PER_FRAME: usize = 68;
/// Frames of a super-frame.
pub const FRAMES_PER_SUPERFRAME: usize = 4;
/// The amplitude of a pilot, $4/3$ ($E|c|^{2} = 16/9$, Sec. 4.5.5).
pub const PILOT_BOOST: f64 = 4.0 / 3.0;

/// The continual pilots of the 8K mode, Table 7; the 2K mode uses the first 45.
const CONTINUAL_PILOTS: [u16; 177] = [
    0, 48, 54, 87, 141, 156, 192, 201, 255, 279, 282, 333, 432, 450, 483, 525, 531, 618, 636, 714, 759, 765, 780, 804,
    873, 888, 918, 939, 942, 969, 984, 1050, 1101, 1107, 1110, 1137, 1140, 1146, 1206, 1269, 1323, 1377, 1491, 1683,
    1704, 1752, 1758, 1791, 1845, 1860, 1896, 1905, 1959, 1983, 1986, 2037, 2136, 2154, 2187, 2229, 2235, 2322, 2340,
    2418, 2463, 2469, 2484, 2508, 2577, 2592, 2622, 2643, 2646, 2673, 2688, 2754, 2805, 2811, 2814, 2841, 2844, 2850,
    2910, 2973, 3027, 3081, 3195, 3387, 3408, 3456, 3462, 3495, 3549, 3564, 3600, 3609, 3663, 3687, 3690, 3741, 3840,
    3858, 3891, 3933, 3939, 4026, 4044, 4122, 4167, 4173, 4188, 4212, 4281, 4296, 4326, 4347, 4350, 4377, 4392, 4458,
    4509, 4515, 4518, 4545, 4548, 4554, 4614, 4677, 4731, 4785, 4899, 5091, 5112, 5160, 5166, 5199, 5253, 5268, 5304,
    5313, 5367, 5391, 5394, 5445, 5544, 5562, 5595, 5637, 5643, 5730, 5748, 5826, 5871, 5877, 5892, 5916, 5985, 6000,
    6030, 6051, 6054, 6081, 6096, 6162, 6213, 6219, 6222, 6249, 6252, 6258, 6318, 6381, 6435, 6489, 6603, 6795, 6816,
];

/// The TPS carriers of the 8K mode, Table 8; the 2K mode uses the first 17.
const TPS_CARRIERS: [u16; 68] = [
    34, 50, 209, 346, 413, 569, 595, 688, 790, 901, 1073, 1219, 1262, 1286, 1469, 1594, 1687, 1738, 1754, 1913, 2050,
    2117, 2273, 2299, 2392, 2494, 2605, 2777, 2923, 2966, 2990, 3173, 3298, 3391, 3442, 3458, 3617, 3754, 3821, 3977,
    4003, 4096, 4198, 4309, 4481, 4627, 4670, 4694, 4877, 5002, 5095, 5146, 5162, 5321, 5458, 5525, 5681, 5707, 5800,
    5902, 6013, 6185, 6331, 6374, 6398, 6581, 6706, 6799,
];

/// The transmission mode (Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// 2K: 1705 carriers, an FFT of 2048.
    K2,
    /// 8K: 6817 carriers, an FFT of 8192.
    K8,
}

impl Mode {
    /// The number of carriers $K$.
    pub fn carriers(self) -> usize {
        match self {
            Mode::K2 => 1705,
            Mode::K8 => 6817,
        }
    }

    /// The size of the FFT, $T_{U} / T$.
    pub fn fft_len(self) -> usize {
        match self {
            Mode::K2 => 2048,
            Mode::K8 => 8192,
        }
    }

    /// The data cells of a symbol, $N_{\max}$ (Sec. 4.3.4.2).
    pub fn data_cells(self) -> usize {
        match self {
            Mode::K2 => 1512,
            Mode::K8 => 6048,
        }
    }

    /// The continual pilots, Table 7.
    pub fn continual_pilots(self) -> &'static [u16] {
        match self {
            Mode::K2 => &CONTINUAL_PILOTS[..45],
            Mode::K8 => &CONTINUAL_PILOTS,
        }
    }

    /// The TPS carriers, Table 8.
    pub fn tps_carriers(self) -> &'static [u16] {
        match self {
            Mode::K2 => &TPS_CARRIERS[..17],
            Mode::K8 => &TPS_CARRIERS,
        }
    }

    /// `2K` or `8K`.
    pub fn name(self) -> &'static str {
        match self {
            Mode::K2 => "2K",
            Mode::K8 => "8K",
        }
    }
}

impl FromStr for Mode {
    type Err = Error;

    /// `2K` or `8K` (any case).
    fn from_str(s: &str) -> Result<Mode> {
        match s.trim().to_ascii_uppercase().as_str() {
            "2K" => Ok(Mode::K2),
            "8K" => Ok(Mode::K8),
            _ => Err(Error::Config(format!("the DVB-T mode must be 2K or 8K, got {s:?}"))),
        }
    }
}

/// The guard interval $\Delta / T_{U}$ (Table 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Guard {
    /// 1/4.
    G1_4,
    /// 1/8.
    G1_8,
    /// 1/16.
    G1_16,
    /// 1/32.
    G1_32,
}

impl Guard {
    /// The four guard intervals.
    pub const ALL: [Guard; 4] = [Guard::G1_4, Guard::G1_8, Guard::G1_16, Guard::G1_32];

    /// The denominator $T_{U} / \Delta$.
    pub fn denominator(self) -> usize {
        match self {
            Guard::G1_4 => 4,
            Guard::G1_8 => 8,
            Guard::G1_16 => 16,
            Guard::G1_32 => 32,
        }
    }

    /// The guard interval in samples of $T$.
    pub fn samples(self, mode: Mode) -> usize {
        mode.fft_len() / self.denominator()
    }

    /// `1/4`, `1/8`, `1/16` or `1/32`.
    pub fn name(self) -> &'static str {
        match self {
            Guard::G1_4 => "1/4",
            Guard::G1_8 => "1/8",
            Guard::G1_16 => "1/16",
            Guard::G1_32 => "1/32",
        }
    }
}

impl FromStr for Guard {
    type Err = Error;

    /// `1/4`, `1/8`, `1/16` or `1/32`.
    fn from_str(s: &str) -> Result<Guard> {
        Guard::ALL
            .into_iter()
            .find(|g| g.name() == s.trim())
            .ok_or_else(|| Error::Config(format!("the guard interval must be 1/4, 1/8, 1/16 or 1/32, got {s:?}")))
    }
}

/// The elementary period $T = 7 / (8 B)$ (s) of a channel of bandwidth $B$ (Hz): 7/64 µs
/// at 8 MHz, 1/8 µs at 7 MHz, 7/48 µs at 6 MHz, 7/40 µs at 5 MHz (Sec. 4.4, Annexes E
/// and G).
pub fn elementary_period(bandwidth_hz: f64) -> f64 {
    7.0 / (8.0 * bandwidth_hz)
}

/// The first `n` values $w_{k}$ of the PRBS of the pilots, $X^{11} + X^{2} + 1$, its 11
/// registers started at one (Sec. 4.5.2, Figure 10): $w_{k+11} = w_{k} \oplus w_{k+2}$.
pub fn pilot_prbs(n: usize) -> Vec<u8> {
    lfsr(&[0, 2], 11, &[1; 11], n)
}

/// The signs $d_{l}$ of the TPS carriers over a frame for the 68 TPS bits $s_{0}, \dots,
/// s_{67}$ of the frame: $d_{0} = 1$ (the first symbol is the reference, $s_{0}$
/// initialises the differential BPSK), $d_{l} = d_{l-1} (1 - 2 s_{l})$ (Sec. 4.6.3).
///
/// # Errors
/// [`Error::Shape`] unless there are 68 bits.
pub fn tps_signs(bits: &[u8]) -> Result<Vec<f64>> {
    ensure_shape(bits.len() == SYMBOLS_PER_FRAME, || format!("{} TPS bits for a frame of 68", bits.len()))?;
    let mut d = 1.0;
    Ok(bits
        .iter()
        .enumerate()
        .map(|(l, &s)| {
            if l > 0 && s & 1 == 1 {
                d = -d;
            }
            d
        })
        .collect())
}

/// What a carrier of a symbol carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell {
    /// A continual pilot (Table 7); it may also be a scattered pilot.
    ContinualPilot,
    /// A scattered pilot, $k = 3 (l \bmod 4) + 12 p$.
    ScatteredPilot,
    /// A TPS carrier (Table 8).
    Tps,
    /// A data cell.
    Data,
}

/// The cells of the frame of a mode: what every carrier of every symbol carries, the
/// pilot PRBS and the data carriers (see the module).
#[derive(Clone, Debug)]
pub struct DvbtFrame {
    mode: Mode,
    /// $2(1/2 - w_{k})$ for every carrier.
    reference: Vec<f64>,
    /// The cells of a symbol for $l \bmod 4 = 0, \dots, 3$.
    cells: [Vec<Cell>; 4],
    /// The data carriers for $l \bmod 4 = 0, \dots, 3$, increasing.
    data: [Vec<usize>; 4],
}

impl DvbtFrame {
    /// The frame of the mode `mode`.
    pub fn new(mode: Mode) -> DvbtFrame {
        let k_total = mode.carriers();
        let reference = pilot_prbs(k_total).into_iter().map(|w| 1.0 - 2.0 * f64::from(w)).collect();
        let mut continual = vec![false; k_total];
        for &k in mode.continual_pilots() {
            continual[usize::from(k)] = true;
        }
        let mut tps = vec![false; k_total];
        for &k in mode.tps_carriers() {
            tps[usize::from(k)] = true;
        }
        let cells: [Vec<Cell>; 4] = std::array::from_fn(|r| {
            (0..k_total)
                .map(|k| {
                    if continual[k] {
                        Cell::ContinualPilot
                    } else if k >= 3 * r && (k - 3 * r) % 12 == 0 {
                        Cell::ScatteredPilot
                    } else if tps[k] {
                        Cell::Tps
                    } else {
                        Cell::Data
                    }
                })
                .collect()
        });
        let data = std::array::from_fn(|r| (0..k_total).filter(|&k| cells[r][k] == Cell::Data).collect());
        DvbtFrame { mode, reference, cells, data }
    }

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// What carrier `k` of symbol `l` carries.
    pub fn cell(&self, l: usize, k: usize) -> Cell {
        self.cells[l % 4][k]
    }

    /// The data carriers of symbol `l`, increasing ($N_{\max}$ of them).
    pub fn data_carriers(&self, l: usize) -> &[usize] {
        &self.data[l % 4]
    }

    /// The sign $2(1/2 - w_{k})$ of the reference signals of carrier `k`.
    pub fn reference(&self, k: usize) -> f64 {
        self.reference[k]
    }

    /// The mean power $E|c|^{2}$ over the $K$ carriers of a symbol: the data and TPS cells
    /// at 1, the pilots at 16/9; the same for every symbol (1.0803 in the 2K mode, 1.0800
    /// in the 8K mode).
    pub fn mean_carrier_power(&self) -> f64 {
        let power: f64 = self.cells[0]
            .iter()
            .map(|c| match c {
                Cell::ContinualPilot | Cell::ScatteredPilot => PILOT_BOOST * PILOT_BOOST,
                Cell::Tps | Cell::Data => 1.0,
            })
            .sum();
        power / self.mode.carriers() as f64
    }

    /// The $K$ cells of symbol `l`: the pilots, the TPS carriers with the sign
    /// `tps_sign` ($d_{l}$ of [`tps_signs`]) and the $N_{\max}$ values `data` on the data
    /// carriers.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `data` has $N_{\max}$ values.
    pub fn symbol(&self, l: usize, data: &[C64], tps_sign: f64) -> Result<Vec<C64>> {
        ensure_shape(data.len() == self.mode.data_cells(), || {
            format!("{} data cells for {} in a {} symbol", data.len(), self.mode.data_cells(), self.mode.name())
        })?;
        let mut out: Vec<C64> = self.cells[l % 4]
            .iter()
            .zip(&self.reference)
            .map(|(c, &r)| match c {
                Cell::ContinualPilot | Cell::ScatteredPilot => C64::new(PILOT_BOOST * r, 0.0),
                Cell::Tps => C64::new(tps_sign * r, 0.0),
                Cell::Data => C64::new(0.0, 0.0),
            })
            .collect();
        for (&k, &v) in self.data[l % 4].iter().zip(data) {
            out[k] = v;
        }
        Ok(out)
    }

    /// The data cells of the $K$ received cells of symbol `l`.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `cells` has $K$ values.
    pub fn data_of(&self, l: usize, cells: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(cells.len() == self.mode.carriers(), || {
            format!("{} cells for the {} carriers of a {} symbol", cells.len(), self.mode.carriers(), self.mode.name())
        })?;
        Ok(self.data[l % 4].iter().map(|&k| cells[k]).collect())
    }

    /// The statistic of the TPS sign of a received symbol: $\sum_{k} \mathrm{Re}\{y_{k}
    /// h_{k}^{*}\} \cdot 2(1/2 - w_{k})$ over the TPS carriers, of the sign of $d_{l}$
    /// (`channel` the gains of the carriers, 1 if `None`).
    ///
    /// # Errors
    /// [`Error::Shape`] unless `cells` (and `channel`) have $K$ values.
    pub fn tps_statistic(&self, cells: &[C64], channel: Option<&[C64]>) -> Result<f64> {
        let k_total = self.mode.carriers();
        ensure_shape(cells.len() == k_total && channel.is_none_or(|h| h.len() == k_total), || {
            format!("the cells and the channel of a {} symbol have {k_total} values", self.mode.name())
        })?;
        Ok(self
            .mode
            .tps_carriers()
            .iter()
            .map(|&k| {
                let k = usize::from(k);
                let h = channel.map_or(C64::new(1.0, 0.0), |h| h[k]);
                (cells[k] * h.conj()).re * self.reference[k]
            })
            .sum())
    }

    /// The OFDM modulator of the mode with the guard interval `guard`: $K$ active carriers
    /// out of the FFT of $T_{U}/T$, one sample per $T$.
    ///
    /// # Errors
    /// Never: the sizes of the mode are valid.
    pub fn ofdm(&self, guard: Guard) -> Result<Ofdm> {
        let base = Ofdm::new(self.mode.fft_len(), self.mode.carriers())?;
        Ok(Ofdm { oversampling: 1, n_cp: guard.samples(self.mode), ..base })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_symbol_has_the_data_cells_of_sec_4_5_1() {
        for mode in [Mode::K2, Mode::K8] {
            let f = DvbtFrame::new(mode);
            for l in 0..4 {
                assert_eq!(f.data_carriers(l).len(), mode.data_cells(), "{} l = {l}", mode.name());
            }
            // no TPS carrier is a pilot (Table 8 against Table 7 and the scattered pattern)
            for &k in mode.tps_carriers() {
                for l in 0..4 {
                    assert_eq!(f.cell(l, usize::from(k)), Cell::Tps);
                }
            }
        }
    }

    #[test]
    fn the_pilot_prbs_starts_as_figure_10() {
        // "PRBS sequence starts: 1111111111100..."
        assert_eq!(&pilot_prbs(13), &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
        let p = pilot_prbs(2 * 2047);
        assert_eq!(&p[..2047], &p[2047..], "period 2^11 - 1");
    }

    #[test]
    fn the_tables_have_the_lengths_of_the_standard() {
        assert_eq!((Mode::K2.continual_pilots().len(), Mode::K8.continual_pilots().len()), (45, 177));
        assert_eq!((Mode::K2.tps_carriers().len(), Mode::K8.tps_carriers().len()), (17, 68));
        assert!(CONTINUAL_PILOTS.windows(2).all(|w| w[0] < w[1]));
        assert!(TPS_CARRIERS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!((CONTINUAL_PILOTS[44], CONTINUAL_PILOTS[176]), (1704, 6816), "the edges are pilots");
    }

    #[test]
    fn the_boost_gives_the_mean_power_of_the_carriers() {
        // 176 boosted cells of the 1705 in the 2K mode, 701 of the 6817 in the 8K mode
        let p2 = (1512.0 + 17.0 + 176.0 * 16.0 / 9.0) / 1705.0;
        let p8 = (6048.0 + 68.0 + 701.0 * 16.0 / 9.0) / 6817.0;
        assert!((DvbtFrame::new(Mode::K2).mean_carrier_power() - p2).abs() < 1e-12);
        assert!((DvbtFrame::new(Mode::K8).mean_carrier_power() - p8).abs() < 1e-12);
    }

    #[test]
    fn a_symbol_places_its_cells() {
        let f = DvbtFrame::new(Mode::K2);
        let data: Vec<C64> = (0..1512).map(|i| C64::new(i as f64, -1.0)).collect();
        let s = f.symbol(1, &data, -1.0).unwrap();
        assert_eq!(f.data_of(1, &s).unwrap(), data);
        assert_eq!(s[0], C64::new(-PILOT_BOOST, 0.0), "carrier 0, a continual pilot, w = 1: 2(1/2 - 1) 4/3");
        assert_eq!(s[34], C64::new(-f.reference(34), 0.0), "the first TPS carrier");
        let bits: Vec<u8> = [0, 1, 1, 0].iter().chain(&[0; 64]).copied().collect();
        assert_eq!(&tps_signs(&bits).unwrap()[..5], &[1.0, -1.0, 1.0, 1.0, 1.0]);
        assert!(f.tps_statistic(&s, None).unwrap() < 0.0);
    }

    #[test]
    fn the_modes_and_guards_parse() {
        assert_eq!("8k".parse::<Mode>().unwrap(), Mode::K8);
        assert_eq!("1/16".parse::<Guard>().unwrap().samples(Mode::K2), 128);
        assert!("4K".parse::<Mode>().is_err());
        assert!((elementary_period(8e6) - 7.0 / 64.0 * 1e-6).abs() < 1e-18);
    }
}
