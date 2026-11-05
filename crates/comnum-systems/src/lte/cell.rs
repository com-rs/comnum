//! The frames that an LTE cell sends, as far as a terminal reads them before it knows
//! anything of the cell: the PSS and the SSS of slots 0 and 10, the CRS of every port in
//! every slot, and the PBCH in slot 1 (TS 36.211, Sec. 6.6, 6.10.1, 6.11). The other
//! resource elements carry QPSK symbols at random on port 0, or nothing; the six central
//! resource blocks of the PBCH symbols and of the synchronisation symbols carry nothing
//! else, the elements reserved for the CRS of four ports included.
//!
//! The PBCH of the frame of number $n_{f}$ (the 8 bits of the MIB being $\lfloor n_{f}/4
//! \rfloor \bmod 256$) is the quarter $n_{f} \bmod 4$ of the 1920 bits
//! of [`super::bch::encode`], scrambled by the Gold sequence of $c_{\mathrm{init}} =
//! N_{\mathrm{ID}}^{\mathrm{cell}}$ started every 40 ms (Sec. 6.6.1), mapped to QPSK (Sec.
//! 7.1.2, [`crate::nr::modulation::constellation`]), sent by transmit diversity on two or
//! four ports ([`comnum_waveform::ofdm::lte::sfbc`]) and placed on the elements of
//! [`comnum_waveform::ofdm::lte::pbch_elements`]. The PSS and the SSS go on port 0.
//!
//! # Examples
//! ```
//! use comnum_systems::lte::cell::Downlink;
//! use comnum_systems::lte::mib::{Mib, PhichDuration, PhichResource};
//! let mib = Mib { n_rb: 6, phich_duration: PhichDuration::Normal, phich_resource: PhichResource::One, sfn_msb: 10, rest: [0; 10] };
//! let cell = Downlink { n_cell_id: 77, mib, ports: 2 };
//! let grids = cell.frame_grids(41, None)?;            // frame 41: the quarter 1 of the PBCH
//! assert_eq!((grids.len(), grids[0].len()), (2, 140 * 72));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 36.211 V18.0.2, Sec. 6.6, 6.10.1, 6.11, 7.1.2, 7.2.

use super::bch::{self, FRAME_BITS};
use super::mib::Mib;
use crate::nr::modulation::constellation;
use comnum_core::rng::Rng;
use comnum_core::{Result, ensure};
use comnum_math::C64;
use comnum_math::sequences::gold_3gpp;
use comnum_waveform::ofdm::lte::{
    self, CENTRAL_SUBCARRIERS, Carrier, PSS_SYMBOL, SLOTS_PER_FRAME, SSS_SYMBOL, SYMBOLS_PER_SLOT, sfbc,
};

/// Symbols of a frame.
pub const SYMBOLS_PER_FRAME: usize = SLOTS_PER_FRAME * SYMBOLS_PER_SLOT;

/// A cell and the master information block it broadcasts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Downlink {
    /// The cell identity $N_{\mathrm{ID}}^{\mathrm{cell}}$.
    pub n_cell_id: u32,
    /// The master information block; its bandwidth is the carrier's, its frame number
    /// is replaced by that of every frame sent.
    pub mib: Mib,
    /// The antenna ports of the CRS and of the PBCH, 1, 2 or 4.
    pub ports: usize,
}

impl Downlink {
    /// The QPSK symbols of the PBCH in the frame `sfn` (240), before transmit diversity.
    ///
    /// # Errors
    /// The errors of [`bch::encode`].
    pub fn pbch_symbols(&self, sfn: usize) -> Result<Vec<C64>> {
        let mib = Mib { sfn_msb: (sfn / 4 % 256) as u8, ..self.mib };
        let e = bch::encode(&mib.to_bits(), self.ports)?;
        let c = gold_3gpp(self.n_cell_id, bch::BCH_BITS);
        let q = (sfn % 4) * FRAME_BITS;
        let points = constellation(2)?;
        Ok((q..q + FRAME_BITS)
            .step_by(2)
            .map(|i| points.points()[usize::from(2 * (e[i] ^ c[i]) + (e[i + 1] ^ c[i + 1]))])
            .collect())
    }

    /// The grid of every port over the frame `sfn` (`grids[p][l * K + k]`, 140 symbols of
    /// $K = 12 N_{\mathrm{RB}}$ subcarriers): the signals of the module, and QPSK symbols
    /// drawn from `data` on the free elements of port 0 (none without).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a cell identity above 503, a number of ports
    /// other than 1, 2 or 4, or a bandwidth outside 6 to 110 resource blocks.
    pub fn frame_grids(&self, sfn: usize, data: Option<&mut Rng>) -> Result<Vec<Vec<C64>>> {
        ensure(matches!(self.ports, 1 | 2 | 4), || format!("a cell has 1, 2 or 4 antenna ports, got {}", self.ports))?;
        let n_rb = self.mib.n_rb;
        let k = lte::SUBCARRIERS_PER_RB * n_rb;
        let zero = C64::new(0.0, 0.0);
        let mut grids = vec![vec![zero; SYMBOLS_PER_FRAME * k]; self.ports];
        // the elements taken: the CRS of four ports, the centre of the PBCH and sync symbols
        let mut taken = vec![false; SYMBOLS_PER_FRAME * k];
        let central = 6 * n_rb - 36..6 * n_rb + 36;
        for ns in 0..SLOTS_PER_FRAME {
            for port in 0..4 {
                for &l in lte::crs_symbols(port)? {
                    let row = (ns * SYMBOLS_PER_SLOT + l) * k;
                    let sub = lte::crs_subcarriers(self.n_cell_id, port, ns, l, n_rb)?;
                    let values = lte::crs(self.n_cell_id, ns, l, n_rb)?;
                    for (&kk, &v) in sub.iter().zip(&values) {
                        taken[row + kk] = true;
                        if port < self.ports {
                            grids[port][row + kk] = v;
                        }
                    }
                }
            }
        }
        let central_rows =
            [0, 10].iter().flat_map(|&ns| [SSS_SYMBOL, PSS_SYMBOL].map(|l| ns * SYMBOLS_PER_SLOT + l)).chain(7..11);
        for row in central_rows {
            for kk in central.clone() {
                taken[row * k + kk] = true;
            }
        }
        // the synchronisation signals, on port 0
        let first = lte::sync_first_subcarrier(n_rb);
        let n_id2 = self.n_cell_id % 3;
        let pss = lte::pss(n_id2)?;
        for (half, ns) in [(false, 0), (true, 10)] {
            let sss = lte::sss(self.n_cell_id / 3, n_id2, half)?;
            let (pss_row, sss_row) =
                ((ns * SYMBOLS_PER_SLOT + PSS_SYMBOL) * k, (ns * SYMBOLS_PER_SLOT + SSS_SYMBOL) * k);
            for n in 0..lte::SYNC_LENGTH {
                grids[0][pss_row + first + n] = pss[n];
                grids[0][sss_row + first + n] = C64::new(sss[n], 0.0);
            }
        }
        // the PBCH, in slot 1
        let d = self.pbch_symbols(sfn)?;
        let elements = lte::pbch_elements(self.n_cell_id, n_rb)?;
        let precoded: Vec<Vec<C64>> = if self.ports == 1 {
            vec![d]
        } else {
            let x = sfbc(self.ports)?.encode(&d)?;
            x.rows().into_iter().map(|r| r.to_vec()).collect()
        };
        for (i, &(kk, l)) in elements.iter().enumerate() {
            for (port, y) in precoded.iter().enumerate() {
                grids[port][(SYMBOLS_PER_SLOT + l) * k + kk] = y[i];
            }
        }
        if let Some(rng) = data {
            let points = constellation(2)?;
            for (g, t) in grids[0].iter_mut().zip(&taken) {
                if !t {
                    *g = points.points()[usize::from(2 * rng.bit() + rng.bit())];
                }
            }
        }
        Ok(grids)
    }

    /// The samples of every port over the frame `sfn` at $N_{\mathrm{FFT}} \cdot 15$ kHz
    /// ([`Carrier::modulate`] of [`Downlink::frame_grids`]).
    ///
    /// # Errors
    /// The errors of [`Downlink::frame_grids`] and of [`Carrier::new`].
    pub fn frame_samples(&self, sfn: usize, n_fft: usize, data: Option<&mut Rng>) -> Result<Vec<Vec<C64>>> {
        let carrier = Carrier::new(self.mib.n_rb, n_fft)?;
        self.frame_grids(sfn, data)?.iter().map(|g| carrier.modulate(g)).collect()
    }
}

/// The central subcarriers of a carrier of `n_rb` resource blocks, as an index range.
pub fn central_subcarriers(n_rb: usize) -> std::ops::Range<usize> {
    6 * n_rb - CENTRAL_SUBCARRIERS / 2..6 * n_rb + CENTRAL_SUBCARRIERS / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lte::mib::{PhichDuration, PhichResource};

    fn mib(n_rb: usize) -> Mib {
        Mib {
            n_rb,
            phich_duration: PhichDuration::Normal,
            phich_resource: PhichResource::One,
            sfn_msb: 3,
            rest: [0; 10],
        }
    }

    #[test]
    fn the_frame_holds_the_signals_where_the_standard_puts_them() {
        let cell = Downlink { n_cell_id: 152, mib: mib(25), ports: 2 };
        let mut rng = Rng::new(1);
        let g = cell.frame_grids(12, Some(&mut rng)).unwrap();
        let k = 300;
        // the PSS of N_ID2 = 152 mod 3 = 2 on symbol 6 of slot 0, subcarriers 119..181
        let pss = lte::pss(2).unwrap();
        assert_eq!(g[0][6 * k + 119..6 * k + 181], pss[..]);
        assert_eq!(g[0][6 * k + 118], C64::new(0.0, 0.0));
        // port 1 sends nothing on the CRS of port 0
        let sub0 = lte::crs_subcarriers(152, 0, 3, 4, 25).unwrap();
        assert!(sub0.iter().all(|&kk| g[1][(3 * 7 + 4) * k + kk] == C64::new(0.0, 0.0)));
        // every element of port 0 carries something outside the reserved ones
        let used = g[0].iter().filter(|v| v.norm() > 0.0).count();
        assert!(used > 140 * k * 9 / 10, "{used}");
        assert_eq!(central_subcarriers(25), 114..186);
        assert!(Downlink { ports: 3, ..cell }.frame_grids(0, None).is_err());
        let x = cell.frame_samples(12, 512, None).unwrap();
        assert_eq!(x[0].len(), 76_800); // 10 ms at 7.68 MS/s
    }
}
