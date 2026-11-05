//! The downlink signals of LTE that a terminal reads before anything else: the
//! synchronisation signals, the cell-specific reference signals, the resource elements
//! of the broadcast channel, the OFDM grid with its unused DC subcarrier, and the
//! space-frequency block code of transmit diversity (frame structure type 1, FDD, normal
//! cyclic prefix).
//!
//! # Grid and numerology
//!
//! A downlink carrier of $N_{\mathrm{RB}}$ resource blocks has $K = 12 N_{\mathrm{RB}}$
//! subcarriers of 15 kHz, numbered $k = 0, \dots, K - 1$ from the lowest; the subcarrier
//! $k$ sits at $(k - K/2)\Delta f$ for $k < K/2$ and $(k - K/2 + 1)\Delta f$ above, the DC
//! subcarrier being left empty (TS 36.211, Sec. 6.12). A slot holds 7 OFDM symbols, with
//! the prefixes of 160 then 144 samples at 30.72 MS/s, the numerology $\mu = 0$ of
//! [`super::nr::Numerology`]; [`Carrier`] modulates the grid with the prefixes of
//! [`super::nr::Carrier`], the empty DC subcarrier inserted. The six central resource
//! blocks, $k = 6 N_{\mathrm{RB}} - 36, \dots, 6 N_{\mathrm{RB}} + 35$, carry the same
//! signals in every bandwidth: a receiver reads them at 1.92 MS/s with the grid of
//! $N_{\mathrm{RB}} = 6$ and an FFT of 128.
//!
//! # Synchronisation signals (Sec. 6.11)
//!
//! The 504 cell identities are $N_{\mathrm{ID}}^{\mathrm{cell}} = 3 N_{\mathrm{ID}}^{(1)} +
//! N_{\mathrm{ID}}^{(2)}$, $N_{\mathrm{ID}}^{(1)} < 168$, $N_{\mathrm{ID}}^{(2)} < 3$. Both
//! signals occupy the 62 subcarriers $k = n - 31 + 6 N_{\mathrm{RB}}$, $n = 0, \dots, 61$
//! (five empty ones on each side), in slots 0 and 10: the PSS in the last symbol of the
//! slot ($l = 6$), the SSS in the symbol before ($l = 5$).
//!
//! - PSS ([`pss`]): the Zadoff–Chu sequence of length 63 and root $u = 25, 29, 34$ for
//!   $N_{\mathrm{ID}}^{(2)} = 0, 1, 2$, its middle element punctured,
//!   $d_{u}(n) = e^{-j\pi u n (n + 1)/63}$ for $n \le 30$ and $e^{-j\pi u (n + 1)(n + 2)/63}$
//!   for $31 \le n \le 61$ (eq. of Sec. 6.11.1.1, Table 6.11.1.1-1);
//! - SSS ([`sss`]): the interleaving of two m-sequences of length 31,
//!   $d(2n) = s_{0}^{(m_{0})}(n) c_{0}(n)$ and $d(2n + 1) = s_{1}^{(m_{1})}(n) c_{1}(n)
//!   z_{1}^{(m_{0})}(n)$ in subframe 0, the roles of $m_{0}$ and $m_{1}$ swapped in subframe
//!   5, so that one SSS tells the half frame; $m_{0} = m' \bmod 31$, $m_{1} = (m_{0} +
//!   \lfloor m'/31 \rfloor + 1) \bmod 31$, $m' = N_{\mathrm{ID}}^{(1)} + q(q + 1)/2$, $q =
//!   \lfloor (N_{\mathrm{ID}}^{(1)} + q'(q' + 1)/2)/30 \rfloor$, $q' = \lfloor
//!   N_{\mathrm{ID}}^{(1)}/30 \rfloor$ (Table 6.11.2.1-1); $\tilde{s}$, $\tilde{c}$ and
//!   $\tilde{z}$ are the m-sequences $x(i + 5) = x(i + 2) + x(i)$, $x(i + 3) + x(i)$ and
//!   $x(i + 4) + x(i + 2) + x(i + 1) + x(i)$, started at $\[0\,0\,0\,0\,1\]$, as $1 - 2x$;
//!   $c_{0}$, $c_{1}$ are $\tilde{c}$ shifted by $N_{\mathrm{ID}}^{(2)}$ and
//!   $N_{\mathrm{ID}}^{(2)} + 3$, $z_{1}^{(m)}$ is $\tilde{z}$ shifted by $m \bmod 8$.
//!
//! # Cell-specific reference signals (Sec. 6.10.1)
//!
//! In the symbol $l$ of the slot $n_{s}$, the CRS is the QPSK of the Gold sequence of
//! 3GPP ([`comnum_math::sequences::gold_3gpp`]) with
//! $c_{\mathrm{init}} = 2^{10}(7(n_{s} + 1) + l + 1)(2N_{\mathrm{ID}}^{\mathrm{cell}} + 1) +
//! 2N_{\mathrm{ID}}^{\mathrm{cell}} + 1$,
//! $r(m) = \[(1 - 2c(2m)) + j(1 - 2c(2m + 1))\]/\sqrt{2}$, of which the carrier uses
//! $r(m + 110 - N_{\mathrm{RB}})$, $m < 2 N_{\mathrm{RB}}$ ([`crs`]), on the subcarriers
//! $k = 6m + (v + v_{\mathrm{shift}}) \bmod 6$, $v_{\mathrm{shift}} =
//! N_{\mathrm{ID}}^{\mathrm{cell}} \bmod 6$ ([`crs_subcarriers`]): ports 0 and 1 in the
//! symbols $l = 0$ and 4 of every slot, $v = 0$ and 3 (port 0) or 3 and 0 (port 1);
//! ports 2 and 3 in $l = 1$, $v = 3(n_{s} \bmod 2)$ and $3 + 3(n_{s} \bmod 2)$. A port
//! sends nothing on the CRS elements of the others.
//!
//! # Broadcast channel (Sec. 6.6)
//!
//! The PBCH occupies the 72 central subcarriers of the first four symbols of slot 1, but
//! for the elements of the CRS of four ports, whatever the number of ports: those $k$
//! with $k \equiv v_{\mathrm{shift}} \pmod 3$ in $l = 0$ and 1; 240 elements per frame,
//! frequency first ([`pbch_elements`]). With two or four ports, the QPSK symbols are
//! sent by transmit diversity ([`sfbc`], Sec. 6.3.3.3 and 6.3.4.3): the pairs
//! $(x_{0}, x_{1}) = (d(2i), d(2i + 1))$ go on two consecutive elements as
//! $\frac{1}{\sqrt{2}}\[x_{0}, -x_{1}^{*}\]$ and $\frac{1}{\sqrt{2}}\[x_{1}, x_{0}^{*}\]$ on ports
//! 0 and 1 (Alamouti's code across subcarriers); with four ports, the pairs alternate
//! between the ports 0, 2 and 1, 3.
//!
//! # Examples
//! ```
//! use comnum_waveform::ofdm::lte::{pss, sss, pbch_elements, Carrier};
//! assert_eq!(pss(1)?.len(), 62);
//! assert_eq!(sss(100, 1, false)?.len(), 62);
//! assert_eq!(pbch_elements(301, 6)?.len(), 240);
//! let c = Carrier::new(6, 128)?;                    // the six central RBs, 1.92 MS/s
//! assert_eq!(c.sample_rate(), 1.92e6);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 36.211 V18.0.2, *E-UTRA; Physical channels and modulation*, Sec. 6.3.3.3,
//!    6.3.4.3, 6.6, 6.10.1, 6.11, 6.12, 7.1.2, 7.2.
//! 2. D. Astély et al., "LTE: the evolution of mobile broadband," *IEEE Commun. Mag.*
//!    47(4), 44–51, 2009.

use super::nr;
use comnum_core::{Error, Result, ensure, ensure_shape};
use comnum_math::C64;
use comnum_math::sequences::{gold_3gpp, lfsr, zadoff_chu};
use comnum_mimo::stbc::Ostbc;
use std::f64::consts::FRAC_1_SQRT_2;

/// The cell identities $N_{\mathrm{ID}}^{\mathrm{cell}} < 504$.
pub const CELL_IDS: u32 = 504;
/// The largest downlink bandwidth, $N_{\mathrm{RB}}^{\max,\mathrm{DL}} = 110$ resource blocks.
pub const MAX_RB: usize = 110;
/// Subcarriers of a resource block.
pub const SUBCARRIERS_PER_RB: usize = nr::SUBCARRIERS_PER_RB;
/// OFDM symbols of a slot (normal cyclic prefix).
pub const SYMBOLS_PER_SLOT: usize = 7;
/// Slots of a 10 ms frame.
pub const SLOTS_PER_FRAME: usize = 20;
/// Length of the PSS and of the SSS.
pub const SYNC_LENGTH: usize = 62;
/// The symbol of the PSS in slots 0 and 10.
pub const PSS_SYMBOL: usize = 6;
/// The symbol of the SSS in slots 0 and 10.
pub const SSS_SYMBOL: usize = 5;
/// The subcarriers of the six central resource blocks (PSS, SSS, PBCH).
pub const CENTRAL_SUBCARRIERS: usize = 72;
/// The PBCH elements of a frame.
pub const PBCH_ELEMENTS: usize = 240;

/// The roots of the PSS, Table 6.11.1.1-1.
const PSS_ROOTS: [u32; 3] = [25, 29, 34];

/// The resource blocks of the six bandwidths of LTE and their FFT at 15 kHz
/// ($N_{\mathrm{FFT}} \Delta f$ the sampling rate: 1.92 to 30.72 MS/s).
pub const BANDWIDTHS: [(usize, usize); 6] = [(6, 128), (15, 256), (25, 512), (50, 1024), (75, 1536), (100, 2048)];

/// The usual FFT size of a carrier of `n_rb` resource blocks ([`BANDWIDTHS`]).
///
/// # Errors
/// [`Error::Config`] for another number of resource blocks.
pub fn fft_len(n_rb: usize) -> Result<usize> {
    BANDWIDTHS
        .iter()
        .find(|(n, _)| *n == n_rb)
        .map(|(_, f)| *f)
        .ok_or_else(|| Error::Config(format!("LTE carriers have 6, 15, 25, 50, 75 or 100 resource blocks, got {n_rb}")))
}

fn check_cell(n_cell_id: u32) -> Result<()> {
    ensure(n_cell_id < CELL_IDS, || format!("the cell identities are 0 to 503, got {n_cell_id}"))
}

fn check_rb(n_rb: usize) -> Result<()> {
    ensure((6..=MAX_RB).contains(&n_rb), || format!("a downlink carrier has 6 to 110 resource blocks, got {n_rb}"))
}

/// The PSS of $N_{\mathrm{ID}}^{(2)}$: 62 values of unit magnitude, on the subcarriers
/// $k = n - 31 + 6 N_{\mathrm{RB}}$ (see the module).
///
/// # Errors
/// [`Error::Config`] for $N_{\mathrm{ID}}^{(2)} > 2$.
pub fn pss(n_id2: u32) -> Result<Vec<C64>> {
    ensure(n_id2 < 3, || format!("N_ID2 is 0, 1 or 2, got {n_id2}"))?;
    let mut x = zadoff_chu(PSS_ROOTS[n_id2 as usize], 63)?;
    x.remove(31);
    Ok(x)
}

/// The indices $(m_{0}, m_{1})$ of the SSS of the group $N_{\mathrm{ID}}^{(1)}$ (Table
/// 6.11.2.1-1).
///
/// # Errors
/// [`Error::Config`] for $N_{\mathrm{ID}}^{(1)} > 167$.
pub fn sss_indices(n_id1: u32) -> Result<(usize, usize)> {
    ensure(n_id1 < 168, || format!("N_ID1 is 0 to 167, got {n_id1}"))?;
    let n = n_id1 as usize;
    let qp = n / 30;
    let q = (n + qp * (qp + 1) / 2) / 30;
    let mp = n + q * (q + 1) / 2;
    let m0 = mp % 31;
    Ok((m0, (m0 + mp / 31 + 1) % 31))
}

/// The SSS of $(N_{\mathrm{ID}}^{(1)}, N_{\mathrm{ID}}^{(2)})$: 62 values $\pm 1$, those of
/// subframe 0, or of subframe 5 when `second_half` (see the module).
///
/// # Errors
/// [`Error::Config`] for $N_{\mathrm{ID}}^{(1)} > 167$ or $N_{\mathrm{ID}}^{(2)} > 2$.
pub fn sss(n_id1: u32, n_id2: u32, second_half: bool) -> Result<Vec<f64>> {
    ensure(n_id2 < 3, || format!("N_ID2 is 0, 1 or 2, got {n_id2}"))?;
    let (m0, m1) = sss_indices(n_id1)?;
    let seq = |taps: &[usize]| -> Vec<f64> {
        lfsr(taps, 5, &[0, 0, 0, 0, 1], 31).iter().map(|&b| 1.0 - 2.0 * f64::from(b)).collect()
    };
    let (s, c, z) = (seq(&[0, 2]), seq(&[0, 3]), seq(&[0, 1, 2, 4]));
    let n2 = n_id2 as usize;
    let (a, b) = if second_half { (m1, m0) } else { (m0, m1) };
    let mut d = vec![0.0; SYNC_LENGTH];
    for n in 0..31 {
        d[2 * n] = s[(n + a) % 31] * c[(n + n2) % 31];
        d[2 * n + 1] = s[(n + b) % 31] * c[(n + n2 + 3) % 31] * z[(n + a % 8) % 31];
    }
    Ok(d)
}

/// The CRS of the symbol `l` of the slot `ns` on a carrier of `n_rb` resource blocks:
/// the $2 N_{\mathrm{RB}}$ values $r(m + 110 - N_{\mathrm{RB}})$, in the order of
/// [`crs_subcarriers`] (see the module).
///
/// # Errors
/// [`Error::Config`] for a cell identity above 503, a slot above 19, a symbol above 6 or
/// a number of resource blocks outside 6 to 110.
pub fn crs(n_cell_id: u32, ns: usize, l: usize, n_rb: usize) -> Result<Vec<C64>> {
    check_cell(n_cell_id)?;
    check_rb(n_rb)?;
    ensure(ns < SLOTS_PER_FRAME && l < SYMBOLS_PER_SLOT, || format!("slot {ns} and symbol {l}: 0 to 19 and 0 to 6"))?;
    let id = n_cell_id as u64;
    let c_init = (1024 * (7 * (ns as u64 + 1) + l as u64 + 1) * (2 * id + 1) + 2 * id + 1) % (1 << 31);
    let start = MAX_RB - n_rb;
    let c = gold_3gpp(c_init as u32, 4 * MAX_RB);
    Ok((start..start + 2 * n_rb)
        .map(|m| C64::new(1.0 - 2.0 * f64::from(c[2 * m]), 1.0 - 2.0 * f64::from(c[2 * m + 1])) * FRAC_1_SQRT_2)
        .collect())
}

/// The symbols of a slot that carry the CRS of `port`: 0 and 4 for ports 0 and 1, 1 for
/// ports 2 and 3.
///
/// # Errors
/// [`Error::Config`] for a port above 3.
pub fn crs_symbols(port: usize) -> Result<&'static [usize]> {
    match port {
        0 | 1 => Ok(&[0, 4]),
        2 | 3 => Ok(&[1]),
        _ => Err(Error::Config(format!("the CRS are sent on ports 0 to 3, got {port}"))),
    }
}

/// The $2 N_{\mathrm{RB}}$ subcarriers $k = 6m + (v + v_{\mathrm{shift}}) \bmod 6$ of the CRS
/// of `port` in the symbol `l` of the slot `ns` (see the module).
///
/// # Errors
/// [`Error::Config`] for a cell identity above 503, a port above 3, a symbol without the
/// CRS of the port ([`crs_symbols`]) or a number of resource blocks outside 6 to 110.
pub fn crs_subcarriers(n_cell_id: u32, port: usize, ns: usize, l: usize, n_rb: usize) -> Result<Vec<usize>> {
    check_cell(n_cell_id)?;
    check_rb(n_rb)?;
    ensure(crs_symbols(port)?.contains(&l), || format!("port {port} has no CRS in symbol {l}"))?;
    let v = match (port, l) {
        (0, 0) | (1, 4) => 0,
        (0 | 1, _) => 3,
        (2, _) => 3 * (ns % 2),
        _ => 3 + 3 * (ns % 2),
    };
    let shift = (v + n_cell_id as usize % 6) % 6;
    Ok((0..2 * n_rb).map(|m| 6 * m + shift).collect())
}

/// The first of the 62 subcarriers of the PSS and of the SSS, $6 N_{\mathrm{RB}} - 31$.
pub fn sync_first_subcarrier(n_rb: usize) -> usize {
    6 * n_rb - 31
}

/// The 240 elements $(k, l)$ of the PBCH in the first four symbols of slot 1, in the
/// order of the mapping (frequency first; see the module).
///
/// # Errors
/// [`Error::Config`] for a cell identity above 503 or a number of resource blocks outside
/// 6 to 110.
pub fn pbch_elements(n_cell_id: u32, n_rb: usize) -> Result<Vec<(usize, usize)>> {
    check_cell(n_cell_id)?;
    check_rb(n_rb)?;
    let first = 6 * n_rb - 36;
    let v_shift = n_cell_id as usize % 3;
    Ok((0..4)
        .flat_map(|l| (first..first + CENTRAL_SUBCARRIERS).map(move |k| (k, l)))
        .filter(|&(k, l)| l >= 2 || k % 3 != v_shift)
        .collect())
}

/// The space-frequency block code of transmit diversity on `ports` = 2 or 4 antenna ports
/// (see the module), as an orthogonal code whose channel uses are consecutive resource
/// elements: [`Ostbc::encode`] gives the precoded symbols of every port,
/// [`Ostbc::decode_blocks`] combines them.
///
/// # Errors
/// [`Error::Config`] for another number of ports.
pub fn sfbc(ports: usize) -> Result<Ostbc> {
    match ports {
        2 => Ostbc::from_generator("SFBC, 2 ports", 2, 2, 2, |s| {
            ndarray::array![[s[0], -s[1].conj()], [s[1], s[0].conj()]]
        }),
        4 => {
            let z = C64::new(0.0, 0.0);
            Ostbc::from_generator("SFBC-FSTD, 4 ports", 4, 4, 4, move |s| {
                ndarray::array![
                    [s[0], z, -s[1].conj(), z],
                    [s[1], z, s[0].conj(), z],
                    [z, s[2], z, -s[3].conj()],
                    [z, s[3], z, s[2].conj()]
                ]
            })
        }
        _ => Err(Error::Config(format!("transmit diversity uses 2 or 4 ports, got {ports}"))),
    }
}

/// A downlink carrier: the grid of $K = 12 N_{\mathrm{RB}}$ subcarriers and its OFDM
/// modulator at $N_{\mathrm{FFT}} \cdot 15$ kHz, with the empty DC subcarrier and the
/// normal prefixes of every symbol of the subframe (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carrier {
    /// Resource blocks $N_{\mathrm{RB}}$.
    pub n_rb: usize,
    /// FFT size $N_{\mathrm{FFT}}$ (a multiple of 128, above $K$).
    pub n_fft: usize,
    /// The position in the subframe (0 to 13) of the first OFDM symbol of the grid.
    pub first_symbol: usize,
}

impl Carrier {
    /// The carrier of `n_rb` resource blocks and an FFT of `n_fft`, its grid starting
    /// with the first symbol of a subframe.
    ///
    /// # Errors
    /// [`Error::Config`] for a number of resource blocks outside 6 to 110 or an FFT size
    /// that is not a multiple of 128 above $K + 1$.
    pub fn new(n_rb: usize, n_fft: usize) -> Result<Carrier> {
        let c = Carrier { n_rb, n_fft, first_symbol: 0 };
        c.nr()?;
        Ok(c)
    }

    /// Subcarriers $K$ of the grid.
    pub fn n_subcarriers(&self) -> usize {
        SUBCARRIERS_PER_RB * self.n_rb
    }

    /// The carrier of NR of the same numerology whose grid holds the $K$ subcarriers, the
    /// DC one and an unused one below: $K + 2$ subcarriers, the LTE subcarrier $k$ at
    /// $k + 1$ below the DC and $k + 2$ above.
    fn nr(&self) -> Result<nr::Carrier> {
        check_rb(self.n_rb)?;
        ensure(self.first_symbol < 2 * SYMBOLS_PER_SLOT, || {
            format!("the first symbol is 0 to 13 in the subframe, got {}", self.first_symbol)
        })?;
        let c = nr::Carrier {
            numerology: nr::Numerology::new(0, false)?,
            n_subcarriers: self.n_subcarriers() + 2,
            n_fft: self.n_fft,
            first_symbol: self.first_symbol,
        };
        c.check()?;
        Ok(c)
    }

    /// The position in the grid of [`Carrier::nr`] of the subcarrier `k`.
    fn position(&self, k: usize) -> usize {
        if k < self.n_subcarriers() / 2 { k + 1 } else { k + 2 }
    }

    /// OFDM-modulate a grid (`grid[l * K + k]` for subcarrier `k` of symbol `l`), every
    /// symbol with the prefix of its position in the subframe.
    ///
    /// # Errors
    /// The errors of [`Carrier::new`]; [`Error::Shape`] for a grid that is not a whole
    /// number of symbols.
    pub fn modulate(&self, grid: &[C64]) -> Result<Vec<C64>> {
        let c = self.nr()?;
        let k = self.n_subcarriers();
        ensure_shape(grid.len() % k == 0, || format!("{} values is not a whole number of symbols of {k}", grid.len()))?;
        let mut wide = vec![C64::new(0.0, 0.0); grid.len() / k * (k + 2)];
        for (sym, row) in grid.chunks_exact(k).zip(wide.chunks_exact_mut(k + 2)) {
            for (i, v) in sym.iter().enumerate() {
                row[self.position(i)] = *v;
            }
        }
        c.modulate(&wide)
    }

    /// The grid of the samples of whole OFDM symbols, their prefixes removed: the
    /// inverse of [`Carrier::modulate`].
    ///
    /// # Errors
    /// The errors of [`Carrier::new`]; [`Error::Shape`] if the samples do not end with a
    /// whole symbol.
    pub fn demodulate(&self, samples: &[C64]) -> Result<Vec<C64>> {
        let wide = self.nr()?.demodulate(samples)?;
        let k = self.n_subcarriers();
        Ok(wide.chunks_exact(k + 2).flat_map(|row| (0..k).map(move |i| row[self.position(i)])).collect())
    }

    /// The prefix of the symbol `l` of the subframe (0 to 13), in samples.
    pub fn prefix(&self, l: usize) -> usize {
        // 160 then 144 samples at N_FFT = 2048 (Table 6.12-1)
        if l % SYMBOLS_PER_SLOT == 0 { 160 * self.n_fft / 2048 } else { 144 * self.n_fft / 2048 }
    }

    /// Samples of a slot, $15360 N_{\mathrm{FFT}}/2048$ (0.5 ms).
    pub fn slot_len(&self) -> usize {
        15_360 * self.n_fft / 2048
    }

    /// The sampling rate $N_{\mathrm{FFT}} \cdot 15$ kHz (Hz).
    pub fn sample_rate(&self) -> f64 {
        self.n_fft as f64 * 15e3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_math::fft;

    #[test]
    fn the_pss_are_the_punctured_zadoff_chu_sequences() {
        for n2 in 0..3 {
            let d = pss(n2).unwrap();
            assert!(d.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
            // the elements 31.. are those of the sequence after the punctured one
            let u = PSS_ROOTS[n2 as usize] as f64;
            let n = 40.0;
            let want = C64::from_polar(1.0, -std::f64::consts::PI * u * (n + 1.0) * (n + 2.0) / 63.0);
            assert!((d[40] - want).norm() < 1e-9);
        }
        // the PSS of the three identities are nearly orthogonal
        let (a, b) = (pss(0).unwrap(), pss(1).unwrap());
        let r: C64 = a.iter().zip(&b).map(|(x, y)| x * y.conj()).sum();
        assert!(r.norm() / 62.0 < 0.2, "{}", r.norm() / 62.0);
        assert!(pss(3).is_err());
    }

    /// Table 6.11.2.1-1 of TS 36.211 V18.0.2, the 168 rows $(m_{0}, m_{1})$ verbatim.
    const TABLE_6_11_2_1_1: [(usize, usize); 168] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (29, 30),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
        (15, 17),
        (16, 18),
        (17, 19),
        (18, 20),
        (19, 21),
        (20, 22),
        (21, 23),
        (22, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 12),
        (10, 13),
        (11, 14),
        (12, 15),
        (13, 16),
        (14, 17),
        (15, 18),
        (16, 19),
        (17, 20),
        (18, 21),
        (19, 22),
        (20, 23),
        (21, 24),
        (22, 25),
        (23, 26),
        (24, 27),
        (25, 28),
        (26, 29),
        (27, 30),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
        (8, 12),
        (9, 13),
        (10, 14),
        (11, 15),
        (12, 16),
        (13, 17),
        (14, 18),
        (15, 19),
        (16, 20),
        (17, 21),
        (18, 22),
        (19, 23),
        (20, 24),
        (21, 25),
        (22, 26),
        (23, 27),
        (24, 28),
        (25, 29),
        (26, 30),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 10),
        (6, 11),
        (7, 12),
        (8, 13),
        (9, 14),
        (10, 15),
        (11, 16),
        (12, 17),
        (13, 18),
        (14, 19),
        (15, 20),
        (16, 21),
        (17, 22),
        (18, 23),
        (19, 24),
        (20, 25),
        (21, 26),
        (22, 27),
        (23, 28),
        (24, 29),
        (25, 30),
        (0, 6),
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 10),
        (5, 11),
        (6, 12),
        (7, 13),
        (8, 14),
        (9, 15),
        (10, 16),
        (11, 17),
        (12, 18),
        (13, 19),
        (14, 20),
        (15, 21),
        (16, 22),
        (17, 23),
        (18, 24),
        (19, 25),
        (20, 26),
        (21, 27),
        (22, 28),
        (23, 29),
        (24, 30),
        (0, 7),
        (1, 8),
        (2, 9),
    ];

    #[test]
    fn the_sss_indices_are_table_6_11_2_1_1() {
        for (n1, m) in TABLE_6_11_2_1_1.iter().enumerate() {
            assert_eq!(sss_indices(n1 as u32).unwrap(), *m, "N_ID1 = {n1}");
        }
    }

    #[test]
    fn the_sss_tell_the_group_and_the_half_frame() {
        let a = sss(57, 2, false).unwrap();
        let b = sss(57, 2, true).unwrap();
        assert!(a.iter().all(|v| v.abs() == 1.0));
        let r: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!(r.abs() < 20.0, "{r}");
        let mut best = (0.0, 0);
        for n1 in 0..168 {
            let c: f64 = sss(n1, 2, false).unwrap().iter().zip(&a).map(|(x, y)| x * y).sum();
            if c > best.0 {
                best = (c, n1);
            }
        }
        assert_eq!(best, (62.0, 57));
        assert!(sss(168, 0, false).is_err() && sss(0, 3, false).is_err());
    }

    #[test]
    fn the_crs_and_the_pbch_share_the_central_blocks_without_overlap() {
        let (id, n_rb) = (301, 25);
        let r = crs(id, 1, 0, n_rb).unwrap();
        assert_eq!(r.len(), 50);
        // the values of a narrower carrier are the central ones of a wider one
        let wide = crs(id, 1, 0, 100).unwrap();
        assert_eq!(r[..], wide[75..125]);
        let k0 = crs_subcarriers(id, 0, 1, 0, n_rb).unwrap();
        let k1 = crs_subcarriers(id, 1, 1, 0, n_rb).unwrap();
        assert_eq!((k0[0], k1[0]), (1, 4)); // v_shift = 301 mod 6 = 1
        let pbch = pbch_elements(id, n_rb).unwrap();
        assert_eq!(pbch.len(), PBCH_ELEMENTS);
        for (port, l) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
            let k = crs_subcarriers(id, port, 1, l, n_rb).unwrap();
            assert!(pbch.iter().all(|e| !(e.1 == l && k.contains(&e.0))));
        }
        assert!(crs_subcarriers(id, 2, 1, 0, n_rb).is_err() && crs_symbols(4).is_err());
        assert!(crs(504, 0, 0, 6).is_err() && crs(0, 20, 0, 6).is_err() && crs(0, 0, 0, 5).is_err());
    }

    #[test]
    fn the_carrier_leaves_the_dc_subcarrier_empty() {
        let c = Carrier::new(6, 128).unwrap();
        let k = c.n_subcarriers();
        // one symbol (the first of a subframe: prefix 10) with only the subcarrier K/2
        let mut grid = vec![C64::new(0.0, 0.0); k];
        grid[k / 2] = C64::new(1.0, 0.0);
        let x = c.modulate(&grid).unwrap();
        assert_eq!(x.len(), 128 + 10);
        let spectrum = fft::fft(&x[10..]);
        let peak = (0..128).max_by(|a, b| spectrum[*a].norm().total_cmp(&spectrum[*b].norm())).unwrap();
        assert_eq!(peak, 1); // +15 kHz, not the DC bin
        // a slot round trip
        let grid: Vec<C64> = (0..7 * k).map(|i| C64::new((i % 5) as f64, (i % 3) as f64)).collect();
        let x = c.modulate(&grid).unwrap();
        assert_eq!(x.len(), c.slot_len());
        let back = c.demodulate(&x).unwrap();
        assert!(back.iter().zip(&grid).all(|(a, b)| (a - b).norm() < 1e-9));
        assert!(Carrier::new(100, 1024).is_err() && Carrier::new(5, 128).is_err());
        assert!(fft_len(25).unwrap() == 512 && fft_len(10).is_err());
    }

    #[test]
    fn transmit_diversity_is_undone_by_the_orthogonal_decoder() {
        for ports in [2, 4] {
            let code = sfbc(ports).unwrap();
            let s: Vec<C64> = (0..8).map(|i| C64::new((i as f64).cos(), (i as f64).sin())).collect();
            let x = code.encode(&s).unwrap();
            // unit total power on every element
            for t in 0..x.ncols() {
                let p: f64 = x.column(t).iter().map(C64::norm_sqr).sum();
                assert!((p - 1.0).abs() < 1e-12, "{p}");
            }
            let h = ndarray::Array2::from_shape_fn((1, ports), |(_, a)| C64::from_polar(1.0 + a as f64, a as f64));
            let y = h.dot(&x);
            let est = code.decode(&h.view(), &y.view(), 1e-3).unwrap();
            assert!(est.symbols.iter().zip(&s).all(|(a, b)| (a - b).norm() < 1e-9));
        }
        // ports 0 and 1 of the first pair: [x0, -x1*] then [x1, x0*], over sqrt 2
        let x = sfbc(2).unwrap().encode(&[C64::new(1.0, 0.0), C64::new(0.0, 1.0)]).unwrap();
        assert!((x[[1, 0]] - C64::new(0.0, 1.0) * FRAC_1_SQRT_2).norm() < 1e-12);
        assert!(sfbc(3).is_err());
    }
}
