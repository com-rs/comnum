//! The cell search of an LTE terminal and the decoding of the master information block,
//! on the six central resource blocks sampled at 1.92 MS/s ([`SEARCH_RATE`]; a capture
//! is brought there by [`comnum_dsp::frontend::select_channel`]).
//!
//! # Cell search ([`cell_search`])
//!
//! 1. **PSS and coarse frequency.** For every $N_{\mathrm{ID}}^{(2)}$ and every frequency
//!    offset of a grid ($\pm$ `max_cfo_hz` by `cfo_step_hz`), the normalised correlation
//!    $M(d) = |\sum_{n} r\[d + n\] p^{*}\[n\]|^{2} / (\sum_{n} |r\[d + n\]|^{2} \sum_{n}
//!    |p\[n\]|^{2})$ of the first 10 ms with the 128 samples $p$ of the PSS symbol (shifted by
//!    the offset) is computed; its peak gives the identity, the offset and the start of the
//!    PSS symbol (after its prefix), modulo 5 ms.
//! 2. **Fine frequency.** The prefixes repeat the end of their symbols: with the slots
//!    aligned on the PSS, the phase of $\sum r\[n + N\] r^{*}\[n\]$ over the prefixes of every
//!    symbol of the capture ($N = 128$) is $2\pi \Delta f N / f_{s}$, unambiguous for
//!    $|\Delta f| < 7.5$ kHz (van de Beek et al. 1997, the correlation term of their eq. 6).
//! 3. **SSS, group and frame.** In every half frame, the SSS (the symbol before the PSS) is
//!    equalised by the channel estimated on the PSS, $z_{n} = y_{\mathrm{SSS}}(n)
//!    \hat{h}^{*}(n)$, $\hat{h}(n) = y_{\mathrm{PSS}}(n)/d_{\mathrm{PSS}}(n)$, and correlated
//!    with the 336 sequences of the identity $N_{\mathrm{ID}}^{(2)}$ (168 groups, subframe
//!    0 or 5), alternately from one half frame to the next; the largest sum gives
//!    $N_{\mathrm{ID}}^{(1)}$ and which half frames are the first of their frame.
//!
//! # MIB ([`decode_mib`])
//!
//! The start of every frame is tracked from the previous one: the peak of the PSS
//! correlation of slot 0 is searched within $\pm$ [`TRACKING_SAMPLES`] of where it is
//! expected, 10 ms after the previous peak, so that the sampling clock of the receiver may
//! drift (a software radio whose oscillator is 17 ppm off slips by 33 samples per second
//! at 1.92 MS/s, more than a prefix). In subframe 0 of every frame, the channel of each port is estimated on its CRS (least
//! squares, interpolated linearly over the 72 subcarriers, averaged over the CRS symbols
//! of the two slots: a static channel), the noise variance from the difference of the CRS
//! of port 0 in the first symbols of the two slots. For 1, 2 and 4 ports, the PBCH is
//! equalised (one port) or decoded by the orthogonal decoder of transmit diversity
//! ([`comnum_waveform::ofdm::lte::sfbc`]), demapped to LLRs, descrambled for each of the
//! four positions of the frame in the 40 ms period and decoded ([`super::bch::decode`]);
//! the first CRC that passes under the mask of the number of ports assumed gives the MIB,
//! the number of ports and the two last bits of the frame number.
//!
//! # Examples
//! ```
//! use comnum_systems::lte::cell::Downlink;
//! use comnum_systems::lte::mib::{Mib, PhichDuration, PhichResource};
//! use comnum_systems::lte::search::{cell_search, decode_mib, SearchConfig};
//! let mib = Mib { n_rb: 6, phich_duration: PhichDuration::Normal, phich_resource: PhichResource::Half, sfn_msb: 200, rest: [0; 10] };
//! let cell = Downlink { n_cell_id: 211, mib, ports: 1 };
//! let mut x = Vec::new();
//! for sfn in 802..804 {
//!     x.extend(cell.frame_samples(sfn, 128, None)?.remove(0));
//! }
//! let config = SearchConfig { max_cfo_hz: 1e3, ..SearchConfig::default() };
//! let found = cell_search(&x, &config)?;
//! assert_eq!((found[0].n_cell_id, found[0].frame_start), (211, 0));
//! let frames = decode_mib(&x, &found[0], 4)?;
//! assert_eq!((frames[0].sfn, frames[1].sfn, frames[0].ports), (802, 803, 1));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 36.211 V18.0.2, Sec. 6.6, 6.10.1, 6.11.
//! 2. 3GPP TS 36.212 V18.1.0, Sec. 5.3.1.
//! 3. J.-J. van de Beek, M. Sandell and P. O. Börjesson, "ML estimation of time and
//!    frequency offset in OFDM systems," *IEEE Trans. Signal Process.* 45(7), 1800–1805,
//!    1997.

use super::bch::{self, FRAME_BITS};
use super::mib::Mib;
use crate::nr::modulation::constellation;
use comnum_core::traits::Jacobian;
use comnum_core::{Result, ensure, ensure_shape};
use comnum_dsp::interpolation::linear;
use comnum_dsp::signal::rotate;
use comnum_math::C64;
use comnum_math::linalg::CMat;
use comnum_math::sequences::gold_3gpp;
use comnum_waveform::ofdm::lte::{self, Carrier, PSS_SYMBOL, SSS_SYMBOL, SYMBOLS_PER_SLOT, sfbc};
use std::f64::consts::PI;

/// The sampling rate of the cell search, 1.92 MS/s: the six central resource blocks and
/// an FFT of 128.
pub const SEARCH_RATE: f64 = 1.92e6;
/// Samples of an OFDM symbol at [`SEARCH_RATE`], without its prefix.
const N: usize = 128;
/// Samples of a slot, a half frame and a frame at [`SEARCH_RATE`].
const SLOT: usize = 960;
const HALF_FRAME: usize = 9600;
const FRAME: usize = 19_200;
/// The six central resource blocks.
const N_RB: usize = 6;
/// The samples by which [`decode_mib`] may move the start of a frame from the one it
/// expects, 10 ms after the previous one: a clock offset of up to 400 ppm.
pub const TRACKING_SAMPLES: usize = 8;

/// The settings of [`cell_search`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// The largest frequency offset searched (Hz), the error of the oscillator of the
    /// receiver: 20 ppm at 1 GHz is 20 kHz.
    pub max_cfo_hz: f64,
    /// The step of the grid of offsets of the PSS search (Hz); the fine estimate on the
    /// prefixes removes the rest.
    pub cfo_step_hz: f64,
    /// The smallest normalised PSS correlation $M(d)$ that counts as a cell (0 to 1).
    pub min_metric: f64,
}

impl Default for SearchConfig {
    /// $\pm 20$ kHz by 1 kHz, $M \ge 0.3$.
    fn default() -> SearchConfig {
        SearchConfig { max_cfo_hz: 20e3, cfo_step_hz: 1e3, min_metric: 0.3 }
    }
}

/// A cell found by [`cell_search`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundCell {
    /// The cell identity $N_{\mathrm{ID}}^{\mathrm{cell}} = 3 N_{\mathrm{ID}}^{(1)} + N_{\mathrm{ID}}^{(2)}$.
    pub n_cell_id: u32,
    /// The frequency offset of the cell with respect to the receiver (Hz): the signal
    /// is $r\[n\] e^{j 2\pi \Delta f n / f_{s}}$.
    pub cfo_hz: f64,
    /// The first sample of the first whole frame of the capture.
    pub frame_start: usize,
    /// The peak of the normalised PSS correlation (0 to 1).
    pub pss_metric: f64,
    /// The normalised SSS correlation of the decision (at most 1).
    pub sss_metric: f64,
}

/// The 128 samples of the PSS symbol of $N_{\mathrm{ID}}^{(2)}$ at [`SEARCH_RATE`], its prefix
/// removed.
///
/// # Errors
/// [`comnum_core::Error::Config`] for $N_{\mathrm{ID}}^{(2)} > 2$.
pub fn pss_symbol(n_id2: u32) -> Result<Vec<C64>> {
    let carrier = Carrier { first_symbol: PSS_SYMBOL, ..Carrier::new(N_RB, N)? };
    let mut grid = vec![C64::new(0.0, 0.0); carrier.n_subcarriers()];
    let first = lte::sync_first_subcarrier(N_RB);
    grid[first..first + lte::SYNC_LENGTH].copy_from_slice(&lte::pss(n_id2)?);
    let x = carrier.modulate(&grid)?;
    Ok(x[x.len() - N..].to_vec())
}

/// The prefix sums of $|r\[n\]|^{2}$, for windowed energies.
fn energy_prefix(r: &[C64]) -> Vec<f64> {
    let mut e = vec![0.0; r.len() + 1];
    for (i, v) in r.iter().enumerate() {
        e[i + 1] = e[i] + v.norm_sqr();
    }
    e
}

/// The peak `(M, d)` of the normalised correlation of `r` with `p`.
fn correlation_peak(r: &[C64], p: &[C64], energy: &[f64]) -> (f64, usize) {
    let ep: f64 = p.iter().map(C64::norm_sqr).sum();
    let mut best = (0.0, 0);
    for d in 0..=r.len().saturating_sub(p.len()) {
        let c: C64 = r[d..d + p.len()].iter().zip(p).map(|(a, b)| a * b.conj()).sum();
        let m = c.norm_sqr() / ((energy[d + p.len()] - energy[d]) * ep).max(1e-300);
        if m > best.0 {
            best = (m, d);
        }
    }
    best
}

/// `r` with the offset `cfo_hz` removed.
fn derotated(r: &[C64], cfo_hz: f64) -> Vec<C64> {
    let mut y = r.to_vec();
    rotate(&mut y, -2.0 * PI * cfo_hz / SEARCH_RATE, 0.0);
    y
}

/// The start of every symbol of a slot, relative to the slot (prefixes 10 then 9).
fn symbol_starts() -> [usize; SYMBOLS_PER_SLOT] {
    std::array::from_fn(|l| if l == 0 { 0 } else { 10 + N + (l - 1) * (9 + N) })
}

/// The residual frequency offset (Hz) of the prefixes of the symbols of every slot that
/// starts at `slot0 + 960 m`.
fn prefix_cfo(y: &[C64], slot0: usize) -> f64 {
    let starts = symbol_starts();
    let mut acc = C64::new(0.0, 0.0);
    let mut s = slot0;
    while s + SLOT <= y.len() {
        for (l, &st) in starts.iter().enumerate() {
            let cp = if l == 0 { 10 } else { 9 };
            let a = s + st;
            acc += (0..cp).map(|n| y[a + n + N] * y[a + n].conj()).sum::<C64>();
        }
        s += SLOT;
    }
    acc.arg() * SEARCH_RATE / (2.0 * PI * N as f64)
}

/// The grid (`symbols × 72`) of `count` symbols from the symbol `first` (in the subframe)
/// of the slot that starts at `slot_start`.
fn symbols_at(y: &[C64], slot_start: usize, first: usize, count: usize) -> Result<Vec<C64>> {
    let carrier = Carrier { first_symbol: first, ..Carrier::new(N_RB, N)? };
    let a = slot_start + symbol_starts()[first % SYMBOLS_PER_SLOT];
    let len: usize = (first..first + count).map(|l| carrier.prefix(l) + N).sum();
    ensure_shape(a + len <= y.len(), || format!("the symbols end at {} after the capture", a + len))?;
    carrier.demodulate(&y[a..a + len])
}

/// The strongest cell of every $N_{\mathrm{ID}}^{(2)}$ whose PSS correlation reaches
/// `config.min_metric`, in the samples `r` at [`SEARCH_RATE`] (see the module), the
/// strongest first.
///
/// # Errors
/// [`comnum_core::Error::Shape`] for fewer than 10 ms and one symbol;
/// [`comnum_core::Error::Config`] for a negative range, a step that is not positive or
/// a threshold outside $(0, 1\]$.
pub fn cell_search(r: &[C64], config: &SearchConfig) -> Result<Vec<FoundCell>> {
    ensure_shape(r.len() >= FRAME + SLOT, || {
        format!("the cell search needs 10.5 ms ({} samples at 1.92 MS/s), got {}", FRAME + SLOT, r.len())
    })?;
    ensure(config.max_cfo_hz >= 0.0 && config.cfo_step_hz > 0.0, || {
        format!("the offsets ±{} Hz by {} Hz: a range and a positive step", config.max_cfo_hz, config.cfo_step_hz)
    })?;
    ensure(config.min_metric > 0.0 && config.min_metric <= 1.0, || {
        format!("the PSS threshold is in (0, 1], got {}", config.min_metric)
    })?;
    let window = &r[..FRAME + N];
    let energy = energy_prefix(window);
    let steps = (config.max_cfo_hz / config.cfo_step_hz).floor() as i64;
    let mut cells = Vec::new();
    for n_id2 in 0..3 {
        let p = pss_symbol(n_id2)?;
        let mut best = (0.0, 0, 0.0);
        for i in -steps..=steps {
            let f = i as f64 * config.cfo_step_hz;
            // the replica shifted by f: the same peak as the derotated signal
            let mut q = p.clone();
            rotate(&mut q, 2.0 * PI * f / SEARCH_RATE, 0.0);
            let (m, d) = correlation_peak(window, &q, &energy);
            if m > best.0 {
                best = (m, d, f);
            }
        }
        let (metric, d, coarse) = best;
        if metric < config.min_metric {
            continue;
        }
        // the slot of the PSS: the symbol 6 starts 832 samples after the slot
        let pss_start = if d >= 832 { d } else { d + HALF_FRAME };
        let slot0 = pss_start - (symbol_starts()[PSS_SYMBOL] + 9);
        let fine = prefix_cfo(&derotated(r, coarse), slot0 % SLOT);
        let cfo_hz = coarse + fine;
        let y = derotated(r, cfo_hz);
        let (n_id1, second, sss_metric) = detect_sss(&y, slot0, n_id2)?;
        let start = if second { slot0 + FRAME - HALF_FRAME } else { slot0 };
        cells.push(FoundCell {
            n_cell_id: 3 * n_id1 + n_id2,
            cfo_hz,
            frame_start: start % FRAME,
            pss_metric: metric,
            sss_metric,
        });
    }
    cells.sort_by(|a, b| b.pss_metric.total_cmp(&a.pss_metric));
    Ok(cells)
}

/// The group, whether the half frame of the slot `slot0` is the second of its frame, and
/// the normalised correlation, from the SSS of every half frame (see the module).
fn detect_sss(y: &[C64], slot0: usize, n_id2: u32) -> Result<(u32, bool, f64)> {
    let first = lte::sync_first_subcarrier(N_RB);
    let pss = lte::pss(n_id2)?;
    let mut table = Vec::with_capacity(336);
    for n1 in 0..168 {
        table.push((lte::sss(n1, n_id2, false)?, lte::sss(n1, n_id2, true)?));
    }
    let mut scores = vec![[0.0; 2]; 168];
    let mut norm = 0.0;
    let mut s = slot0;
    let mut odd = false;
    while s + SLOT <= y.len() {
        let g = symbols_at(y, s, SSS_SYMBOL, 2)?;
        let k = lte::SUBCARRIERS_PER_RB * N_RB;
        let z: Vec<C64> = (0..lte::SYNC_LENGTH).map(|n| g[first + n] * (g[k + first + n] / pss[n]).conj()).collect();
        norm += z.iter().map(C64::norm_sqr).sum::<f64>().sqrt() * (lte::SYNC_LENGTH as f64).sqrt();
        for (score, (d0, d5)) in scores.iter_mut().zip(&table) {
            let c0: f64 = z.iter().zip(d0).map(|(a, b)| a.re * b).sum();
            let c5: f64 = z.iter().zip(d5).map(|(a, b)| a.re * b).sum();
            // hypothesis 0: the first half frame is subframe 0
            score[0] += if odd { c5 } else { c0 };
            score[1] += if odd { c0 } else { c5 };
        }
        s += HALF_FRAME;
        odd = !odd;
    }
    let mut best = (f64::NEG_INFINITY, 0, false);
    for (n1, sc) in scores.iter().enumerate() {
        for (h, &v) in sc.iter().enumerate() {
            if v > best.0 {
                best = (v, n1 as u32, h == 1);
            }
        }
    }
    Ok((best.1, best.2, best.0 / norm.max(1e-300)))
}

/// A frame whose master information block was decoded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MibFrame {
    /// The first sample of the frame in the capture.
    pub start: usize,
    /// The antenna ports of the cell (from the CRC mask).
    pub ports: usize,
    /// The master information block.
    pub mib: Mib,
    /// The system frame number of the frame.
    pub sfn: usize,
}

/// The channel of every port over the 72 subcarriers, and the noise variance, from the
/// CRS of subframe 0 (`grid`: 14 symbols × 72).
fn crs_channels(grid: &[C64], n_cell_id: u32) -> Result<(Vec<Vec<C64>>, f64)> {
    let k = lte::SUBCARRIERS_PER_RB * N_RB;
    let mut channels = Vec::with_capacity(4);
    let mut ls_first = Vec::new();
    for port in 0..4 {
        let mut sum = vec![C64::new(0.0, 0.0); k];
        let mut count = 0.0;
        for ns in 0..2 {
            for &l in lte::crs_symbols(port)? {
                let sub = lte::crs_subcarriers(n_cell_id, port, ns, l, N_RB)?;
                let values = lte::crs(n_cell_id, ns, l, N_RB)?;
                let row = (ns * SYMBOLS_PER_SLOT + l) * k;
                let ls: Vec<C64> = sub.iter().zip(&values).map(|(&kk, v)| grid[row + kk] / v).collect();
                if port == 0 && l == 0 {
                    ls_first.push(ls.clone());
                }
                for (s, h) in sum.iter_mut().zip(linear(&sub, &ls, k)?) {
                    *s += h;
                }
                count += 1.0;
            }
        }
        channels.push(sum.iter().map(|s| s / count).collect());
    }
    // two LS estimates of the same channel: their difference has twice the noise
    let noise = ls_first[0].iter().zip(&ls_first[1]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>()
        / (2.0 * ls_first[0].len() as f64);
    Ok((channels, noise.max(1e-12)))
}

/// The equalised PBCH symbols of slot 1 and their noise variances, for `ports` ports.
fn equalise(
    grid: &[C64],
    elements: &[(usize, usize)],
    h: &[Vec<C64>],
    noise: f64,
    ports: usize,
) -> Result<(Vec<C64>, Vec<f64>)> {
    let k = lte::SUBCARRIERS_PER_RB * N_RB;
    let y: Vec<C64> = elements.iter().map(|&(kk, l)| grid[(SYMBOLS_PER_SLOT + l) * k + kk]).collect();
    if ports == 1 {
        let z = elements.iter().zip(&y).map(|(&(kk, _), v)| v / h[0][kk]).collect();
        let var = elements.iter().map(|&(kk, _)| noise / h[0][kk].norm_sqr().max(1e-300)).collect();
        return Ok((z, var));
    }
    let code = sfbc(ports)?;
    let blocks: Vec<CMat> = elements
        .chunks_exact(ports)
        .map(|block| {
            CMat::from_shape_fn((1, ports), |(_, p)| block.iter().map(|&(kk, _)| h[p][kk]).sum::<C64>() / ports as f64)
        })
        .collect();
    let y = CMat::from_shape_vec((1, y.len()), y).map_err(|e| comnum_core::Error::Shape(e.to_string()))?;
    let est = code.decode_blocks(&blocks, &y.view(), noise)?;
    Ok((est.symbols, est.noise_var))
}

/// The master information block of every whole subframe 0 of the capture `r` (at
/// [`SEARCH_RATE`]) of the cell `cell` whose PBCH decodes, by `passes` passes of the
/// wrap-around Viterbi algorithm (see the module).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a cell identity above 503 or `passes = 0`.
pub fn decode_mib(r: &[C64], cell: &FoundCell, passes: usize) -> Result<Vec<MibFrame>> {
    ensure(passes > 0, || "the Viterbi decoder needs at least one pass".into())?;
    let y = derotated(r, cell.cfo_hz);
    let elements = lte::pbch_elements(cell.n_cell_id, N_RB)?;
    let scrambling = gold_3gpp(cell.n_cell_id, bch::BCH_BITS);
    let points = constellation(2)?;
    let carrier = Carrier::new(N_RB, N)?;
    let pss = pss_symbol(cell.n_cell_id % 3)?;
    // the PSS of slot 0 starts 832 samples after the frame
    let to_pss = symbol_starts()[PSS_SYMBOL] + 9;
    let mut out = Vec::new();
    let mut start = cell.frame_start;
    while start + to_pss + N + TRACKING_SAMPLES <= y.len() && start + 2 * SLOT <= y.len() {
        let lo = (start + to_pss).saturating_sub(TRACKING_SAMPLES);
        let window = &y[lo..start + to_pss + N + TRACKING_SAMPLES];
        let (_, d) = correlation_peak(window, &pss, &energy_prefix(window));
        start = lo + d - to_pss;
        if start + 2 * SLOT > y.len() {
            break;
        }
        let grid = carrier.demodulate(&y[start..start + 2 * SLOT])?;
        let (h, noise) = crs_channels(&grid, cell.n_cell_id)?;
        'ports: for ports in [1, 2, 4] {
            let (z, var) = equalise(&grid, &elements, &h, noise, ports)?;
            let llr = points.llr_with_variances(&z, &var, None, Jacobian::Exact)?;
            for quarter in 0..4 {
                let q = quarter * FRAME_BITS;
                let descrambled: Vec<f64> = llr
                    .iter()
                    .zip(&scrambling[q..q + FRAME_BITS])
                    .map(|(l, &c)| if c == 0 { *l } else { -l })
                    .collect();
                // (nested `if`s rather than a `let` chain: the MSRV is 1.85)
                if let Some(d) = bch::decode(&descrambled, q, passes)? {
                    if d.ports == ports {
                        if let Ok(mib) = Mib::from_bits(&d.bits) {
                            out.push(MibFrame { start, ports, mib, sfn: mib.sfn(quarter) });
                            break 'ports;
                        }
                    }
                }
            }
        }
        start += FRAME;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lte::cell::Downlink;
    use crate::lte::mib::{PhichDuration, PhichResource};
    use comnum_core::rng::Rng;
    use comnum_dsp::frontend::select_channel;

    fn mib(n_rb: usize, sfn_msb: u8) -> Mib {
        Mib {
            n_rb,
            phich_duration: PhichDuration::Extended,
            phich_resource: PhichResource::Two,
            sfn_msb,
            rest: [0; 10],
        }
    }

    /// `frames` frames of `cell` from `sfn`, through a channel per port and noise, delayed
    /// by `delay` samples, offset by `cfo_hz`, at the rate of an FFT of `n_fft`.
    fn received(
        cell: &Downlink,
        sfn: usize,
        frames: usize,
        n_fft: usize,
        delay: usize,
        cfo_hz: f64,
        snr_db: f64,
        rng: &mut Rng,
    ) -> Vec<C64> {
        let h: Vec<C64> = (0..cell.ports).map(|p| C64::from_polar(1.0, 0.7 * p as f64 + 0.3)).collect();
        let mut x = vec![C64::new(0.0, 0.0); delay];
        for f in 0..frames {
            let ports = cell.frame_samples(sfn + f, n_fft, Some(rng)).unwrap();
            x.extend((0..ports[0].len()).map(|n| ports.iter().zip(&h).map(|(s, g)| s[n] * g).sum::<C64>()));
        }
        let fs = n_fft as f64 * 15e3;
        rotate(&mut x, 2.0 * PI * cfo_hz / fs, 0.0);
        let power = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
        let sigma = (power / 10f64.powf(snr_db / 10.0)).sqrt();
        x.iter().map(|v| v + rng.complex_gaussian() * sigma).collect()
    }

    #[test]
    fn a_cell_of_two_ports_is_found_and_its_mib_decoded_after_the_front_end() {
        let mut rng = Rng::new(36_211);
        let cell = Downlink { n_cell_id: 287, mib: mib(25, 0xa7), ports: 2 };
        // 5 MHz at 7.68 MS/s, 13.4 kHz off, 5 dB of SNR over the 4.5 MHz of the carrier
        let x = received(&cell, 669, 5, 512, 12_345, 13.4e3, 5.0, &mut rng);
        let y = select_channel(&x, 7.68e6, 0.0, SEARCH_RATE, 0.6e6, 129).unwrap();
        let found = cell_search(&y, &SearchConfig::default()).unwrap();
        let c = found[0];
        assert_eq!(c.n_cell_id, 287);
        assert!((c.cfo_hz - 13.4e3).abs() < 50.0, "{}", c.cfo_hz);
        // the delay of 12 345 samples at 7.68 MS/s is 3086.25 at 1.92 MS/s
        assert!((c.frame_start as f64 - 3086.25).abs() <= 1.0, "{}", c.frame_start);
        let frames = decode_mib(&y, &c, 4).unwrap();
        assert!(frames.len() >= 4, "{frames:?}");
        for f in &frames {
            assert_eq!((f.ports, f.mib), (2, Mib { sfn_msb: (f.sfn / 4) as u8, ..cell.mib }));
        }
        // the frame numbers follow the frames, from 669, the first whole frame after the
        // delay (at 5 dB, a frame may fail its CRC)
        assert!(frames.iter().all(|f| f.sfn == 669 + (f.start - c.frame_start) / FRAME), "{frames:?}");
    }

    #[test]
    fn one_and_four_ports_are_told_by_the_crc_mask() {
        let mut rng = Rng::new(5);
        for (ports, id) in [(1, 0), (4, 503)] {
            let cell = Downlink { n_cell_id: id, mib: mib(6, 3), ports };
            let x = received(&cell, 4, 2, 128, 700, -2.2e3, 10.0, &mut rng);
            let config = SearchConfig { max_cfo_hz: 5e3, ..SearchConfig::default() };
            let c = cell_search(&x, &config).unwrap()[0];
            assert_eq!((c.n_cell_id, c.frame_start), (id, 700));
            let frames = decode_mib(&x, &c, 4).unwrap();
            assert!(!frames.is_empty() && frames.iter().all(|f| f.ports == ports && f.sfn >= 4));
        }
    }

    #[test]
    fn the_frames_are_tracked_under_a_sampling_clock_offset() {
        let mut rng = Rng::new(17);
        let cell = Downlink { n_cell_id: 55, mib: mib(6, 0), ports: 2 };
        let x = received(&cell, 700, 12, 128, 300, 1.0e3, 15.0, &mut rng);
        // resampled by a clock 400 ppm fast: 92 samples late after 12 frames
        let ppm = 400e-6;
        let t: Vec<f64> = (0..(x.len() as f64 / (1.0 + ppm)) as usize).map(|n| n as f64 * (1.0 + ppm)).collect();
        let y: Vec<C64> = t
            .iter()
            .map(|&u| {
                let (i, f) = (u.floor() as usize, u - u.floor());
                x[i] * (1.0 - f) + x[(i + 1).min(x.len() - 1)] * f
            })
            .collect();
        let c = cell_search(&y, &SearchConfig { max_cfo_hz: 2e3, ..SearchConfig::default() }).unwrap()[0];
        let frames = decode_mib(&y, &c, 4).unwrap();
        assert!(frames.len() >= 11, "{} frames of 12", frames.len());
        let last = frames.last().unwrap();
        assert_eq!(last.sfn, 700 + (last.start as f64 * (1.0 + ppm) / FRAME as f64) as usize);
    }

    #[test]
    fn noise_alone_finds_no_cell_and_bad_settings_are_errors() {
        let mut rng = Rng::new(9);
        let x: Vec<C64> = (0..FRAME + SLOT).map(|_| rng.complex_gaussian()).collect();
        let config = SearchConfig { max_cfo_hz: 2e3, ..SearchConfig::default() };
        assert_eq!(cell_search(&x, &config).unwrap(), Vec::new());
        assert!(cell_search(&x[..FRAME], &config).is_err());
        assert!(cell_search(&x, &SearchConfig { cfo_step_hz: 0.0, ..config }).is_err());
        assert!(cell_search(&x, &SearchConfig { min_metric: 0.0, ..config }).is_err());
        let cell = FoundCell { n_cell_id: 0, cfo_hz: 0.0, frame_start: 0, pss_metric: 1.0, sss_metric: 1.0 };
        assert_eq!(decode_mib(&x, &cell, 4).unwrap(), Vec::new());
        assert!(decode_mib(&x, &cell, 0).is_err());
        assert!(pss_symbol(3).is_err());
    }
}
