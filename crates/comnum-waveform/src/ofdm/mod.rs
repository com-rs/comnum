//! OFDM and DFT-spread OFDM.
//!
//! A block carries $K$ symbols on $K$ contiguous subcarriers centred on the carrier,
//! out of a grid of $N$ subcarriers; the time signal is computed by an IFFT of size
//! $L N$, an oversampling by $L$ that the peak measurements need (the peaks of the
//! continuous signal fall between the Nyquist-rate samples; $L = 4$ is the usual
//! choice), and a cyclic prefix of $N_{\mathrm{cp}}$ Nyquist-rate samples ($L N_{\mathrm{cp}}$
//! samples) is prepended.
//!
//! # Model
//!
//! With $X_{k}$ the symbol of subcarrier $k \in \\{-K/2, \dots, K/2 - 1\\}$ (unit average
//! energy),
//!
//! $$
//! x\[n\] = \frac{1}{\sqrt{K}} \sum_{k} X_{k}\\, e^{\\,j 2 \pi k n / (L N)}, \qquad
//! n = 0, \dots, L N - 1,
//! $$
//!
//! so that $\mathbb{E}|x\[n\]|^{2} = 1$: the samples have unit average power whatever $K$,
//! $N$ and $L$. With `dft_spread`, the $K$ data symbols are first transformed by a
//! unitary DFT of size $K$ (localised SC-FDMA, the uplink of LTE): the signal is then a
//! single-carrier signal interpolated by the subcarrier mapping, whose PAPR is that of
//! the constellation, not that of a Gaussian signal.
//!
//! The submodules build the rest of an OFDM link: [`grid`] (data and pilot resource
//! elements), [`estimation`] (LS and LMMSE channel estimation on the grid), [`wiener`]
//! (two-dimensional Wiener interpolation of pilots, Hoeher et al. 1997), [`sync`]
//! (Schmidl & Cox timing and frequency synchronisation), [`budget`] (the SNR loss of
//! residual synchronisation and estimation errors, Speth et al. 1999), [`detection`] (MIMO-OFDM,
//! per-subcarrier detection) and [`correction`] (pilot-based gain correction against a
//! cochannel interferer, Cimini 1985); the signals of the standards are in [`nr`], [`lte`]
//! (the synchronisation, reference and broadcast signals of the LTE downlink) and [`dvbt`].
//!
//! # Examples
//! ```
//! use comnum_waveform::ofdm::Ofdm;
//! use comnum_math::C64;
//! let ofdm = Ofdm { n_cp: 16, ..Ofdm::new(256, 200)? };   // L = 4 by default
//! let d: Vec<C64> = (0..400).map(|i| C64::new(if i % 3 == 0 { 1.0 } else { -1.0 }, 0.0)).collect();
//! let x = ofdm.modulate(&d)?;
//! assert_eq!(x.len(), 2 * 4 * (256 + 16));
//! let back = ofdm.demodulate(&x)?;
//! assert!(back.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-12));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. R. van Nee and R. Prasad, *OFDM for Wireless Multimedia Communications*, Artech
//!    House, 2000.
//! 2. H. G. Myung, J. Lim and D. J. Goodman, "Single carrier FDMA for uplink wireless
//!    transmission," *IEEE Veh. Technol. Mag.*, 1(3), 30–38, 2006.

pub mod budget;
pub mod correction;
pub mod detection;
pub mod dvbt;
pub mod estimation;
pub mod grid;
pub mod link;
pub mod lte;
pub mod nr;
pub mod sync;
pub mod wiener;

use crate::{check_input, check_rate};
use comnum_core::signal::{Signal, Symbols};
use comnum_core::traits::{Demodulate, Modulate};
use comnum_core::units::Hertz;
use comnum_core::{Result, ensure, ensure_shape};
use comnum_dsp::signal::{extend_with_prefix, strip_prefix};
use comnum_math::C64;
use comnum_math::fft;

/// An OFDM (or DFT-spread OFDM) modulator and demodulator.
#[derive(Clone, Debug, PartialEq)]
pub struct Ofdm {
    /// Size $N$ of the subcarrier grid.
    pub n_fft: usize,
    /// Number $K$ of active (modulated) subcarriers, centred on the carrier.
    pub n_active: usize,
    /// Oversampling factor $L$ (IFFT of size $L N$).
    pub oversampling: usize,
    /// Cyclic prefix, in Nyquist-rate samples ($L N_{\mathrm{cp}}$ samples).
    pub n_cp: usize,
    /// Leave the DC subcarrier empty (the active subcarriers skip $k = 0$).
    pub dc_null: bool,
    /// DFT spreading of the data (localised SC-FDMA).
    pub dft_spread: bool,
    /// The rate of the modulated samples, $L N \Delta f$ ($\Delta f$ the subcarrier
    /// spacing): the rate of the [`Signal`] of [`Modulate`], and the rate [`Demodulate`]
    /// expects; 1 Hz by default (time in samples).
    pub sample_rate: Hertz,
    /// Optional complex weights $w_{k}$ of the $K$ active subcarriers (power loading,
    /// a precoder per subcarrier): the transmitter sends $w_{k} X_{k}$ instead of
    /// $X_{k}$ ([`Ofdm::to_time`]) and the receiver divides by $w_{k}$
    /// ([`Ofdm::to_active`]), so that a round trip returns the data. The average power
    /// of the samples becomes $\frac{1}{K} \sum_{k} |w_{k}|^{2}$: weights with a mean
    /// square of 1 (for instance $w_{k} = \sqrt{p_{k}}$ with the powers $p_{k}$ of
    /// [`comnum_mimo::precoding::water_filling_powers`] for a total of $K$) keep the unit power.
    /// A zero weight leaves its subcarrier empty, and the receiver returns 0 on it.
    /// `None` (the default) is $w_{k} = 1$.
    pub weights: Option<Vec<C64>>,
}

impl Ofdm {
    /// $K$ active subcarriers out of $N$, oversampling 4, no prefix, at the sample rate
    /// 1 Hz.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] unless $1 \le K \le N$.
    pub fn new(n_fft: usize, n_active: usize) -> Result<Ofdm> {
        let o = Ofdm {
            n_fft,
            n_active,
            oversampling: 4,
            n_cp: 0,
            dc_null: false,
            dft_spread: false,
            sample_rate: Hertz(1.0),
            weights: None,
        };
        o.check()?;
        Ok(o)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for an empty or overfull grid, a zero oversampling, a
    /// sample rate that is not positive and finite, or weights that are not $K$ finite
    /// values.
    pub fn check(&self) -> Result<()> {
        ensure(self.n_active >= 1 && self.n_active + usize::from(self.dc_null) <= self.n_fft, || {
            format!("{} active subcarriers do not fit a grid of {}", self.n_active, self.n_fft)
        })?;
        ensure(self.oversampling >= 1, || "the oversampling factor must be at least 1".into())?;
        if let Some(w) = &self.weights {
            ensure(w.len() == self.n_active, || {
                format!("{} weights for {} active subcarriers", w.len(), self.n_active)
            })?;
            ensure(w.iter().all(|v| v.re.is_finite() && v.im.is_finite()), || "the weights must be finite".into())?;
        }
        check_rate(self.sample_rate)
    }

    /// Size of the IFFT, $L N$.
    pub fn fft_len(&self) -> usize {
        self.oversampling * self.n_fft
    }

    /// Samples per block, prefix included, $L (N + N_{\mathrm{cp}})$.
    pub fn block_len(&self) -> usize {
        self.oversampling * (self.n_fft + self.n_cp)
    }

    /// Subcarrier index $k$ of the active subcarrier `i`.
    pub fn subcarrier(&self, i: usize) -> isize {
        let k = i as isize - (self.n_active / 2) as isize;
        if self.dc_null && k >= 0 { k + 1 } else { k }
    }

    /// FFT bin of the active subcarrier `i` ([`fft::centred_bin`], one bin further from
    /// the subcarrier 0 on with `dc_null`).
    pub fn bin(&self, i: usize) -> usize {
        let skip = usize::from(self.dc_null && i >= self.n_active / 2);
        fft::centred_bin(i + skip, self.n_active, self.fft_len())
    }

    /// The transmit pulse of the active subcarrier `i` of symbol `n`, prefix included, on
    /// its support: `(first sample, values)`, $\frac{1}{\sqrt{LN}} e^{j 2\pi k (t -
    /// L N_{\mathrm{cp}})/(LN)}$ for $t = 0, \dots, L(N + N_{\mathrm{cp}}) - 1$ (energy
    /// $(N + N_{\mathrm{cp}})/N$); a column of the matrix $\mathbf{G}$ of Nissel, Schwarz &
    /// Rupp 2017, eq. (20).
    pub fn pulse(&self, i: usize, n: usize) -> (usize, Vec<C64>) {
        let (len, cp) = (self.fft_len(), self.oversampling * self.n_cp);
        let (bin, scale) = (self.bin(i) as f64, 1.0 / (len as f64).sqrt());
        let values = (0..self.block_len())
            .map(|t| comnum_math::cis(std::f64::consts::TAU * bin * (t as f64 - cp as f64) / len as f64) * scale)
            .collect();
        (n * self.block_len(), values)
    }

    /// The receive pulse (the window of the demodulator) of the active subcarrier `i` of
    /// symbol `n`: the [`Ofdm::pulse`] without its prefix, of unit energy, on its support
    /// `(first sample, values)`.
    pub fn window(&self, i: usize, n: usize) -> (usize, Vec<C64>) {
        let (start, mut values) = self.pulse(i, n);
        let cp = self.oversampling * self.n_cp;
        values.drain(..cp);
        (start + cp, values)
    }

    /// Whether an FFT bin carries an active subcarrier.
    pub fn is_active_bin(&self, bin: usize) -> bool {
        (0..self.n_active).any(|i| self.bin(i) == bin)
    }

    fn scale(&self) -> f64 {
        self.fft_len() as f64 / (self.n_active as f64).sqrt()
    }

    /// The values of the active subcarriers of `data` (the identity, or the unitary DFT
    /// of size $K$ with `dft_spread`).
    pub fn spread(&self, data: &[C64]) -> Vec<C64> {
        if self.dft_spread {
            let s = 1.0 / (data.len() as f64).sqrt();
            fft::fft(data).into_iter().map(|v| v * s).collect()
        } else {
            data.to_vec()
        }
    }

    /// The inverse of [`Ofdm::spread`].
    pub fn despread(&self, active: &[C64]) -> Vec<C64> {
        if self.dft_spread {
            let s = (active.len() as f64).sqrt();
            fft::ifft(active).into_iter().map(|v| v * s).collect()
        } else {
            active.to_vec()
        }
    }

    /// The time block (without prefix, $L N$ samples) of the values of the active
    /// subcarriers, each multiplied by its weight $w_{k}$ ([`Ofdm::weights`]).
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] if `active` does not have $K$ values, or the weights
    /// do not have $K$ values.
    pub fn to_time(&self, active: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(active.len() == self.n_active, || {
            format!("{} values for {} active subcarriers", active.len(), self.n_active)
        })?;
        let w = self.checked_weights()?;
        let mut bins = vec![C64::new(0.0, 0.0); self.fft_len()];
        for (i, v) in active.iter().enumerate() {
            bins[self.bin(i)] = match w {
                Some(w) => *v * w[i],
                None => *v,
            };
        }
        let s = self.scale();
        Ok(fft::ifft(&bins).into_iter().map(|v| v * s).collect())
    }

    /// The values of the active subcarriers of a time block (without prefix), each
    /// divided by its weight $w_{k}$ ([`Ofdm::weights`]; 0 where $w_{k} = 0$).
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] if `block` does not have $L N$ samples, or the
    /// weights do not have $K$ values.
    pub fn to_active(&self, block: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(block.len() == self.fft_len(), || {
            format!("a block of {} samples for an IFFT of {}", block.len(), self.fft_len())
        })?;
        let w = self.checked_weights()?;
        let bins = fft::fft(block);
        let s = 1.0 / self.scale();
        Ok((0..self.n_active)
            .map(|i| {
                let v = bins[self.bin(i)] * s;
                match w {
                    Some(w) if w[i] == C64::new(0.0, 0.0) => C64::new(0.0, 0.0),
                    Some(w) => v / w[i],
                    None => v,
                }
            })
            .collect())
    }

    /// The weights, checked against $K$ (`None` for unit weights).
    fn checked_weights(&self) -> Result<Option<&[C64]>> {
        match &self.weights {
            Some(w) => {
                ensure_shape(w.len() == self.n_active, || {
                    format!("{} weights for {} active subcarriers", w.len(), self.n_active)
                })?;
                Ok(Some(w))
            }
            None => Ok(None),
        }
    }

    /// Keep only the active subcarriers of a time block (an ideal band-pass filter).
    ///
    /// # Errors
    /// As [`Ofdm::to_active`].
    pub fn filter(&self, block: &[C64]) -> Result<Vec<C64>> {
        self.to_time(&self.to_active(block)?)
    }

    /// Prepend the cyclic prefix to a time block.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] if the prefix is longer than the block.
    pub fn with_prefix(&self, block: &[C64]) -> Result<Vec<C64>> {
        let mut out = Vec::with_capacity(block.len() + self.oversampling * self.n_cp);
        extend_with_prefix(&mut out, block, self.oversampling * self.n_cp)?;
        Ok(out)
    }

    /// The raised-cosine window of whole blocks of [`Ofdm::block_len`] samples (prefix
    /// included): the first `ramp` samples of every prefix rise as
    ///
    /// $$
    /// w\[n\] = \frac{1}{2}\left(1 - \cos\frac{\pi (n + 1/2)}{R}\right), \qquad n = 0 \dots R - 1,
    /// $$
    ///
    /// a cyclic postfix of $R$ samples of the block falls as $1 - w$ and overlaps the
    /// ramp of the next block (the postfix of the last block wraps onto the first: the
    /// stream stays periodic), so the output has the length of the input and the FFT
    /// part of every block is untouched. The pulse of a block then lasts
    /// $L(N + N_{\mathrm{cp}}) + R$ samples with $(1 - \cos)$ edges: the raised-cosine
    /// impulse shape of windowed OFDM (Saul, *Proc. IEEE VTC2004-Fall*, 2004, eq. (1):
    /// a guard interval and a roll-off of 11.1 %), whose out-of-band sidelobes fall as
    /// $f^{-3}$ instead of the $f^{-1}$ of the rectangular pulse. `ramp = 0` returns the
    /// samples unchanged.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] if `ramp` exceeds the prefix $L N_{\mathrm{cp}}$,
    /// [`comnum_core::Error::Shape`] for a partial block.
    ///
    /// # Examples
    /// ```
    /// use comnum_waveform::ofdm::Ofdm;
    /// use comnum_math::C64;
    /// let ofdm = Ofdm { n_cp: 4, oversampling: 1, ..Ofdm::new(16, 16).unwrap() };
    /// let data: Vec<C64> = (0..32).map(|k| C64::new(if k % 3 == 0 { 1.0 } else { -1.0 }, 0.0)).collect();
    /// let x = ofdm.modulate(&data).unwrap();
    /// let y = ofdm.windowed(&x, 4).unwrap();
    /// assert_eq!(y.len(), x.len());
    /// // the FFT part of every block is untouched
    /// assert!(y[4..20].iter().zip(&x[4..20]).all(|(a, b)| (a - b).norm() < 1e-12));
    /// ```
    pub fn windowed(&self, samples: &[C64], ramp: usize) -> Result<Vec<C64>> {
        self.check()?;
        let prefix = self.oversampling * self.n_cp;
        ensure(ramp <= prefix, || format!("a ramp of {ramp} samples exceeds the prefix of {prefix}"))?;
        let len = self.block_len();
        ensure_shape(samples.len() % len == 0, || {
            format!("{} samples is not a multiple of the block length {len}", samples.len())
        })?;
        let mut out = samples.to_vec();
        if ramp == 0 || samples.is_empty() {
            return Ok(out);
        }
        let n_blocks = samples.len() / len;
        for b in 0..n_blocks {
            let o = b * len;
            // the postfix of the previous block: its FFT part continued cyclically
            let prev = (b + n_blocks - 1) % n_blocks * len + prefix;
            for n in 0..ramp {
                let w = 0.5 * (1.0 - (std::f64::consts::PI * (n as f64 + 0.5) / ramp as f64).cos());
                out[o + n] = samples[o + n] * w + samples[prev + n] * (1.0 - w);
            }
        }
        Ok(out)
    }

    /// Modulate data symbols (a multiple of $K$): $L (N + N_{\mathrm{cp}})$ samples per block.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] if the number of symbols is not a multiple of $K$, and the
    /// errors of [`Ofdm::check`].
    pub fn modulate(&self, data: &[C64]) -> Result<Vec<C64>> {
        Ok(self.modulate_with(data, &NoReduction)?.0)
    }

    /// Demodulate whole blocks into data symbols.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] if the number of samples is not a multiple of the block length.
    pub fn demodulate(&self, samples: &[C64]) -> Result<Vec<C64>> {
        let blocks = samples.len() / self.block_len().max(1);
        self.demodulate_with(samples, &NoReduction, &vec![0; blocks])
    }

    /// The PAPR (dB) of every whole block of `samples`, the cyclic prefix excluded
    /// ([`comnum_math::papr_db`] of the useful part).
    pub fn papr_db_per_block(&self, samples: &[C64]) -> Vec<f64> {
        let cp = self.oversampling * self.n_cp;
        samples.chunks_exact(self.block_len()).map(|b| comnum_math::papr_db(&b[cp..])).collect()
    }

    /// Modulate with a PAPR reduction technique: the samples and the side information
    /// of every block (the selected candidate of SLM or PTS, 0 otherwise).
    ///
    /// # Errors
    /// As [`Ofdm::modulate`], with the number of data symbols per block of the technique,
    /// and the errors of the technique.
    pub fn modulate_with(&self, data: &[C64], reduction: &dyn PaprReduction) -> Result<(Vec<C64>, Vec<usize>)> {
        self.check()?;
        let k = reduction.data_len(self);
        ensure(k >= 1, || "no data subcarrier".into())?;
        ensure_shape(data.len() % k == 0, || format!("{} symbols is not a multiple of {k} per block", data.len()))?;
        let mut out = Vec::with_capacity(data.len() / k * self.block_len());
        let mut side = Vec::with_capacity(data.len() / k);
        for chunk in data.chunks(k) {
            let (block, s) = reduction.reduce(self, chunk)?;
            extend_with_prefix(&mut out, &block, self.oversampling * self.n_cp)?;
            side.push(s);
        }
        Ok((out, side))
    }

    /// Demodulate blocks transmitted with a PAPR reduction technique and its side
    /// information.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] for a partial block or a side information of another length.
    pub fn demodulate_with(&self, samples: &[C64], reduction: &dyn PaprReduction, side: &[usize]) -> Result<Vec<C64>> {
        self.check()?;
        let len = self.block_len();
        ensure_shape(samples.len() % len == 0, || {
            format!("{} samples is not a multiple of the block length {len}", samples.len())
        })?;
        ensure_shape(side.len() == samples.len() / len, || {
            format!("{} side informations for {} blocks", side.len(), samples.len() / len)
        })?;
        let mut out = Vec::new();
        for (block, &s) in samples.chunks(len).zip(side) {
            let mut b = strip_prefix(block, self.oversampling * self.n_cp)?.to_vec();
            reduction.restore_time(&mut b);
            out.extend(reduction.recover(self, self.to_active(&b)?, s)?);
        }
        Ok(out)
    }
}

/// A PAPR reduction technique applied block by block by [`Ofdm::modulate_with`].
pub trait PaprReduction {
    /// Data symbols per block (all the active subcarriers by default).
    fn data_len(&self, ofdm: &Ofdm) -> usize {
        ofdm.n_active
    }

    /// The time block (without prefix) of one block of data, and its side information.
    ///
    /// # Errors
    /// For a configuration the technique cannot handle.
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)>;

    /// Undo a memoryless transformation of the time block at the receiver (the
    /// expansion of companding); nothing by default.
    fn restore_time(&self, _block: &mut [C64]) {}

    /// The data of a block from the values of its active subcarriers and its side
    /// information (despreading only, by default).
    ///
    /// # Errors
    /// For an invalid side information.
    fn recover(&self, ofdm: &Ofdm, active: Vec<C64>, _side: usize) -> Result<Vec<C64>> {
        Ok(ofdm.despread(&active))
    }
}

/// Plain OFDM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoReduction;

impl PaprReduction for NoReduction {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        Ok((ofdm.to_time(&ofdm.spread(data))?, 0))
    }
}

/// OFDM as a [`Modulate`] block: one stream of data symbols into one stream of samples
/// at [`Ofdm::sample_rate`].
impl Modulate for Ofdm {
    fn modulate_into(&self, x: &Symbols, out: &mut Signal) -> Result<()> {
        out.set_stream(&Ofdm::modulate(self, crate::one_stream(x, "OFDM")?)?, self.sample_rate);
        Ok(())
    }
}

/// OFDM as a [`Demodulate`] block: one stream of samples at [`Ofdm::sample_rate`].
impl Demodulate for Ofdm {
    /// # Errors
    /// [`comnum_core::Error::Config`] for several streams or a signal at another sample rate,
    /// and the errors of [`Ofdm::demodulate`].
    fn demodulate_into(&mut self, y: &Signal, out: &mut Symbols) -> Result<()> {
        check_input(y, self.sample_rate, "OFDM")?;
        out.set_stream(&Ofdm::demodulate(self, y.stream(0))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    /// Weights scale every subcarrier at the transmitter and are divided out at the
    /// receiver: the round trip returns the data, with and without DFT spreading.
    #[test]
    fn weighted_subcarriers_come_back_unweighted() {
        let mut rng = Rng::new(3);
        let w: Vec<C64> = (0..12).map(|_| rng.complex_gaussian()).collect();
        let data: Vec<C64> = (0..36).map(|_| rng.complex_gaussian()).collect();
        for dft_spread in [false, true] {
            let ofdm = Ofdm { n_cp: 4, dft_spread, weights: Some(w.clone()), ..Ofdm::new(16, 12).unwrap() };
            let back = ofdm.demodulate(&ofdm.modulate(&data).unwrap()).unwrap();
            assert!(back.iter().zip(&data).all(|(a, b)| (a - b).norm() < 1e-10));
        }
    }

    /// The weight $w_{k}$ multiplies subcarrier $k$ (read without weights), unit weights
    /// change nothing, and the power is the mean square of the weights.
    #[test]
    fn a_weight_multiplies_its_subcarrier_and_the_power_follows() {
        let plain = Ofdm::new(16, 8).unwrap();
        let w: Vec<C64> = (0..8).map(|i| C64::new(0.5 + 0.25 * i as f64, -0.1 * i as f64)).collect();
        let weighted = Ofdm { weights: Some(w.clone()), ..plain.clone() };
        let ones = Ofdm { weights: Some(vec![C64::new(1.0, 0.0); 8]), ..plain.clone() };
        let data = vec![C64::new(1.0, 0.0); 8];
        let x = weighted.modulate(&data).unwrap();
        let seen = plain.demodulate(&x).unwrap();
        assert!(seen.iter().zip(&w).all(|(a, b)| (a - b).norm() < 1e-12));
        assert_eq!(ones.modulate(&data).unwrap(), plain.modulate(&data).unwrap());
        let power = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
        let expected = w.iter().map(C64::norm_sqr).sum::<f64>() / 8.0;
        assert!((power - expected).abs() < 1e-12);
    }

    /// A zero weight leaves its subcarrier empty and comes back as 0; weights of another
    /// length or not finite are errors.
    #[test]
    fn a_zero_weight_switches_its_subcarrier_off() {
        let mut w = vec![C64::new(1.0, 0.0); 8];
        w[3] = C64::new(0.0, 0.0);
        let ofdm = Ofdm { weights: Some(w), ..Ofdm::new(16, 8).unwrap() };
        let data = vec![C64::new(1.0, 1.0); 8];
        let x = ofdm.modulate(&data).unwrap();
        assert!(Ofdm::new(16, 8).unwrap().demodulate(&x).unwrap()[3].norm() < 1e-12);
        let back = ofdm.demodulate(&x).unwrap();
        assert_eq!(back[3], C64::new(0.0, 0.0));
        assert!(back.iter().enumerate().filter(|(i, _)| *i != 3).all(|(_, v)| (v - data[0]).norm() < 1e-12));
        let short = Ofdm { weights: Some(vec![C64::new(1.0, 0.0); 7]), ..Ofdm::new(16, 8).unwrap() };
        assert!(short.check().is_err() && short.modulate(&data).is_err());
        let nan = Ofdm { weights: Some(vec![C64::new(f64::NAN, 0.0); 8]), ..Ofdm::new(16, 8).unwrap() };
        assert!(nan.check().is_err());
    }

    /// The window keeps the FFT part of every block, keeps the length, and the two ramps
    /// of an overlap sum to one; a too-long ramp and a partial block are errors.
    #[test]
    fn the_raised_cosine_window_overlaps_the_ramps_and_keeps_the_fft_part() {
        let ofdm = Ofdm { n_cp: 8, oversampling: 2, ..Ofdm::new(32, 32).unwrap() };
        let mut rng = Rng::new(7);
        let data: Vec<C64> = (0..96).map(|_| rng.complex_gaussian()).collect();
        let x = ofdm.modulate(&data).unwrap();
        let y = ofdm.windowed(&x, 12).unwrap();
        assert_eq!(y.len(), x.len());
        let len = ofdm.block_len();
        for b in 0..3 {
            let o = b * len;
            assert!(y[o + 16..o + len].iter().zip(&x[o + 16..o + len]).all(|(a, b)| (a - b).norm() < 1e-12));
            // a constant block would stay constant: the ramps sum to one
            let prev = (b + 2) % 3 * len + 16;
            for n in 0..12 {
                let w = 0.5 * (1.0 - (std::f64::consts::PI * (n as f64 + 0.5) / 12.0).cos());
                let expected = x[o + n] * w + x[prev + n] * (1.0 - w);
                assert!((y[o + n] - expected).norm() < 1e-12);
            }
        }
        assert_eq!(ofdm.windowed(&x, 0).unwrap(), x);
        assert!(ofdm.windowed(&x, 17).is_err());
        assert!(ofdm.windowed(&x[1..], 4).is_err());
    }

    /// A constant stream (every block the same, the data all ones on one subcarrier at
    /// DC) is unchanged by the window: the postfix of a block equals the prefix of the
    /// next.
    #[test]
    fn the_window_leaves_a_periodic_stream_unchanged() {
        let ofdm = Ofdm { n_cp: 4, oversampling: 1, dc_null: false, ..Ofdm::new(16, 1).unwrap() };
        let data = vec![C64::new(1.0, 0.0); 3];
        let x = ofdm.modulate(&data).unwrap();
        let y = ofdm.windowed(&x, 4).unwrap();
        assert!(y.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
    }

    /// The bins were computed from the signed subcarrier index before they came from
    /// `fft::centred_bin`: the same bins.
    #[test]
    fn the_bins_are_the_subcarriers_modulo_the_fft_size() {
        for (n_fft, n_active) in [(8, 4), (16, 7), (64, 64), (64, 63), (256, 200), (12, 1)] {
            for dc_null in [false, true] {
                for oversampling in [1, 4] {
                    let ofdm = Ofdm { oversampling, dc_null, ..Ofdm::new(n_fft, n_active).unwrap() };
                    if ofdm.check().is_err() {
                        continue;
                    }
                    for i in 0..n_active {
                        let old = ofdm.subcarrier(i).rem_euclid(ofdm.fft_len() as isize) as usize;
                        assert_eq!(ofdm.bin(i), old, "{n_fft} {n_active} {dc_null} {oversampling} {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn round_trip_unit_power_and_band_occupancy() {
        let mut rng = Rng::new(1);
        for (dc_null, spread) in [(false, false), (true, false), (false, true)] {
            let o = Ofdm { n_cp: 8, dc_null, dft_spread: spread, ..Ofdm::new(64, 48).unwrap() };
            let d: Vec<C64> = (0..48 * 200).map(|_| rng.complex_gaussian()).collect();
            let x = o.modulate(&d).unwrap();
            assert!(o.demodulate(&x).unwrap().iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-10));
            let p = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
            assert!((p - 1.0).abs() < 0.03, "power {p}");
            // nothing outside the active bins
            let block = &x[8 * 4..8 * 4 + o.fft_len()];
            let spec = fft::fft(block);
            for (b, v) in spec.iter().enumerate() {
                if !o.is_active_bin(b) {
                    assert!(v.norm() < 1e-9);
                }
            }
            if dc_null {
                assert!(!o.is_active_bin(0));
            }
        }
        assert!(Ofdm::new(64, 65).is_err() && Ofdm::new(64, 0).is_err());
    }

    #[test]
    fn the_traits_wrap_the_methods() {
        let o = Ofdm::new(32, 20).unwrap();
        let d = Symbols::from_stream(vec![C64::new(1.0, 0.0); 40]).unwrap();
        let x = Modulate::modulate(&o, &d).unwrap();
        assert_eq!((x.len(), x.sample_rate()), (2 * 128, Hertz(1.0)));
        let mut o2 = o.clone();
        let back = Demodulate::demodulate(&mut o2, &x).unwrap();
        assert!(back.as_slice().iter().all(|v| (v - C64::new(1.0, 0.0)).norm() < 1e-12));
        // a real rate: 15 kHz subcarriers, L N = 128 samples per symbol
        let mut lte = Ofdm { sample_rate: Hertz(128.0 * 15e3), ..o };
        let y = Modulate::modulate(&lte, &d).unwrap();
        assert_eq!(y.sample_rate(), Hertz(1.92e6));
        assert!(Demodulate::demodulate(&mut lte, &x).is_err() && Demodulate::demodulate(&mut lte, &y).is_ok());
        assert!(Ofdm { sample_rate: Hertz(0.0), ..o2 }.check().is_err());
    }
}
