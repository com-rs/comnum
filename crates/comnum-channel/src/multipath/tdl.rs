//! The tapped delay line with Doppler, [`Tdl`], and its parameters, [`TdlParams`].

use super::{LOS_DOPPLER_RATIO, Taps};
use crate::fading::Fading;
use comnum_core::ensure;
use comnum_core::prelude::*;
use comnum_math::cis;
use comnum_math::db2lin;
use std::f64::consts::PI;

/// A tapped-delay-line channel with Clarke/Jakes Doppler (see the module).
///
/// Its delays are put on the sampling grid of the rate given at construction, which
/// [`Tdl::sample_rate`] returns: the channel applies only to signals of that rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Tdl {
    pub(super) delays: Vec<usize>,
    pub(super) powers: Vec<f64>,
    /// Maximum Doppler shift $f_{D}$ (Hz); 0 for a channel constant over a call.
    pub doppler_hz: f64,
    pub(super) sample_rate: f64,
    /// Receive antennas.
    pub n_rx: usize,
    /// Transmit antennas.
    pub n_tx: usize,
    /// Sinusoids per tap of the sum-of-sinusoids generator (at least 1; 0 is read as 1).
    pub sinusoids: usize,
    /// The specular component: the index of its tap in [`Tdl::delays`] and its normalised
    /// power.
    pub(super) specular: Option<(usize, f64)>,
}

/// The parameters of a [`Tdl`]: its taps, Doppler shift, sampling rate and antennas.
///
/// # Examples
/// ```
/// use comnum_channel::multipath::{Profile, Tdl, TdlParams};
/// let p = TdlParams { taps: Profile::Eva.taps(), doppler_hz: 70.0, sample_rate: 15.36e6, ..TdlParams::default() };
/// let ch = Tdl::new(&p)?;
/// assert_eq!((ch.n_rx, ch.n_tx, ch.delays().last()), (1, 1, Some(&39)));      // 2.51 µs at 15.36 MHz
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TdlParams {
    /// The taps `(delay in s, power in dB)` (powers of any normalisation).
    pub taps: Vec<(f64, f64)>,
    /// The specular component `(delay in s, power in dB)` of a Ricean tap, in the scale
    /// of `taps`, of Doppler shift [`LOS_DOPPLER_RATIO`] times `doppler_hz`; `None` for
    /// Rayleigh taps only.
    pub specular: Option<(f64, f64)>,
    /// Maximum Doppler shift $f_{D}$ (Hz); 0 for a channel constant over a call.
    pub doppler_hz: f64,
    /// Sampling rate of the delays and of the Doppler spectrum (Hz).
    pub sample_rate: f64,
    /// Receive antennas.
    pub n_rx: usize,
    /// Transmit antennas.
    pub n_tx: usize,
}

impl Default for TdlParams {
    /// One tap of 0 dB at delay 0 without Doppler, at 1 Hz, one antenna on each side.
    fn default() -> TdlParams {
        TdlParams { taps: vec![(0.0, 0.0)], specular: None, doppler_hz: 0.0, sample_rate: 1.0, n_rx: 1, n_tx: 1 }
    }
}

impl TdlParams {
    /// Check the parameters.
    ///
    /// # Errors
    /// [`Error::Config`] for no taps, a negative delay, a non-positive rate, a negative
    /// Doppler shift or no antennas.
    pub fn check(&self) -> Result<()> {
        self.grid().map(drop)
    }

    /// The taps on the sampling grid, in their order, then the specular component; the
    /// powers normalised to a unit total, the specular component included.
    fn grid(&self) -> Result<Vec<(usize, f64)>> {
        let (doppler_hz, n_rx, n_tx) = (self.doppler_hz, self.n_rx, self.n_tx);
        ensure(doppler_hz >= 0.0, || format!("the Doppler shift must be non-negative, got {doppler_hz}"))?;
        ensure(n_rx >= 1 && n_tx >= 1, || format!("at least one antenna on each side, got {n_rx} × {n_tx}"))?;
        let mut taps = self.taps.clone();
        taps.extend(self.specular);
        discretise(&taps, self.sample_rate)
    }
}

impl Tdl {
    /// The channel of the parameters `params`, its taps put on the sampling grid (taps
    /// falling on the same sample merged), with 16 sinusoids per tap.
    ///
    /// # Errors
    /// As [`TdlParams::check`].
    pub fn new(params: &TdlParams) -> Result<Tdl> {
        let mut grid = params.grid()?;
        let specular = params.specular.map(|_| grid.pop().unwrap_or_default());
        let mut merged: Vec<(usize, f64)> = Vec::new();
        if let Some((s, _)) = specular {
            merged.push((s, 0.0));
        }
        for (s, p) in grid {
            match merged.iter_mut().find(|(x, _)| *x == s) {
                Some(m) => m.1 += p,
                None => merged.push((s, p)),
            }
        }
        merged.sort_by_key(|m| m.0);
        let delays: Vec<usize> = merged.iter().map(|m| m.0).collect();
        let specular = specular.map(|(s, p)| (delays.iter().position(|d| *d == s).unwrap_or(0), p));
        Ok(Tdl {
            delays,
            powers: merged.iter().map(|m| m.1).collect(),
            doppler_hz: params.doppler_hz,
            sample_rate: params.sample_rate,
            n_rx: params.n_rx,
            n_tx: params.n_tx,
            sinusoids: 16,
            specular,
        })
    }

    /// Delays of the taps on the sampling grid (samples).
    pub fn delays(&self) -> &[usize] {
        &self.delays
    }

    /// Sampling rate of the delays and of the Doppler spectrum (Hz).
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Normalised powers of the Rayleigh part of the taps (with [`Tdl::specular`], a unit
    /// total).
    pub fn powers(&self) -> &[f64] {
        &self.powers
    }

    /// The specular component: the index of its tap in [`Tdl::delays`] and its normalised
    /// power.
    pub fn specular(&self) -> Option<(usize, f64)> {
        self.specular
    }

    /// The frequency correlation $R(\Delta k) = \sum_{l} p_{l} e^{-j 2\pi \Delta k d_{l} / N}$
    /// of the channel between two subcarriers `delta` bins apart on a grid of `fft_len`
    /// (`comnum_dsp::estimation::frequency_correlation`).
    pub fn frequency_correlation(&self, delta: isize, fft_len: usize) -> C64 {
        let delays: Vec<f64> = self.delays.iter().map(|&d| d as f64).collect();
        let mut powers = self.powers.clone();
        if let Some((i, p)) = self.specular {
            powers[i] += p;
        }
        comnum_dsp::estimation::frequency_correlation(&delays, &powers, delta as f64, fft_len)
    }

    /// Draw the taps for `n_samples` samples (stored once for all of them without
    /// Doppler, [`Taps::is_time_invariant`]).
    pub fn draw(&self, n_samples: usize, rng: &mut SimRng) -> Taps {
        self.draw_with(n_samples, rng, |n| n as f64)
    }

    /// The taps at the instants `times_s` (s, any order and spacing) of one realisation:
    /// sample `n` of the result is the channel at `times_s[n]` (the sums of sinusoids of
    /// [`Tdl::draw`] evaluated there). For a channel applied in the frequency domain to
    /// OFDM symbols, at the times of their samples.
    pub fn draw_at(&self, times_s: &[f64], rng: &mut SimRng) -> Taps {
        self.draw_with(times_s.len(), rng, |n| times_s[n] * self.sample_rate)
    }

    /// [`Tdl::draw`] at the instants `t(n)` (in samples) of the `count` samples.
    fn draw_with(&self, count: usize, rng: &mut SimRng, t: impl Fn(usize) -> f64) -> Taps {
        let n_samples = count;
        let mut taps = self.draw_rayleigh(n_samples, rng, &t);
        if let Some((i, p)) = self.specular {
            // the specular component: constant amplitude, Doppler f_S = 0.7 f_D, a uniform
            // phase per link (TR 38.901 §7.7.2)
            let links = self.n_rx * self.n_tx;
            let w = 2.0 * PI * LOS_DOPPLER_RATIO * self.doppler_hz / self.sample_rate;
            let a = p.sqrt();
            let phases: Vec<f64> = (0..links).map(|_| rng.phase()).collect();
            let rows = if taps.invariant { 1 } else { n_samples };
            for n in 0..rows {
                for (link, phi) in phases.iter().enumerate() {
                    taps.coeffs[i][n * links + link] += cis(w * t(n) + phi) * a;
                }
            }
        }
        taps
    }

    /// The Rayleigh part of [`Tdl::draw`].
    fn draw_rayleigh(&self, n_samples: usize, rng: &mut SimRng, time: &impl Fn(usize) -> f64) -> Taps {
        let m = self.sinusoids.max(1);
        let links = self.n_rx * self.n_tx;
        if self.doppler_hz == 0.0 {
            let coeffs = self.powers.iter().map(|&p| (0..links).map(|_| rng.complex_gaussian() * p.sqrt()).collect());
            return Taps {
                delays: self.delays.clone(),
                n_rx: self.n_rx,
                n_tx: self.n_tx,
                n_samples,
                coeffs: coeffs.collect(),
                invariant: true,
            };
        }
        let coeffs = self
            .powers
            .iter()
            .map(|&p| {
                let mut c = vec![C64::new(0.0, 0.0); n_samples * links];
                for link in 0..links {
                    // Zheng & Xiao: angles (2πi - π + θ)/(4M), independent phases on I and Q
                    let theta = rng.phase();
                    let comps: Vec<(f64, f64, f64)> = (0..m)
                        .map(|i| {
                            let alpha = (2.0 * PI * (i + 1) as f64 - PI + theta) / (4.0 * m as f64);
                            (alpha.cos(), rng.phase(), rng.phase())
                        })
                        .collect();
                    let scale = (p / m as f64).sqrt();
                    let w = 2.0 * PI * self.doppler_hz / self.sample_rate;
                    for n in 0..n_samples {
                        let t = time(n);
                        let (mut re, mut im) = (0.0, 0.0);
                        for (ca, phi, psi) in &comps {
                            re += (w * t * ca + phi).cos();
                            im += (w * t * ca + psi).cos();
                        }
                        c[n * links + link] = C64::new(re, im) * scale;
                    }
                }
                c
            })
            .collect();
        Taps { delays: self.delays.clone(), n_rx: self.n_rx, n_tx: self.n_tx, n_samples, coeffs, invariant: false }
    }
}

/// The delays (samples) and normalised powers of the taps `(delay in s, power in dB)` of
/// a profile at the sampling rate `sample_rate`, in their order (taps falling on the same
/// sample are not merged).
pub(super) fn discretise(taps: &[(f64, f64)], sample_rate: f64) -> Result<Vec<(usize, f64)>> {
    ensure(!taps.is_empty(), || "a channel needs at least one tap".into())?;
    ensure(sample_rate > 0.0, || format!("the sampling rate must be positive, got {sample_rate}"))?;
    ensure(taps.iter().all(|(d, _)| *d >= 0.0), || "negative delay".into())?;
    let total: f64 = taps.iter().map(|(_, db)| db2lin(*db)).sum();
    Ok(taps.iter().map(|&(d, db)| ((d * sample_rate).round() as usize, db2lin(db) / total)).collect())
}

/// The taps of [`Tdl::draw`].
impl Fading for Tdl {
    fn n_rx(&self) -> usize {
        self.n_rx
    }

    fn n_tx(&self) -> usize {
        self.n_tx
    }

    fn sample_rate(&self) -> Option<f64> {
        Some(self.sample_rate)
    }

    fn draw_taps(&self, n_samples: usize, rng: &mut SimRng) -> Result<Taps> {
        Ok(self.draw(n_samples, rng))
    }
}

impl Propagate for Tdl {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        self.fade_into(x, rng, out).map(drop)
    }
}
