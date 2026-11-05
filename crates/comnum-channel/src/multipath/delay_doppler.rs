//! The delay-Doppler channels of OTFS: [`DelayDoppler`] and the model that draws them,
//! [`DelayDopplerModel`].

use super::tdl::discretise;
use super::{Path, Taps};
use crate::fading::Fading;
use comnum_core::ensure;
use comnum_core::prelude::*;
use comnum_math::cis;
use std::f64::consts::PI;

/// A realisation of a doubly-dispersive channel of a few discrete paths, each with its
/// own delay and a single Doppler shift: the channel of OTFS (Raviteja et al. 2019,
/// eqs. (3)–(6); Hadani et al. 2017),
///
/// $$
/// r\[n\] = \sum_{i=1}^{P} h_{i}\, e^{j 2\pi \nu_{i} (n - l_{i}) / f_{s}}\, s\[n - l_{i}\],
/// $$
///
/// $n$ counted from the first sample of the signal. [`DelayDopplerModel`] draws it from
/// a power-delay profile; [`DelayDoppler::taps`] gives it as [`Taps`], which apply it.
///
/// # Examples
/// ```
/// use comnum_channel::multipath::{DelayDoppler, Path};
/// use comnum_math::C64;
/// // one path, 2 samples late, 1 kHz of Doppler at 1 MHz
/// let ch = DelayDoppler::new(vec![Path { gain: C64::new(1.0, 0.0), delay: 2, doppler_hz: 1e3 }], 1e6)?;
/// let x = vec![C64::new(1.0, 0.0); 8];
/// let y = ch.taps(8).apply(&[x])?.remove(0);
/// assert_eq!(y[1], C64::new(0.0, 0.0));
/// assert!((y[5] - comnum_math::cis(2.0 * std::f64::consts::PI * 1e3 * 3.0 / 1e6)).norm() < 1e-12);
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DelayDoppler {
    paths: Vec<Path>,
    sample_rate: f64,
}

impl DelayDoppler {
    /// The channel of `paths` at `sample_rate`.
    ///
    /// # Errors
    /// [`Error::Config`] for no path or a non-positive sampling rate.
    pub fn new(paths: Vec<Path>, sample_rate: f64) -> Result<DelayDoppler> {
        ensure(!paths.is_empty(), || "a channel needs at least one path".into())?;
        ensure(sample_rate > 0.0, || format!("the sampling rate must be positive, got {sample_rate}"))?;
        Ok(DelayDoppler { paths, sample_rate })
    }

    /// The paths.
    pub fn paths(&self) -> &[Path] {
        &self.paths
    }

    /// The sampling rate (Hz).
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// The realisation over `n_samples` samples (paths of the same delay summed into one
    /// tap).
    pub fn taps(&self, n_samples: usize) -> Taps {
        let mut delays: Vec<usize> = self.paths.iter().map(|p| p.delay).collect();
        delays.sort_unstable();
        delays.dedup();
        let coeffs = delays
            .iter()
            .map(|&d| {
                (0..n_samples)
                    .map(|n| {
                        self.paths
                            .iter()
                            .filter(|p| p.delay == d)
                            .map(|p| p.gain * cis(2.0 * PI * p.doppler_hz * (n as f64 - d as f64) / self.sample_rate))
                            .sum()
                    })
                    .collect()
            })
            .collect();
        Taps { delays, n_rx: 1, n_tx: 1, n_samples, coeffs, invariant: false }
    }
}

/// The model of [`DelayDoppler`] channels: one path per tap of a power-delay profile, of
/// Rayleigh gain $\mathcal{CN}(0, p_{i})$ and Doppler shift $\nu_{\max} \cos\theta_{i}$, $\theta_{i}$
/// uniform on $[-\pi, \pi)$ (Jakes' angle of arrival, as in Raviteja et al. 2019, Sec.
/// III-A). Unlike [`Tdl`](super::Tdl), whose every tap is a sum of many sinusoids, every path here is
/// one; single antenna.
///
/// # Examples
/// ```
/// use comnum_channel::multipath::{DelayDopplerModel, Profile};
/// use comnum_core::rng::Rng;
/// let model = DelayDopplerModel::new(&Profile::Eva.taps(), 444.0, 15.36e6)?;
/// let ch = model.draw(&mut Rng::new(1));
/// assert_eq!(ch.paths().len(), 9);
/// assert!(ch.paths().iter().all(|p| p.doppler_hz.abs() <= 444.0));
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DelayDopplerModel {
    taps: Vec<(usize, f64)>,
    /// Maximum Doppler shift $\nu_{\max}$ (Hz).
    pub max_doppler_hz: f64,
    sample_rate: f64,
}

impl DelayDopplerModel {
    /// The model of the taps `(delay in s, power in dB)` (powers normalised to a unit sum,
    /// delays rounded to the sampling grid) at `sample_rate`.
    ///
    /// # Errors
    /// [`Error::Config`] for no tap, a negative delay or Doppler shift, or a non-positive
    /// sampling rate.
    pub fn new(taps: &[(f64, f64)], max_doppler_hz: f64, sample_rate: f64) -> Result<DelayDopplerModel> {
        ensure(max_doppler_hz >= 0.0, || format!("the Doppler shift must be non-negative, got {max_doppler_hz}"))?;
        Ok(DelayDopplerModel { taps: discretise(taps, sample_rate)?, max_doppler_hz, sample_rate })
    }

    /// Sampling rate of the delays and Doppler shifts (Hz).
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Draw a realisation.
    pub fn draw(&self, rng: &mut SimRng) -> DelayDoppler {
        let paths = self
            .taps
            .iter()
            .map(|&(delay, p)| {
                let gain = rng.complex_gaussian() * p.sqrt();
                let theta = rng.phase();
                Path { gain, delay, doppler_hz: self.max_doppler_hz * theta.cos() }
            })
            .collect();
        DelayDoppler { paths, sample_rate: self.sample_rate }
    }
}

/// A single-antenna channel: the taps of a new [`DelayDopplerModel::draw`].
impl Fading for DelayDopplerModel {
    fn n_rx(&self) -> usize {
        1
    }

    fn n_tx(&self) -> usize {
        1
    }

    fn sample_rate(&self) -> Option<f64> {
        Some(self.sample_rate)
    }

    fn draw_taps(&self, n_samples: usize, rng: &mut SimRng) -> Result<Taps> {
        Ok(self.draw(rng).taps(n_samples))
    }
}

/// The channel of a new realisation on a single-antenna signal sampled at the rate of
/// the model ([`Error::Config`] otherwise).
impl Propagate for DelayDopplerModel {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        self.fade_into(x, rng, out).map(drop)
    }
}
