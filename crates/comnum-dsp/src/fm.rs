//! Analogue frequency modulation: the modulator, the discriminator and the emphasis of
//! the modulating signal.
//!
//! # Model
//!
//! A message $m(t)$ frequency-modulates a carrier with the peak deviation $\Delta f$:
//! the complex envelope is $x(t) = e^{j 2\pi \Delta f \int^{t} m(u)\\, du}$, of
//! instantaneous frequency $\Delta f\\, m(t)$. In discrete time, [`fm_modulate`] sums the
//! phase increments $2\pi \Delta f\\, m\[n\] / f_{s}$ and [`fm_discriminator`] takes them
//! back, $f\[n\] = \frac{f_{s}}{2\pi} \arg\left(x\[n\]\\, x^{*}\[n - 1\]\right)$, the polar
//! discriminator: exact as long as the increments stay within $(-\pi, \pi)$, i.e. the
//! instantaneous frequency within $\pm f_{s}/2$, and insensitive to the amplitude of
//! the signal.
//!
//! # Emphasis
//!
//! The noise at the output of a discriminator grows with the frequency (its density as
//! $f^{2}$): the transmitter boosts the high frequencies of the message by $1 + j 2\pi f
//! \tau$ (pre-emphasis) and the receiver restores them by $1/(1 + j 2\pi f \tau)$
//! (de-emphasis), $\tau = 50$ µs in Europe and 75 µs in the United States (ITU-R
//! BS.450-4, Sec. 1.2). [`Emphasis`] implements the de-emphasis by the matched pole
//! $y\[n\] = a\\, y\[n - 1\] + (1 - a)\\, x\[n\]$, $a = e^{-1/(\tau f_{s})}$ (unit gain at
//! DC), and the pre-emphasis by its exact inverse, the FIR
//! $(x\[n\] - a\\, x\[n - 1\])/(1 - a)$, stable at any rate. At 228 kHz and 50 µs the de-emphasis is within 0.1 dB of
//! $1/|1 + j 2\pi f \tau|$ up to 15 kHz.
//!
//! # Examples
//! ```
//! use comnum_dsp::fm::{fm_discriminator, fm_modulate};
//! let fs = 228e3;
//! let m: Vec<f64> = (0..1000).map(|n| (2.0 * std::f64::consts::PI * 1e3 * n as f64 / fs).sin()).collect();
//! let x = fm_modulate(&m, fs, 75e3);
//! let f = fm_discriminator(&x, fs);
//! assert!((1..1000).all(|n| (f[n] - 75e3 * m[n]).abs() < 1e-6));
//! ```
//!
//! # References
//!
//! 1. Recommendation ITU-R BS.450-4, *Transmission standards for FM sound broadcasting
//!    at VHF*, 2019, Sec. 1.

use comnum_core::{Result, ensure};
use comnum_math::C64;
use std::f64::consts::PI;

/// The complex envelope $e^{j 2\pi \Delta f \sum_{k \le n} m\[k\] / f_{s}}$ of the message `m`
/// (at `fs`) with the peak deviation `deviation_hz` for $|m| = 1$ (see the module).
pub fn fm_modulate(m: &[f64], fs: f64, deviation_hz: f64) -> Vec<C64> {
    let k = 2.0 * PI * deviation_hz / fs;
    let mut phase = 0.0;
    m.iter()
        .map(|v| {
            phase += k * v;
            // keep the phase small: a long message would lose the digits of the increments
            phase = phase.rem_euclid(2.0 * PI);
            C64::from_polar(1.0, phase)
        })
        .collect()
}

/// The instantaneous frequency (Hz) of `x` (at `fs`) by the polar discriminator,
/// $\frac{f_{s}}{2\pi} \arg(x\[n\]\\, x^{*}\[n - 1\])$, 0 for the first sample (see the
/// module).
pub fn fm_discriminator(x: &[C64], fs: f64) -> Vec<f64> {
    let k = fs / (2.0 * PI);
    let mut out = Vec::with_capacity(x.len());
    let mut prev = x.first().copied().unwrap_or(C64::new(1.0, 0.0));
    for v in x {
        out.push((v * prev.conj()).arg() * k);
        prev = *v;
    }
    out
}

/// The pre-emphasis and the de-emphasis of time constant $\tau$ (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emphasis {
    /// The time constant $\tau$ (s): 50 µs in Europe, 75 µs in the United States.
    pub tau_s: f64,
}

impl Emphasis {
    /// The emphasis of Europe, $\tau = 50$ µs (ITU-R BS.450-4, Note 2).
    pub const EUROPE: Emphasis = Emphasis { tau_s: 50e-6 };
    /// The emphasis of the United States, $\tau = 75$ µs.
    pub const UNITED_STATES: Emphasis = Emphasis { tau_s: 75e-6 };

    /// The pole $a = e^{-1/(\tau f_{s})}$.
    fn pole(self, fs: f64) -> Result<f64> {
        ensure(self.tau_s > 0.0 && self.tau_s.is_finite() && fs > 0.0 && fs.is_finite(), || {
            format!("the emphasis needs a positive time constant and rate, got {} s at {fs} Hz", self.tau_s)
        })?;
        Ok((-1.0 / (self.tau_s * fs)).exp())
    }

    /// The de-emphasised signal $y\[n\] = a\\, y\[n - 1\] + (1 - a)\\, x\[n\]$, from rest.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a time constant or a rate that is not positive.
    pub fn deemphasize(self, x: &[f64], fs: f64) -> Result<Vec<f64>> {
        let a = self.pole(fs)?;
        let mut y = 0.0;
        Ok(x.iter()
            .map(|v| {
                y = a * y + (1.0 - a) * v;
                y
            })
            .collect())
    }

    /// The pre-emphasised signal $(x\[n\] - a\\, x\[n - 1\])/(1 - a)$, the inverse of
    /// [`Emphasis::deemphasize`] ($x\[-1\] = 0$).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a time constant or a rate that is not positive.
    pub fn preemphasize(self, x: &[f64], fs: f64) -> Result<Vec<f64>> {
        let a = self.pole(fs)?;
        let mut prev = 0.0;
        Ok(x.iter()
            .map(|v| {
                let y = (v - a * prev) / (1.0 - a);
                prev = *v;
                y
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_discriminator_inverts_the_modulator_whatever_the_amplitude() {
        let fs = 240e3;
        let m: Vec<f64> = (0..5000).map(|n| (0.013 * n as f64).sin() * 0.9 + 0.1).collect();
        let x: Vec<C64> = fm_modulate(&m, fs, 75e3).iter().map(|v| v * 0.37).collect();
        let f = fm_discriminator(&x, fs);
        assert_eq!(f.len(), m.len());
        assert_eq!(f[0], 0.0);
        assert!((1..m.len()).all(|n| (f[n] - 75e3 * m[n]).abs() < 1e-6));
        assert_eq!(fm_discriminator(&[], fs), Vec::<f64>::new());
    }

    #[test]
    fn the_de_emphasis_follows_the_rc_curve_and_the_pre_emphasis_undoes_it() {
        let fs = 228e3;
        let e = Emphasis::EUROPE;
        // the response at 15 kHz, from a long tone: within 0.1 dB of 1/|1 + j w tau|
        let f0 = 15e3;
        let x: Vec<f64> = (0..20_000).map(|n| (2.0 * PI * f0 * n as f64 / fs).cos()).collect();
        let y = e.deemphasize(&x, fs).unwrap();
        let rms = |v: &[f64]| (v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64).sqrt();
        let gain_db = 20.0 * (rms(&y[10_000..]) / rms(&x[10_000..])).log10();
        let rc_db = -10.0 * (1.0 + (2.0 * PI * f0 * 50e-6).powi(2)).log10();
        assert!((gain_db - rc_db).abs() < 0.1, "{gain_db} vs {rc_db}");
        let back = e.deemphasize(&e.preemphasize(&x, fs).unwrap(), fs).unwrap();
        assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-9));
        assert!(Emphasis { tau_s: 0.0 }.deemphasize(&x, fs).is_err());
        assert!(Emphasis::UNITED_STATES.preemphasize(&x, -1.0).is_err());
    }
}
