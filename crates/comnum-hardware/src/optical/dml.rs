//! A directly modulated semiconductor laser: the optical field of a power waveform with
//! the frequency chirp that the carrier-induced index change imposes on it (Agrawal,
//! *Fiber-Optic Communication Systems*, 3rd ed., 2002, Sec. 3.5.3, eqs. (3.5.16) and
//! (3.5.25); the adiabatic term after Koch & Bowers 1984).
//!
//! # Model
//!
//! The phase of the mode follows the gain through the linewidth enhancement factor
//! $\beta_{c}$ (eq. (3.5.16)); for a power $P(t)$ that follows the drive, the frequency
//! chirp $\delta\nu = \frac{1}{2\pi}\frac{d\phi}{dt}$ (eq. (3.5.25)) is the sum of a
//! transient term, which follows the logarithmic derivative of the power, and an
//! adiabatic term proportional to the power:
//!
//! $$
//! \delta\nu(t) = \frac{\beta_{c}}{4\pi}\left[\frac{d}{dt}\ln P(t) + \kappa P(t)\right],
//! $$
//!
//! with $\kappa$ (Hz/W) the adiabatic chirp coefficient (gain compression; 0 for the
//! transient chirp alone). The field is $E(t) = \sqrt{P(t)}\, e^{j\phi(t)}$ with
//! $\phi(t) = -2\pi \int \delta\nu\, dt$: the convention of the fibre models of comnum
//! ([`comnum_math::nlse::Medium::linear_response`], $e^{+j\beta_{2}\omega^{2}L/2}$, the
//! form of Agrawal's eq. (2.4.15) with his carrier $e^{-i\omega_{0} t}$), in which a
//! frequency increase is a phase that decreases (eq. (2.4.12)). The leading edge of a
//! pulse is thus shifted to higher frequencies and the trailing edge to lower ones, and
//! for a Gaussian pulse the transient chirp alone gives the chirped Gaussian of
//! [`direct_detection::gaussian_broadening_factor`](super::direct_detection::gaussian_broadening_factor)
//! with the chirp parameter $C = -\beta_{c}$ (Agrawal, Sec. 3.5.3, the sentence after
//! Fig. 3.22), which broadens from the start on standard fibre ($\beta_{2} C > 0$). The
//! modulation bandwidth of the laser is a Gaussian low-pass on the power (its 3 dB
//! bandwidth), none when zero; the relaxation oscillations are not modelled (the rate
//! equations would be).
//!
//! The power must stay positive: a finite extinction ratio, as a real laser biased
//! above threshold.
//!
//! # Examples
//! ```
//! use comnum_hardware::optical::DirectlyModulatedLaser;
//! // a Gaussian power pulse: the transient chirp is linear in time, (β_c / 2π) (-t/T0²)
//! let dml = DirectlyModulatedLaser { linewidth_enhancement: 5.0, ..DirectlyModulatedLaser::default() };
//! let (fs, t0) = (1e12, 50e-12);
//! let p: Vec<f64> = (0..400).map(|k| 1e-3 * (-((k as f64 - 200.0) / fs / t0).powi(2)).exp() + 1e-6).collect();
//! let chirp = dml.chirp_hz(&p, fs)?;
//! let t = (150.0 - 200.0) / fs;
//! let expected = -5.0 / (2.0 * std::f64::consts::PI) * t / (t0 * t0);
//! assert!((chirp[150] / expected - 1.0).abs() < 1e-2);
//! let e = dml.field(&p, fs)?;
//! assert!((e[200].norm_sqr() - p[200]).abs() < 1e-12);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002,
//!    Sec. 3.5.3, eqs. (3.5.16), (3.5.25).
//! 2. T. L. Koch, J. E. Bowers, "Nature of wavelength chirping in directly modulated
//!    semiconductor lasers," *Electron. Lett.* 20(25–26), 1038–1040, 1984.

use comnum_core::{Result, ensure};
use comnum_dsp::filter::{LowPass, fir_filter, low_pass_fir};
use comnum_math::{C64, cis};
use std::f64::consts::PI;

/// A directly modulated laser (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct DirectlyModulatedLaser {
    /// Linewidth enhancement factor $\beta_{c}$ (4 to 8 for the quaternary lasers of the 1.55 µm window; 0 for no chirp).
    pub linewidth_enhancement: f64,
    /// Adiabatic chirp coefficient $\kappa$ (Hz/W), 0 for the transient chirp alone.
    pub adiabatic_chirp_hz_w: f64,
    /// 3 dB modulation bandwidth (Hz) of the Gaussian low-pass on the power, 0 for none.
    pub bandwidth_hz: f64,
    /// Number of taps of that low-pass filter (made odd).
    pub n_taps: usize,
}

impl Default for DirectlyModulatedLaser {
    /// $\beta_{c} = 5$, no adiabatic chirp, no bandwidth limit, 255 taps.
    fn default() -> Self {
        DirectlyModulatedLaser { linewidth_enhancement: 5.0, adiabatic_chirp_hz_w: 0.0, bandwidth_hz: 0.0, n_taps: 255 }
    }
}

impl DirectlyModulatedLaser {
    /// Validate the parameters.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a non-finite factor or coefficient, or a
    /// negative bandwidth.
    pub fn check(&self) -> Result<()> {
        ensure(self.linewidth_enhancement.is_finite() && self.adiabatic_chirp_hz_w.is_finite(), || {
            "the chirp parameters must be finite".into()
        })?;
        ensure(self.bandwidth_hz >= 0.0 && self.bandwidth_hz.is_finite(), || {
            format!("the bandwidth {} Hz must be non-negative", self.bandwidth_hz)
        })
    }

    /// The power the laser emits for the drive `power_w` (W per sample) at `fs`: the
    /// low-pass of its modulation bandwidth, or the drive itself.
    ///
    /// # Errors
    /// As [`DirectlyModulatedLaser::check`]; [`comnum_core::Error::Config`] for a
    /// sampling rate below twice the bandwidth.
    pub fn emitted_power(&self, power_w: &[f64], fs: f64) -> Result<Vec<f64>> {
        self.check()?;
        ensure(fs > 0.0, || format!("the sampling rate {fs} Hz must be positive"))?;
        if self.bandwidth_hz == 0.0 {
            return Ok(power_w.to_vec());
        }
        ensure(fs >= 2.0 * self.bandwidth_hz, || {
            format!("the sampling rate {fs} Hz is below twice the bandwidth {} Hz", self.bandwidth_hz)
        })?;
        let h = low_pass_fir(self.bandwidth_hz, fs, self.n_taps | 1, LowPass::Gauss);
        let c: Vec<C64> = power_w.iter().map(|&v| C64::new(v, 0.0)).collect();
        Ok(fir_filter(&h, &c).into_iter().map(|v| v.re).collect())
    }

    /// The frequency chirp $\delta\nu$ (Hz) of every sample of the emitted power
    /// ([`DirectlyModulatedLaser::emitted_power`]), the logarithmic derivative by central
    /// differences (one-sided at the ends).
    ///
    /// # Errors
    /// As [`DirectlyModulatedLaser::emitted_power`]; [`comnum_core::Error::Config`] if
    /// the emitted power is not positive everywhere.
    pub fn chirp_hz(&self, power_w: &[f64], fs: f64) -> Result<Vec<f64>> {
        let p = self.emitted_power(power_w, fs)?;
        self.chirp_of(&p, fs)
    }

    fn chirp_of(&self, p: &[f64], fs: f64) -> Result<Vec<f64>> {
        ensure(p.iter().all(|&v| v > 0.0 && v.is_finite()), || {
            "the emitted power must be positive everywhere (a finite extinction ratio)".into()
        })?;
        let n = p.len();
        let lnp: Vec<f64> = p.iter().map(|v| v.ln()).collect();
        let k = self.linewidth_enhancement / (4.0 * PI);
        Ok((0..n)
            .map(|i| {
                let d = if n < 2 {
                    0.0
                } else if i == 0 {
                    (lnp[1] - lnp[0]) * fs
                } else if i == n - 1 {
                    (lnp[n - 1] - lnp[n - 2]) * fs
                } else {
                    (lnp[i + 1] - lnp[i - 1]) * fs / 2.0
                };
                k * (d + self.adiabatic_chirp_hz_w * p[i])
            })
            .collect())
    }

    /// The field $\sqrt{P}\, e^{j\phi}$ (√W per sample) of the drive `power_w` at `fs`:
    /// the emitted power with the phase of its chirp, $\phi_{k} = \phi_{k-1} - \pi
    /// (\delta\nu_{k-1} + \delta\nu_{k}) / f_{s}$, $\phi_{0} = 0$ (see the module for
    /// the sign).
    ///
    /// # Errors
    /// As [`DirectlyModulatedLaser::chirp_hz`].
    pub fn field(&self, power_w: &[f64], fs: f64) -> Result<Vec<C64>> {
        let p = self.emitted_power(power_w, fs)?;
        let chirp = self.chirp_of(&p, fs)?;
        let mut phase = 0.0;
        Ok(p.iter()
            .enumerate()
            .map(|(k, &pw)| {
                if k > 0 {
                    phase -= PI * (chirp[k - 1] + chirp[k]) / fs;
                }
                pw.sqrt() * cis(phase)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical::direct_detection::gaussian_broadening_factor;
    use comnum_math::fft;
    use comnum_math::nlse::Medium;

    fn gaussian_power(n: usize, fs: f64, t0: f64, floor: f64) -> Vec<f64> {
        (0..n).map(|k| (-((k as f64 - n as f64 / 2.0) / fs / t0).powi(2)).exp() + floor).collect()
    }

    fn rms_width(p: &[f64], fs: f64) -> f64 {
        let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
        for (k, &v) in p.iter().enumerate() {
            let t = k as f64 / fs;
            s0 += v;
            s1 += v * t;
            s2 += v * t * t;
        }
        (s2 / s0 - (s1 / s0).powi(2)).sqrt()
    }

    #[test]
    fn the_transient_chirp_of_a_gaussian_pulse_is_the_chirped_gaussian_of_minus_beta_c() {
        // a Gaussian power pulse of width T0 chirped by β_c, then 60 km of β₂ = -20 ps²/km:
        // its r.m.s. width grows by the broadening factor of the pulse of chirp C = -β_c
        // (β₂ C > 0: it broadens from the start, 3.4 times here, against 1.1 unchirped)
        let (n, fs, t0) = (4096, 2e12, 50e-12);
        let beta_c = 5.0;
        let dml = DirectlyModulatedLaser { linewidth_enhancement: beta_c, ..Default::default() };
        let p = gaussian_power(n, fs, t0, 1e-12);
        let e = dml.field(&p, fs).unwrap();
        let (beta2, l) = (-20e-24, 60.0);
        let m = Medium { beta2_s2_km: beta2, ..Medium::default() };
        let out = fft::filter(&e, m.linear_response(&fft::angular_frequencies(n, fs), l));
        let pout: Vec<f64> = out.iter().map(C64::norm_sqr).collect();
        let f_b = rms_width(&pout, fs) / rms_width(&p, fs);
        let expected = gaussian_broadening_factor(t0, -beta_c, beta2, 0.0, l).unwrap();
        assert!((f_b / expected - 1.0).abs() < 0.02, "{f_b} against {expected}");
        let wrong = gaussian_broadening_factor(t0, beta_c, beta2, 0.0, l).unwrap();
        assert!((f_b / wrong - 1.0).abs() > 0.5);
        // no chirp: the unchirped broadening
        let plain = DirectlyModulatedLaser { linewidth_enhancement: 0.0, ..Default::default() };
        let e0 = plain.field(&p, fs).unwrap();
        assert!(e0.iter().all(|v| v.im.abs() < 1e-12));
        let out0 = fft::filter(&e0, m.linear_response(&fft::angular_frequencies(n, fs), l));
        let f0 = rms_width(&out0.iter().map(C64::norm_sqr).collect::<Vec<_>>(), fs) / rms_width(&p, fs);
        let expected0 = gaussian_broadening_factor(t0, 0.0, beta2, 0.0, l).unwrap();
        assert!((f0 / expected0 - 1.0).abs() < 0.02, "{f0} against {expected0}");
    }

    #[test]
    fn the_adiabatic_chirp_follows_the_power_and_the_edges_are_blue_then_red() {
        let fs = 1e12;
        let p: Vec<f64> = (0..200).map(|k| if (50..150).contains(&k) { 2e-3 } else { 1e-4 }).collect();
        let dml =
            DirectlyModulatedLaser { linewidth_enhancement: 4.0, adiabatic_chirp_hz_w: 1e13, ..Default::default() };
        let chirp = dml.chirp_hz(&p, fs).unwrap();
        // flat top: the adiabatic term alone, (β_c / 4π) κ P
        let top = 4.0 / (4.0 * PI) * 1e13 * 2e-3;
        assert!((chirp[100] - top).abs() < 1e-6 * top);
        assert!(chirp[50] > chirp[100] && chirp[150] < chirp[100]);
        // the field carries the power and the modulation bandwidth smooths the edges
        let slow = DirectlyModulatedLaser { bandwidth_hz: 20e9, ..dml.clone() };
        let e = slow.field(&p, fs).unwrap();
        let emitted = slow.emitted_power(&p, fs).unwrap();
        assert!(e.iter().zip(&emitted).all(|(a, b)| (a.norm_sqr() - b).abs() < 1e-15));
        assert!(emitted[52] < 0.9 * 2e-3 && emitted[52] > p[49]);
        assert!(dml.chirp_hz(&[1e-3, 0.0, 1e-3], fs).is_err());
        assert!(DirectlyModulatedLaser { bandwidth_hz: -1.0, ..Default::default() }.check().is_err());
        assert!(slow.field(&p, 30e9).is_err());
    }
}
