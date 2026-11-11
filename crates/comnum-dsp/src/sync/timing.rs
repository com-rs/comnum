//! Clock recovery (OptiCommPy `optic.dsp.clockRecovery`).
//!
//! The signal is at two samples per symbol. The Gardner timing-error detector compares
//! the sample between two symbols with the difference of the two symbols,
//!
//! `e = Re{x[1]* (x[2] - x[0])}` ([`gardner_ted`]), or, for Nyquist pulses,
//! `e = |x[1]|² (|x[0]|² - |x[2]|²)` ([`gardner_ted_nyquist`]),
//!
//! which is zero on average when the samples `x[0]`, `x[2]` sit on the symbol instants.
//! [`lagrange_cubic`] evaluates the cubic Lagrange (Farrow) interpolator of four
//! samples at a fractional position `μ ∈ [-1, 1]` (OptiCommPy `interpolator`), and
//! [`GardnerLoop`] closes the loop: every other output sample, the detector output goes
//! through a proportional-integral filter (`k_p`, `k_i`, [`crate::adaptive::ProportionalIntegral`]) that moves the fractional
//! sampling instant, with one sample skipped or repeated when it leaves `[-1, 1]`
//! (OptiCommPy `gardnerClockRecovery`).
//!
//! # Examples
//! ```
//! use comnum_dsp::interpolation::lagrange_cubic;
//! use comnum_dsp::sync::timing::gardner_ted;
//! use comnum_math::C64;
//! // a symmetric transition sampled at the right instants gives no error
//! let e = gardner_ted(&[C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(-1.0, 0.0)]);
//! assert_eq!(e, 0.0);
//! let x = [C64::new(1.0, 0.0), C64::new(2.0, 0.0), C64::new(3.0, 0.0), C64::new(4.0, 0.0)];
//! assert!((lagrange_cubic(&x, 0.0) - x[2]).norm() < 1e-15); // μ = 0 returns x[2]
//! ```

use crate::adaptive::{LoopFilter, ProportionalIntegral};
use crate::interpolation::lagrange_cubic;
use comnum_math::C64;

/// Gardner timing error of three consecutive samples at two samples per symbol.
pub fn gardner_ted(x: &[C64; 3]) -> f64 {
    (x[1].conj() * (x[2] - x[0])).re
}

/// The S-curve of Gardner's detector (Gardner 1986, eqs. (15) and (29)): the mean of
/// [`gardner_ted`] on a real raised-cosine signal of roll-off $\beta$ = `rolloff` and
/// i.i.d. symbols $\pm 1$, sampled at $\tau + qT$ and $\tau + (q - \tfrac12)T$, with
/// $\tau$ = `tau` in symbols:
/// $$U(\tau) = \frac{4}{T} G_{R} \sin\frac{2\pi\tau}{T}, \qquad G_{R} = \int_{0}^{1/T} G(f)\, G\!\left(\frac1T - f\right) \sin(\pi f T)\, df,$$
/// $G$ the raised-cosine spectrum of unit value at $f = 0$ (the pulse of peak $g(0) = 1$),
/// $T = 1$. The integrand lives on the overlap $|f - \tfrac{1}{2T}| \le \tfrac{\beta}{2T}$
/// and is integrated by Gauss–Legendre with 64 nodes. The paper prints eq. (29) with a
/// minus sign; the sum (15) as printed gives the plus sign used here, the slope of
/// [`gardner_ted`] at $\tau = 0$ being positive.
///
/// # Examples
/// ```
/// use comnum_dsp::sync::timing::{gardner_gain, gardner_s_curve};
/// // full roll-off: G_R = 1/(3 pi), a gain of 8/3
/// assert!((gardner_gain(1.0) - 8.0 / 3.0).abs() < 1e-12);
/// assert!(gardner_s_curve(0.5, 0.0).abs() < 1e-15);
/// ```
pub fn gardner_s_curve(rolloff: f64, tau: f64) -> f64 {
    4.0 * gardner_overlap(rolloff) * (2.0 * std::f64::consts::PI * tau).sin()
}

/// The gain of Gardner's detector at zero timing error, $dU/d\tau = 8\pi G_{R} / T^{2}$
/// ([`gardner_s_curve`]; it vanishes as the roll-off goes to zero, Gardner 1986, after
/// eq. (29)).
pub fn gardner_gain(rolloff: f64) -> f64 {
    8.0 * std::f64::consts::PI * gardner_overlap(rolloff)
}

/// $G_{R}$ of [`gardner_s_curve`] for the raised cosine of roll-off `rolloff` and $T = 1$.
fn gardner_overlap(rolloff: f64) -> f64 {
    let b = rolloff.clamp(0.0, 1.0);
    if b == 0.0 {
        return 0.0;
    }
    let (lo, hi) = ((1.0 - b) / 2.0, (1.0 + b) / 2.0);
    let (x, w) = comnum_math::gauss_legendre(64);
    x.iter()
        .zip(&w)
        .map(|(&x, &w)| {
            let f = lo + (hi - lo) * (x + 1.0) / 2.0;
            let g = |f: f64| crate::filter::raised_cosine_spectrum(f, 1.0, b);
            w * g(f) * g(1.0 - f) * (std::f64::consts::PI * f).sin()
        })
        .sum::<f64>()
        * (hi - lo)
        / 2.0
}

/// Gardner timing error for Nyquist pulses (power version).
pub fn gardner_ted_nyquist(x: &[C64; 3]) -> f64 {
    x[1].norm_sqr() * (x[0].norm_sqr() - x[2].norm_sqr())
}

/// The Gardner clock-recovery loop.
#[derive(Clone, Debug, PartialEq)]
pub struct GardnerLoop {
    /// Proportional gain of the loop filter.
    pub kp: f64,
    /// Integral gain of the loop filter.
    pub ki: f64,
    /// Use the Nyquist (power) detector.
    pub nyquist: bool,
    /// Largest clock deviation handled (ppm): sizes the output.
    pub max_ppm: f64,
}

impl Default for GardnerLoop {
    /// `k_p = 10⁻³`, `k_i = 10⁻⁶`, Nyquist detector, 500 ppm (OptiCommPy defaults).
    fn default() -> Self {
        GardnerLoop { kp: 1e-3, ki: 1e-6, nyquist: true, max_ppm: 500.0 }
    }
}

/// The output of [`GardnerLoop::run`].
#[derive(Clone, Debug, PartialEq)]
pub struct ClockRecoveryOutput {
    /// Resampled signal at two samples per symbol.
    pub signal: Vec<C64>,
    /// Fractional sampling instant of every output sample.
    pub timing: Vec<f64>,
}

impl GardnerLoop {
    /// Recover the clock of a signal at (about) two samples per symbol.
    ///
    /// The error of the detector grows as the power of the signal (its square for the
    /// Nyquist detector): the gains assume a signal of about unit power.
    ///
    /// # Errors
    /// [`comnum_core::Error::Numerical`] for a loop that diverges (a timing that is no
    /// longer finite, or that stops advancing in the signal): a signal far from unit
    /// power, or gains too large.
    pub fn run(&self, x: &[C64]) -> comnum_core::Result<ClockRecoveryOutput> {
        let mut sig = x.to_vec();
        sig.push(C64::new(0.0, 0.0));
        let n_samples = sig.len();
        let ln = ((1.0 - self.max_ppm / 1e6) * n_samples as f64) as usize;
        let mut out = vec![C64::new(0.0, 0.0); ln];
        let mut timing = vec![0.0; ln];
        let mut filter = ProportionalIntegral::new(self.kp, self.ki);
        let mut t_nco = 0.0f64;
        let (mut n, mut m) = (2usize, 2usize);
        let mut last = 0;
        // a locked loop takes one iteration per sample, and a few more for the samples
        // it takes again: one that takes four times as many no longer advances
        let mut budget = 4 * n_samples;
        let diverged = |t: f64| {
            comnum_core::Error::Numerical(format!(
                "the clock recovery diverges (timing {t:e} samples): its gains assume a signal of about unit power"
            ))
        };
        while n + 1 < ln && m + 2 < n_samples {
            // a locked loop keeps its timing within a sample of zero
            if !t_nco.is_finite() || t_nco.abs() > 8.0 || budget == 0 {
                return Err(diverged(t_nco));
            }
            budget -= 1;
            out[n] = lagrange_cubic(&[sig[m - 2], sig[m - 1], sig[m], sig[m + 1]], t_nco);
            if n % 2 == 0 {
                let w = [out[n - 2], out[n - 1], out[n]];
                let ted = if self.nyquist { gardner_ted_nyquist(&w) } else { gardner_ted(&w) };
                t_nco -= filter.update(ted);
            }
            if t_nco > 1.0 {
                t_nco -= 1.0;
                // an output sample is taken again; never before the first two, which the
                // detector needs (a large error at the start would reach them)
                n = n.saturating_sub(1).max(2);
            } else if t_nco < -1.0 {
                t_nco += 1.0;
                n += 2;
                m += 1;
            } else {
                n += 1;
                m += 1;
            }
            if n < ln {
                timing[n] = t_nco;
            }
            last = last.max(n);
        }
        out.truncate(last.min(ln));
        timing.truncate(last.min(ln));
        if !t_nco.is_finite() || t_nco.abs() > 8.0 {
            return Err(diverged(t_nco));
        }
        Ok(ClockRecoveryOutput { signal: out, timing })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{PulseShape, fir_filter, pulse_shape};
    use crate::resample::{delay_fft, upsample};
    use comnum_core::rng::Rng;

    #[test]
    fn a_loud_signal_does_not_drive_the_loop_before_its_first_samples() {
        // the error of the detector grows as the power: 10⁴ makes it jump back at once
        let x: Vec<C64> = (0..400).map(|i| C64::new(if (i / 2) % 3 == 0 { 1e4 } else { -1e4 }, 0.0)).collect();
        for nyquist in [true, false] {
            // no panic: an output no longer than the input, or the error of a loop that diverges
            if let Ok(out) = (GardnerLoop { nyquist, ..GardnerLoop::default() }).run(&x) {
                assert!(out.signal.len() <= x.len());
            }
        }
    }

    #[test]
    fn a_loop_that_diverges_is_an_error_and_returns() {
        // noise far from unit power: the loop used to return NaN samples (amplitudes of
        // 5 to 20) or never to return (40 and more, the timing pinned on its first samples)
        let mut r = Rng::new(3);
        let noise: Vec<C64> = (0..20_000).map(|_| r.complex_gaussian()).collect();
        for amplitude in [1.0, 2.0] {
            let x: Vec<C64> = noise.iter().map(|v| v * amplitude).collect();
            let out = GardnerLoop::default().run(&x).unwrap();
            assert!(out.signal.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
        }
        for amplitude in [5.0, 10.0, 20.0, 40.0, 100.0, 1000.0, 1e160] {
            let x: Vec<C64> = noise.iter().map(|v| v * amplitude).collect();
            let e = GardnerLoop::default().run(&x).unwrap_err();
            assert!(matches!(e, comnum_core::Error::Numerical(_)), "{amplitude}: {e}");
        }
        // gains that are too large
        let x: Vec<C64> = noise.iter().take(2000).copied().collect();
        assert!(GardnerLoop { kp: 10.0, ki: 1e9, ..GardnerLoop::default() }.run(&x).is_err());
    }

    #[test]
    fn gardner_loop_locks_on_a_delayed_signal() {
        let mut r = Rng::new(1);
        let s: Vec<C64> = (0..4000)
            .map(|_| C64::new(if r.bit() == 1 { 1.0 } else { -1.0 }, if r.bit() == 1 { 1.0 } else { -1.0 }))
            .collect();
        let h = pulse_shape(PulseShape::Rrc { rolloff: 0.5 }, 2, 64).unwrap();
        let mut x = fir_filter(&h, &upsample(&s, 2));
        x = fir_filter(&h, &x);
        let x = delay_fft(&x, 0.4, 1.0); // 0.4 sample timing offset
        let out = GardnerLoop { kp: 5e-3, ki: 1e-5, nyquist: false, ..GardnerLoop::default() }.run(&x).unwrap();
        // the loop converges to a constant fractional instant
        let t = &out.timing[6000..7000];
        let mean = t.iter().sum::<f64>() / t.len() as f64;
        let var = t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / t.len() as f64;
        assert!(var < 1e-3, "timing variance {var}");
    }
}
