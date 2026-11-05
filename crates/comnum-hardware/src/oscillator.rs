//! Oscillator impairments: Wiener phase noise, carrier frequency offset and sampling
//! clock offset.
//!
//! - [`phase_noise`]: the Wiener process of an oscillator (or a laser) of linewidth
//!   `Δν` (OptiCommPy `phaseNoise`); [`wiener_phase`]: the same for a given increment
//!   standard deviation and [`PhaseOrigin`];
//! - [`frequency_offset`]: the rotation of a constant frequency offset, from the first
//!   sample or from the middle of the block;
//! - [`sampling_offset`]: the samples of a receiver whose clock runs at $1/T'$ instead
//!   of $1/T$, $T' = (1 + \zeta) T$;
//! - [`PhaseNoise`], [`Cfo`] and [`ClockOffset`]: the [`Impair`] blocks that rotate every
//!   stream of a [`Signal`] by the phase noise or by a constant frequency offset, or
//!   resample it with a clock offset.
//!
//! # Model
//!
//! $$
//! \phi_{n+1} = \phi_{n} + \Delta_{n}, \qquad \Delta_{n} \sim \mathcal{N}(0, 2\pi \Delta\nu T_{s}),
//! \qquad \phi_{0} = 0
//! $$
//!
//! ([`PhaseOrigin::Step`] draws $\phi_{0} = \Delta_{0}$ instead: the first sample is
//! rotated too), and a frequency offset $\omega = 2\pi \Delta f / f_{s}$ rotates sample
//! $n$ by $e^{j\omega (n - n_{c})}$, with $n_{c} = 0$, or $n_{c} = N/2$ for a block of
//! $N$ samples whose phase is zero at its middle.
//!
//! A sampling clock offset $\zeta = (T' - T)/T$ (a relative error of the clock of the
//! converter: $10^{-4}$ for 100 ppm) takes sample $m$ of the receiver at the instant
//! $m (1 + \zeta) T$ of the transmitted signal (Speth et al. 1999, eqs. (12)–(13)),
//!
//! $$
//! y\[m\] = x\big(m (1 + \zeta)\big),
//! $$
//!
//! interpolated between the samples of $x$ by the cubic Lagrange (Farrow) kernel of
//! [`comnum_dsp::interpolation::lagrange_cubic`], zero beyond the ends. The kernel is
//! exact on cubics: its error falls with the oversampling of the signal (about
//! $10^{-3}$ of the power for an OFDM signal oversampled by 4). In an OFDM receiver the
//! offset rotates subcarrier $k$ by $2\pi \zeta k$ per useful duration and slides the FFT
//! window by $\zeta$ samples per sample (their eqs. (36)–(40)).
//!
//! # Examples
//! ```
//! use comnum_core::prelude::*;
//! use comnum_hardware::oscillator::{Cfo, PhaseNoise};
//! let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 1000], Hertz(1e9))?;
//! let y = PhaseNoise { linewidth: Hertz(1e5) }.impair(&x, &mut batch_rng(1, 0))?;
//! assert!(y.stream(0).iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));   // constant envelope
//! let z = Cfo { offset: Hertz(1e6), centred: false }.impair(&x, &mut batch_rng(1, 0))?;
//! assert!((z.stream(0)[250].arg() - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
//! # Ok::<(), comnum_core::Error>(())
//! ```

use comnum_core::ensure;
use comnum_core::prelude::*;
use comnum_math::cis;
use std::f64::consts::PI;

/// Wiener phase noise of linewidth `linewidth_hz` over `n` samples spaced by `ts_s`
/// seconds (radians, `φ[0] = 0`): [`wiener_phase`] with the increments of
/// [`increment_sd`] at $f_{s} = 1/T_{s}$.
///
/// # Errors
/// [`Error::Config`] for a negative or non-finite linewidth, or a sample spacing that is
/// not positive and finite.
pub fn phase_noise(linewidth_hz: f64, n: usize, ts_s: f64, rng: &mut Rng) -> Result<Vec<f64>> {
    ensure(ts_s > 0.0 && ts_s.is_finite(), || format!("the sample spacing must be positive, got {ts_s} s"))?;
    let sd = increment_sd(linewidth_hz, 1.0 / ts_s)?;
    Ok(wiener_phase(sd, n, PhaseOrigin::Zero, rng))
}

/// The standard deviation $\sqrt{2\pi \Delta\nu / f_{s}}$ of the increments of the
/// Wiener phase of linewidth `linewidth_hz` sampled at `fs`.
///
/// # Errors
/// [`Error::Config`] for a negative or non-finite linewidth, or a sampling rate that is
/// not positive and finite.
pub fn increment_sd(linewidth_hz: f64, fs: f64) -> Result<f64> {
    ensure(linewidth_hz >= 0.0 && linewidth_hz.is_finite(), || {
        format!("the linewidth must be non-negative, got {linewidth_hz} Hz")
    })?;
    ensure(fs > 0.0 && fs.is_finite(), || format!("the sampling rate must be positive, got {fs} Hz"))?;
    Ok((2.0 * PI * linewidth_hz / fs).sqrt())
}

/// Where a Wiener phase starts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PhaseOrigin {
    /// $\phi_{0} = 0$: the first sample is not rotated (OptiCommPy `phaseNoise`).
    #[default]
    Zero,
    /// $\phi_{0} = \Delta_{0}$: an increment is drawn before the first sample too, which
    /// is rotated.
    Step,
}

impl std::str::FromStr for PhaseOrigin {
    type Err = Error;

    /// `zero` or `step` (case-insensitive).
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(s: &str) -> Result<PhaseOrigin> {
        match s.to_ascii_lowercase().as_str() {
            "zero" => Ok(PhaseOrigin::Zero),
            "step" => Ok(PhaseOrigin::Step),
            _ => Err(Error::Config(format!("the phase origin must be zero or step, got {s:?}"))),
        }
    }
}

/// A Wiener phase over `n` samples with increments of standard deviation `sd` (radians),
/// starting at `origin`: one Gaussian draw per increment, sample after sample.
pub fn wiener_phase(sd: f64, n: usize, origin: PhaseOrigin, rng: &mut Rng) -> Vec<f64> {
    let step = origin == PhaseOrigin::Step;
    let mut phi = Vec::with_capacity(n);
    let mut p = 0.0;
    for k in 0..n {
        if k > 0 || step {
            p += sd * rng.gaussian();
        }
        phi.push(p);
    }
    phi
}

/// Rotate `x` by the frequency offset `omega` (rad/sample): sample $n$ by
/// $e^{j\omega (n - n_{c})}$, $n_{c} = N/2$ if `centred` (the phase is zero at the middle
/// of the block), $0$ otherwise.
pub fn frequency_offset(x: &mut [C64], omega: f64, centred: bool) {
    let nc = if centred { x.len() as f64 / 2.0 } else { 0.0 };
    comnum_dsp::signal::rotate(x, omega, nc);
}

/// The Wiener phase noise of an oscillator, drawn independently for every stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseNoise {
    /// Linewidth `Δν` (the sum of the transmitter and receiver linewidths for a link).
    pub linewidth: Hertz,
}

impl Impair for PhaseNoise {
    /// # Errors
    /// The errors of [`increment_sd`] (a negative linewidth, a signal without a positive
    /// sampling rate).
    fn impair_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let sd = increment_sd(self.linewidth.0, x.sample_rate().0)?;
        x.map_streams_in_place(out, |_, y| {
            let phi = wiener_phase(sd, y.len(), PhaseOrigin::Zero, rng);
            for (v, p) in y.iter_mut().zip(phi) {
                *v *= cis(p);
            }
        });
        Ok(())
    }
}

/// A carrier frequency offset, `x[n] e^{j2πΔf (n - n_c)/f_s}` on every stream
/// ([`frequency_offset`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfo {
    /// Offset `Δf`.
    pub offset: Hertz,
    /// Zero phase at the middle of the signal ($n_{c} = N/2$) instead of its first
    /// sample.
    pub centred: bool,
}

impl Impair for Cfo {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let w = 2.0 * PI * self.offset.0 / x.sample_rate().0;
        x.map_streams_in_place(out, |_, y| frequency_offset(y, w, self.centred));
        Ok(())
    }
}

/// The samples $y\[m\] = x(m (1 + \zeta))$ of a receiver whose sampling period is
/// $(1 + \zeta)$ times that of `x` (`zeta` the relative clock offset; see the module): as
/// many samples as `x`, interpolated by the cubic Lagrange kernel, zero beyond the end of
/// `x`.
///
/// # Examples
/// ```
/// use comnum_hardware::oscillator::sampling_offset;
/// use comnum_math::C64;
/// // a slow tone sampled 1 % too slowly: sample m is the tone at 1.01 m
/// let x: Vec<C64> = (0..64).map(|n| C64::from_polar(1.0, 0.05 * f64::from(n))).collect();
/// let y = sampling_offset(&x, 0.01);
/// assert!((y[40] - C64::from_polar(1.0, 0.05 * 40.4)).norm() < 1e-5);
/// ```
pub fn sampling_offset(x: &[C64], zeta: f64) -> Vec<C64> {
    let at = |i: isize| if i >= 0 && (i as usize) < x.len() { x[i as usize] } else { C64::new(0.0, 0.0) };
    (0..x.len())
        .map(|m| {
            let p = m as f64 * (1.0 + zeta);
            if p > (x.len() - 1) as f64 {
                return C64::new(0.0, 0.0);
            }
            let n = p.floor() as isize;
            if p == n as f64 {
                // an instant of the grid: the sample itself, exactly
                return at(n);
            }
            // the nodes n-1 .. n+2 centred on the interval [n, n+1]
            comnum_dsp::interpolation::lagrange_cubic(&[at(n - 1), at(n), at(n + 1), at(n + 2)], p - n as f64 - 1.0)
        })
        .collect()
}

/// A sampling clock offset of `ppm` parts per million on every stream
/// ([`sampling_offset`] with $\zeta = 10^{-6}$ `ppm`; positive for a receiver clock
/// slower than the transmitter's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockOffset {
    /// The relative offset $\zeta$ of the sampling period, in parts per million.
    pub ppm: f64,
}

impl Impair for ClockOffset {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let zeta = self.ppm * 1e-6;
        x.map_streams_in_place(out, |_, y| {
            let r = sampling_offset(y, zeta);
            y.copy_from_slice(&r);
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wiener_increments_have_the_linewidth_variance() {
        let phi = phase_noise(1e6, 100_001, 1e-9, &mut Rng::new(2)).unwrap();
        let inc: Vec<f64> = phi.windows(2).map(|w| w[1] - w[0]).collect();
        let v = inc.iter().map(|d| d * d).sum::<f64>() / inc.len() as f64;
        assert!((v / (2.0 * PI * 1e6 * 1e-9) - 1.0).abs() < 0.02);
    }

    #[test]
    fn the_origins_differ_by_the_first_increment() {
        let zero = wiener_phase(0.1, 50, PhaseOrigin::Zero, &mut Rng::new(3));
        let step = wiener_phase(0.1, 50, PhaseOrigin::Step, &mut Rng::new(3));
        assert_eq!(zero[0], 0.0);
        assert!(step[0] != 0.0);
        let pn = phase_noise(0.01 / (2.0 * PI), 50, 1.0, &mut Rng::new(3)).unwrap();
        assert!(zero.iter().zip(&pn).all(|(a, b)| (a - b).abs() < 1e-12));
        let mut x = vec![C64::new(1.0, 0.0); 8];
        frequency_offset(&mut x, 0.3, true);
        assert!((x[4] - C64::new(1.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn zero_impairments_are_the_identity() {
        let x = Signal::from_streams(
            &[(0..64).map(|n| C64::new(f64::from(n), 1.0)).collect(), vec![C64::new(0.5, -0.5); 64]],
            Hertz(1e9),
        )
        .unwrap();
        let mut rng = Rng::new(0);
        assert_eq!(PhaseNoise { linewidth: Hertz(0.0) }.impair(&x, &mut rng).unwrap(), x);
        assert_eq!(Cfo { offset: Hertz(0.0), centred: false }.impair(&x, &mut rng).unwrap(), x);
        assert_eq!(Cfo { offset: Hertz(0.0), centred: true }.impair(&x, &mut rng).unwrap(), x);
        assert_eq!(ClockOffset { ppm: 0.0 }.impair(&x, &mut rng).unwrap(), x);
        assert!(PhaseNoise { linewidth: Hertz(-1.0) }.impair(&x, &mut rng).is_err());
        assert!(phase_noise(1e3, 4, 0.0, &mut rng).is_err() && increment_sd(f64::NAN, 1.0).is_err());
    }

    /// The blocks rotated a copy of every stream before they rotated the output in place,
    /// and the phase noise took the increments of `phase_noise` (the deviation
    /// $\sqrt{2\pi\Delta\nu T_{s}}$) before those of `increment_sd`
    /// ($\sqrt{2\pi\Delta\nu / f_{s}}$): the same draws, rotations within the rounding of
    /// the deviation.
    #[test]
    fn the_in_place_blocks_are_the_former_ones() {
        let x = Signal::from_streams(
            &[(0..500).map(|n| C64::new(f64::from(n).sin(), 1.0)).collect(), vec![C64::new(0.5, -0.5); 500]],
            Hertz(3e9),
        )
        .unwrap();
        let (lw, ts) = (2e6, x.sample_rate().period().0);
        let mut rng = Rng::new(4);
        let former: Vec<Vec<C64>> = (0..2)
            .map(|s| {
                let phi = wiener_phase((2.0 * PI * lw * ts).sqrt(), 500, PhaseOrigin::Zero, &mut rng);
                x.stream(s).iter().zip(phi).map(|(v, p)| v * cis(p)).collect()
            })
            .collect();
        let y = PhaseNoise { linewidth: Hertz(lw) }.impair(&x, &mut Rng::new(4)).unwrap();
        for (s, f) in former.iter().enumerate() {
            let err = y.stream(s).iter().zip(f).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-13, "{err}");
        }
        let mut cfo = x.stream(1).to_vec();
        frequency_offset(&mut cfo, 2.0 * PI * 1e6 / 3e9, true);
        assert_eq!(Cfo { offset: Hertz(1e6), centred: true }.impair(&x, &mut rng).unwrap().stream(1), &cfo[..]);
    }
}
