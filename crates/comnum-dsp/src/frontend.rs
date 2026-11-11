//! The front end of a receiver of measured signals: one channel of a capture brought
//! to baseband at the sampling rate of its receiver.
//!
//! A software radio records a band wider than the channel of interest, centred on its
//! tuning frequency, with the offset of its converter at 0 Hz. [`select_channel`]
//!
//! 1. removes the mean of the capture (the DC offset of the converter, a spike at
//!    0 Hz that the shift below would carry into the band);
//! 2. shifts the channel, `offset_hz` above the tuning frequency, to 0 Hz:
//!    $x\[n\] e^{-j 2\pi f_{\mathrm{off}} n / f_{s}}$ ([`crate::signal::freq_shift`]);
//! 3. resamples by the exact ratio $U/D = f_{\mathrm{out}}/f_{\mathrm{in}}$ of the two rates
//!    (integers in hertz, reduced by their greatest common divisor) with the low-pass
//!    filter of cut-off `cutoff_hz` designed at $U f_{\mathrm{in}}$: a sinc of
//!    $U N_{\mathrm{taps}}$ taps (made odd) under a Dolph–Chebyshev window of 80 dB
//!    ([`crate::filter::low_pass_fir`], [`crate::window::chebyshev`]), its gain $U$ at
//!    DC, applied by [`crate::filter::upfirdn`] (the `same` alignment: no delay).
//!
//! The cut-off is the frequency of half the amplitude, the middle of a transition band
//! about $4 f_{\mathrm{in}}/N_{\mathrm{taps}}$ wide (the main lobe of the window): the
//! passband ends about $2 f_{\mathrm{in}}/N_{\mathrm{taps}}$ below the cut-off, and the channels next to the selected one are rejected
//! by the 80 dB of the window once past it.
//!
//! # Examples
//! ```
//! use comnum_dsp::frontend::select_channel;
//! use comnum_math::C64;
//! // a tone 2 MHz above the tuning frequency, captured at 15.36 MS/s
//! let fs = 15.36e6;
//! let x: Vec<C64> = (0..15_360)
//!     .map(|n| C64::from_polar(1.0, 2.0 * std::f64::consts::PI * 2.0e6 * n as f64 / fs))
//!     .collect();
//! let y = select_channel(&x, fs, 2.0e6, 1.92e6, 0.8e6, 129)?;
//! assert_eq!(y.len(), 1920);                          // 1 ms at 1.92 MS/s
//! assert!((y[960].norm() - 1.0).abs() < 1e-3);        // the tone, at 0 Hz
//! # Ok::<(), comnum_core::Error>(())
//! ```

use crate::filter::{LowPass, low_pass_fir, upfirdn};
use crate::signal::freq_shift;
use crate::window::chebyshev;
use comnum_core::{Result, ensure};
use comnum_math::C64;

/// The largest interpolation factor $U$ of [`select_channel`].
pub const MAX_UP: u64 = 1024;

/// The attenuation of the sidelobes of the window of the filter (dB).
const WINDOW_DB: f64 = 80.0;

/// The integer of a rate in hertz, if it is one.
fn hertz(fs: f64, name: &str) -> Result<u64> {
    ensure(fs.is_finite() && fs.round() >= 1.0 && fs < 1e12 && (fs - fs.round()).abs() < 1e-6, || {
        format!("{name} must be a positive whole number of hertz, got {fs}")
    })?;
    Ok(fs.round() as u64)
}

/// The resampling ratio $U/D = f_{\mathrm{out}}/f_{\mathrm{in}}$, reduced.
///
/// # Errors
/// [`comnum_core::Error::Config`] for rates that are not positive whole numbers of hertz,
/// or a reduced $U$ above [`MAX_UP`].
pub fn rational_ratio(fs_in: f64, fs_out: f64) -> Result<(usize, usize)> {
    let (a, b) = (hertz(fs_in, "the input rate")?, hertz(fs_out, "the output rate")?);
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    let (up, down) = (b / x, a / x);
    ensure(up <= MAX_UP, || {
        format!("the ratio {fs_out}/{fs_in} reduces to {up}/{down}: an interpolation by more than {MAX_UP}")
    })?;
    Ok((up as usize, down as usize))
}

/// The channel `offset_hz` above the tuning frequency of the capture `x` (sampled at
/// `fs_in`), at baseband and sampled at `fs_out`, filtered to `cutoff_hz` by a filter of
/// about `n_taps` taps at the input rate (see the module). The output has
/// $\lceil U N / D \rceil$ samples for the $N$ of `x`.
///
/// # Errors
/// [`comnum_core::Error::Config`] for rates that are not positive whole numbers of hertz,
/// a ratio that [`rational_ratio`] refuses, an offset outside the captured band, a
/// cut-off that is not in $(0, f_{\mathrm{out}}/2\]$, or no tap.
pub fn select_channel(
    x: &[C64],
    fs_in: f64,
    offset_hz: f64,
    fs_out: f64,
    cutoff_hz: f64,
    n_taps: usize,
) -> Result<Vec<C64>> {
    let (up, down) = rational_ratio(fs_in, fs_out)?;
    ensure(offset_hz.is_finite() && offset_hz.abs() < fs_in / 2.0, || {
        format!("the offset {offset_hz} Hz is outside the captured band of ±{} Hz", fs_in / 2.0)
    })?;
    ensure(cutoff_hz > 0.0 && cutoff_hz <= fs_out / 2.0, || {
        format!("the cut-off must be in (0, {}] Hz, got {cutoff_hz}", fs_out / 2.0)
    })?;
    ensure(n_taps > 0, || "the filter needs at least one tap".into())?;
    let mean = if x.is_empty() { C64::new(0.0, 0.0) } else { x.iter().sum::<C64>() / x.len() as f64 };
    let centred: Vec<C64> = x.iter().map(|v| v - mean).collect();
    let shifted = freq_shift(&centred, -offset_hz, fs_in);
    Ok(upfirdn(&resampling_filter(cutoff_hz, fs_in, up, n_taps), &shifted, up, down))
}

/// The samples `x` (at `fs_in`) resampled to `fs_out` by the exact ratio of
/// [`rational_ratio`] under the filter of [`resampling_filter`]: [`select_channel`]
/// without the removal of the mean and the shift (for a signal already at baseband, or
/// a real signal such as an audio channel). The output has $\lceil U N / D \rceil$
/// samples.
///
/// # Errors
/// As [`select_channel`], but for the offset.
pub fn resample_rational(x: &[C64], fs_in: f64, fs_out: f64, cutoff_hz: f64, n_taps: usize) -> Result<Vec<C64>> {
    let (up, down) = rational_ratio(fs_in, fs_out)?;
    ensure(cutoff_hz > 0.0 && cutoff_hz <= fs_in.min(fs_out) / 2.0, || {
        format!("the cut-off must be in (0, {}] Hz, got {cutoff_hz}", fs_in.min(fs_out) / 2.0)
    })?;
    ensure(n_taps > 0, || "the filter needs at least one tap".into())?;
    Ok(upfirdn(&resampling_filter(cutoff_hz, fs_in, up, n_taps), x, up, down))
}

/// The low-pass filter of a resampling by $U/D$ (see the module): a sinc of cut-off
/// `cutoff_hz` at $U f_{\mathrm{in}}$, $U N_{\mathrm{taps}}$ taps made odd, under a
/// Dolph–Chebyshev window of 80 dB, its gain $U$ at DC.
pub fn resampling_filter(cutoff_hz: f64, fs_in: f64, up: usize, n_taps: usize) -> Vec<f64> {
    let n = (n_taps * up.max(1)) | 1;
    let w = chebyshev(n, WINDOW_DB);
    let mut h: Vec<f64> =
        low_pass_fir(cutoff_hz, fs_in * up as f64, n, LowPass::Rect).iter().zip(&w).map(|(a, b)| a * b).collect();
    let dc: f64 = h.iter().sum();
    for v in &mut h {
        *v *= up as f64 / dc;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn tone(f: f64, fs: f64, n: usize) -> Vec<C64> {
        (0..n).map(|i| C64::from_polar(1.0, 2.0 * PI * f * i as f64 / fs)).collect()
    }

    fn power(x: &[C64]) -> f64 {
        // the middle half, away from the edges of the filter
        let m = &x[x.len() / 4..3 * x.len() / 4];
        m.iter().map(C64::norm_sqr).sum::<f64>() / m.len() as f64
    }

    #[test]
    fn the_selected_channel_passes_and_the_others_are_rejected() {
        let fs = 15.36e6;
        // in the channel at 2 MHz: 2.3 MHz is 300 kHz off its centre
        let inside = select_channel(&tone(2.3e6, fs, 30_720), fs, 2.0e6, 1.92e6, 0.6e6, 129).unwrap();
        assert!((power(&inside) - 1.0).abs() < 1e-2, "{}", power(&inside)); // 0.04 dB
        // a channel 3 MHz away, folded onto the output band by the decimation
        let outside = select_channel(&tone(5.0e6, fs, 30_720), fs, 2.0e6, 1.92e6, 0.6e6, 129).unwrap();
        assert!(10.0 * power(&outside).log10() < -70.0, "{}", 10.0 * power(&outside).log10());
        // the DC offset of the converter is removed
        let dc: Vec<C64> = tone(2.0e6, fs, 30_720).iter().map(|v| v + C64::new(3.0, -1.0)).collect();
        let y = select_channel(&dc, fs, 2.0e6, 1.92e6, 0.6e6, 129).unwrap();
        assert!((power(&y) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn the_ratio_is_exact_and_interpolation_works() {
        assert_eq!(rational_ratio(15.36e6, 1.92e6).unwrap(), (1, 8));
        assert_eq!(rational_ratio(20e6, 1.92e6).unwrap(), (12, 125));
        let y = select_channel(&tone(1.0e5, 20e6, 20_000), 20e6, 0.0, 1.92e6, 0.5e6, 256).unwrap();
        assert_eq!(y.len(), 1920);
        assert!((power(&y) - 1.0).abs() < 1e-3, "{}", power(&y));
        assert!(rational_ratio(1e6, 1e6 + 1.0).is_err()); // 1000001/1000000
        // the same filter without the shift, on a signal at baseband
        let r = resample_rational(&tone(1.0e5, 20e6, 20_000), 20e6, 1.92e6, 0.5e6, 256).unwrap();
        assert!((power(&r) - 1.0).abs() < 1e-3);
        assert!(resample_rational(&r, 1.92e6, 1e6, 0.6e6, 9).is_err());
        assert!(rational_ratio(1.5, 1.0).is_err() && rational_ratio(-1.0, 1.0).is_err());
        // a rate that rounds to zero hertz (it used to divide by zero)
        assert!(rational_ratio(1e-9, 1e-9).is_err() && rational_ratio(1e6, 1e-9).is_err());
        let x = tone(0.0, 1e6, 10);
        assert!(select_channel(&x, 1e6, 6e5, 1e6, 1e5, 9).is_err());
        assert!(select_channel(&x, 1e6, 0.0, 1e6, 6e5, 9).is_err());
        assert!(select_channel(&x, 1e6, 0.0, 1e6, 1e5, 0).is_err());
        assert_eq!(select_channel(&[], 1e6, 0.0, 1e6, 1e5, 9).unwrap(), Vec::<C64>::new());
    }
}
