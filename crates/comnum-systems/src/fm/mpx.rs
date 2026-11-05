//! The stereophonic multiplex of FM broadcasting, pilot-tone system (ITU-R BS.450-4,
//! Sec. 2.2), with the RDS subcarrier.
//!
//! # Transmitter ([`multiplex`])
//!
//! From the left and right signals $A$, $B$ (pre-emphasised, [`comnum_dsp::fm::Emphasis`]),
//! $M = (A + B)/2$ and $S = (A - B)/2$, the multiplex is
//!
//! $$
//! m(t) = \Delta_{a} \left[M(t) + S(t) \sin 2\theta(t)\right] + \Delta_{p} \sin\theta(t) +
//! \Delta_{r}\\, r(t) \sin 3\theta(t), \qquad \theta(t) = 2\pi\\, 19\\,\mathrm{kHz}\\, t,
//! $$
//!
//! in hertz of deviation of the carrier: the audio $\Delta_{a}$ = 90 % of the 75 kHz
//! (Sec. 2.2.2.4: $M$ and the two sidebands of $S$ at most 90 % each), the pilot
//! $\Delta_{p}$ = 8 to 10 %, the 38 kHz subcarrier suppressed, and the phase of the pilot
//! such that $m$ crosses zero upwards with the pilot when $A > 0$, $B = -A$ (Sec. 2.2.2.5:
//! the subcarrier is $\sin 2\theta$ for a pilot $\sin\theta$). The RDS biphase baseband
//! $r(t)$ ([`super::rds::biphase_baseband`]) modulates $\sin 3\theta$, the third harmonic
//! of the pilot in phase (NRSC-4 Sec. 1.1–1.2), $\Delta_{r}$ = 2 kHz (Sec. 1.3, the
//! recommended deviation).
//!
//! # Receiver ([`stereo_decode`])
//!
//! The pilot is mixed down, $q\[n\] = \overline{m\[n\] e^{-j\theta_{0}\[n\]}}$ (a moving
//! average of 10 ms, [`comnum_dsp::filter::moving_average`]), so that $\sin(\theta_{0} +
//! \varphi)$ gives $q = \frac{\Delta_{p}}{2} e^{j(\varphi - \pi/2)}$: its amplitude tells
//! whether the station is stereo, its phase rebuilds $\sin 2\theta$, and $S =
//! \mathrm{LP}\{2 m \sin 2\theta\}$, $M = \mathrm{LP}\{m\}$ at 15 kHz (Sec. 2.2.2.1), $A =
//! M + S$, $B = M - S$, scaled by $1/\Delta_{a}$, de-emphasised, then resampled to the rate
//! of the audio ([`comnum_dsp::frontend::resample_rational`]).
//!
//! # Examples
//! ```
//! use comnum_systems::fm::mpx::{multiplex, stereo_decode, Levels, MPX_RATE};
//! use comnum_dsp::fm::Emphasis;
//! let n = 45_600;                                              // 0.2 s
//! let left: Vec<f64> = (0..n).map(|i| 0.5 * (2.0 * std::f64::consts::PI * 1e3 * i as f64 / MPX_RATE).sin()).collect();
//! let right = vec![0.0; n];
//! let m = multiplex(&left, &right, None, MPX_RATE, &Levels::default(), Emphasis::EUROPE)?;
//! let audio = stereo_decode(&m, MPX_RATE, Emphasis::EUROPE, 48e3)?;
//! assert!(audio.stereo);
//! assert!((audio.pilot_hz - 6750.0).abs() < 1.0);               // 9 % of 75 kHz
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. Recommendation ITU-R BS.450-4, *Transmission standards for FM sound broadcasting
//!    at VHF*, 2019, Sec. 2.2.
//! 2. NRSC-4, *United States RBDS Standard*, April 1998, Sec. 1.1–1.3.

use super::rds;
use comnum_core::{Result, ensure, ensure_shape};
use comnum_dsp::filter::moving_average;
use comnum_dsp::fm::Emphasis;
use comnum_dsp::frontend::resample_rational;
use comnum_math::C64;
use std::f64::consts::PI;

/// The pilot (Hz).
pub const PILOT_HZ: f64 = 19e3;
/// The maximum deviation of the carrier in Western Europe and the United States (Hz).
pub const MAX_DEVIATION_HZ: f64 = 75e3;
/// A rate for the multiplex: 12 times the pilot, 4 times the RDS subcarrier, 192
/// samples per RDS bit.
pub const MPX_RATE: f64 = 228e3;
/// The bandwidth of the audio (Hz).
pub const AUDIO_BANDWIDTH_HZ: f64 = 15e3;
/// The pilot deviation below which a station counts as mono (Hz), a third of the 8 % of
/// BS.450.
pub const STEREO_THRESHOLD_HZ: f64 = 2e3;

/// The deviations of the components of the multiplex (Hz).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Levels {
    /// The audio for $|A|, |B| \le 1$: 90 % of 75 kHz.
    pub audio_hz: f64,
    /// The pilot: 9 % of 75 kHz.
    pub pilot_hz: f64,
    /// The RDS subcarrier: 2 kHz.
    pub rds_hz: f64,
}

impl Default for Levels {
    /// 67.5, 6.75 and 2 kHz.
    fn default() -> Levels {
        Levels { audio_hz: 0.9 * MAX_DEVIATION_HZ, pilot_hz: 0.09 * MAX_DEVIATION_HZ, rds_hz: 2e3 }
    }
}

/// The multiplex (Hz of deviation, at `fs`) of the audio `left`, `right` (at `fs`,
/// $|A|, |B| \le 1$, pre-emphasised here by `emphasis`) and of the transmitted RDS bits
/// `rds_bits` (differentially encoded, [`rds::differential_encode`]; repeated to the
/// length of the audio), see the module.
///
/// # Errors
/// [`comnum_core::Error::Shape`] for channels of different lengths;
/// [`comnum_core::Error::Config`] for a rate below 120 kHz, or that is not a multiple of
/// 2375 Hz with RDS, or an invalid emphasis.
pub fn multiplex(
    left: &[f64],
    right: &[f64],
    rds_bits: Option<&[u8]>,
    fs: f64,
    levels: &Levels,
    emphasis: Emphasis,
) -> Result<Vec<f64>> {
    ensure_shape(left.len() == right.len(), || format!("{} left and {} right samples", left.len(), right.len()))?;
    ensure(fs >= 120e3, || format!("the multiplex needs 120 kHz at least, got {fs}"))?;
    let a = emphasis.preemphasize(left, fs)?;
    let b = emphasis.preemphasize(right, fs)?;
    let r = match rds_bits {
        Some(bits) if !bits.is_empty() => {
            let per_bit = fs / rds::BIT_RATE;
            let n_bits = (left.len() as f64 / per_bit).ceil() as usize;
            let d: Vec<u8> = (0..n_bits).map(|k| bits[k % bits.len()]).collect();
            rds::biphase_baseband(&d, fs)?
        }
        _ => Vec::new(),
    };
    let w = 2.0 * PI * PILOT_HZ / fs;
    Ok((0..left.len())
        .map(|n| {
            let th = w * n as f64;
            let (m, s) = ((a[n] + b[n]) / 2.0, (a[n] - b[n]) / 2.0);
            let rds = r.get(n).map_or(0.0, |v| levels.rds_hz * v * (3.0 * th).sin());
            levels.audio_hz * (m + s * (2.0 * th).sin()) + levels.pilot_hz * th.sin() + rds
        })
        .collect())
}

/// The audio of a station.
#[derive(Clone, Debug, PartialEq)]
pub struct StereoAudio {
    /// The left channel, $A$ (full scale 1).
    pub left: Vec<f64>,
    /// The right channel, $B$ (the left one for a mono station).
    pub right: Vec<f64>,
    /// The sampling rate of the audio (Hz).
    pub fs: f64,
    /// The deviation of the pilot (Hz), 0 without.
    pub pilot_hz: f64,
    /// Whether the pilot reaches [`STEREO_THRESHOLD_HZ`].
    pub stereo: bool,
}

/// The left and right channels of the multiplex `mpx` (Hz of deviation, at `fs`),
/// de-emphasised by `emphasis` and resampled to `fs_audio` (see the module).
///
/// # Errors
/// [`comnum_core::Error::Config`] for a rate below 120 kHz, an audio rate below 30 kHz,
/// rates whose ratio [`comnum_dsp::frontend::rational_ratio`] refuses, or an invalid
/// emphasis.
pub fn stereo_decode(mpx: &[f64], fs: f64, emphasis: Emphasis, fs_audio: f64) -> Result<StereoAudio> {
    ensure(fs >= 120e3, || format!("the multiplex needs 120 kHz at least, got {fs}"))?;
    ensure(fs_audio >= 2.0 * AUDIO_BANDWIDTH_HZ, || format!("the audio needs 30 kHz at least, got {fs_audio}"))?;
    let w = 2.0 * PI * PILOT_HZ / fs;
    let mixed: Vec<C64> = mpx.iter().enumerate().map(|(n, v)| C64::from_polar(*v, -w * n as f64)).collect();
    let q = moving_average(&mixed, (0.01 * fs) as usize);
    // the median of |q|: the edges of the average (zeros outside the signal) count little
    let mut mags: Vec<f64> = q.iter().map(|v| v.norm()).collect();
    mags.sort_by(f64::total_cmp);
    let pilot_hz = mags.get(mags.len() / 2).map_or(0.0, |v| 2.0 * v);
    let stereo = pilot_hz >= STEREO_THRESHOLD_HZ;
    let levels = Levels::default();
    let (mut m, mut s) = (Vec::with_capacity(mpx.len()), Vec::with_capacity(mpx.len()));
    for (n, v) in mpx.iter().enumerate() {
        m.push(C64::new(v / levels.audio_hz, 0.0));
        // the pilot sin(theta0 + phi) has q = |q| e^{j(phi - pi/2)}: phi = arg q + pi/2
        let sub = if stereo { (2.0 * (w * n as f64 + q[n].arg() + PI / 2.0)).sin() } else { 0.0 };
        s.push(C64::new(2.0 * v * sub / levels.audio_hz, 0.0));
    }
    let lp = |x: &[C64]| resample_rational(x, fs, fs, AUDIO_BANDWIDTH_HZ, 255);
    let (m, s) = (lp(&m)?, lp(&s)?);
    let left: Vec<f64> = m.iter().zip(&s).map(|(a, b)| a.re + b.re).collect();
    let right: Vec<f64> = m.iter().zip(&s).map(|(a, b)| a.re - b.re).collect();
    let out = |x: &[f64]| -> Result<Vec<f64>> {
        let d: Vec<C64> = emphasis.deemphasize(x, fs)?.iter().map(|v| C64::new(*v, 0.0)).collect();
        Ok(resample_rational(&d, fs, fs_audio, AUDIO_BANDWIDTH_HZ, 129)?.iter().map(|v| v.re).collect())
    };
    Ok(StereoAudio { left: out(&left)?, right: out(&right)?, fs: fs_audio, pilot_hz, stereo })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(f: f64, a: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| a * (2.0 * PI * f * i as f64 / MPX_RATE).sin()).collect()
    }

    fn rms(x: &[f64]) -> f64 {
        let m = &x[x.len() / 4..3 * x.len() / 4];
        (m.iter().map(|v| v * v).sum::<f64>() / m.len() as f64).sqrt()
    }

    #[test]
    fn the_channels_come_back_apart_with_their_levels() {
        let n = 68_400; // 0.3 s
        let (l, r) = (tone(1e3, 0.5, n), tone(3e3, 0.2, n));
        let m = multiplex(&l, &r, None, MPX_RATE, &Levels::default(), Emphasis::EUROPE).unwrap();
        // the peak deviation stays below 75 kHz
        assert!(m.iter().all(|v| v.abs() < MAX_DEVIATION_HZ), "{}", m.iter().fold(0.0f64, |a, v| a.max(v.abs())));
        let a = stereo_decode(&m, MPX_RATE, Emphasis::EUROPE, 48e3).unwrap();
        assert!(a.stereo && (a.pilot_hz - 6750.0).abs() < 1.0, "{}", a.pilot_hz);
        let (gl, gr) = (rms(&a.left) / (0.5 / 2f64.sqrt()), rms(&a.right) / (0.2 / 2f64.sqrt()));
        assert!((gl - 1.0).abs() < 0.02 && (gr - 1.0).abs() < 0.02, "{gl} {gr}");
        // the separation: the 1 kHz of the left in the right channel, by projection
        let at = |x: &[f64], f: f64| -> f64 {
            let k = 2.0 * PI * f / 48e3;
            let c: C64 = x[4800..].iter().enumerate().map(|(i, v)| C64::from_polar(*v, -k * (i + 4800) as f64)).sum();
            c.norm() / (x.len() - 4800) as f64
        };
        let separation = 20.0 * (at(&a.left, 1e3) / at(&a.right, 1e3)).log10();
        assert!(separation > 40.0, "{separation} dB");
    }

    #[test]
    fn without_pilot_the_station_is_mono() {
        let n = 45_600;
        let l = tone(1e3, 0.5, n);
        let levels = Levels { pilot_hz: 0.0, ..Levels::default() };
        let m = multiplex(&l, &l, None, MPX_RATE, &levels, Emphasis::EUROPE).unwrap();
        let a = stereo_decode(&m, MPX_RATE, Emphasis::EUROPE, 48e3).unwrap();
        assert!(!a.stereo);
        assert_eq!(a.left, a.right);
        assert!(multiplex(&l, &l[1..], None, MPX_RATE, &levels, Emphasis::EUROPE).is_err());
        assert!(stereo_decode(&m, 100e3, Emphasis::EUROPE, 48e3).is_err());
        assert!(stereo_decode(&m, MPX_RATE, Emphasis::EUROPE, 20e3).is_err());
    }
}
