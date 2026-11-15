//! A sound card as a channel: a complex baseband signal carried on an audio tone, and
//! the WAV container of the samples.
//!
//! # Model
//!
//! A loudspeaker and a microphone carry one real signal, band-limited to half the
//! audio rate $f_{a}$ (48 kHz on most cards). A complex baseband signal $x$ sampled at
//! $f_{s}$ is played on the carrier $f_{c}$ as ([`upconvert`])
//!
//! $$
//! s\[n\] = \operatorname{Re}\\!\left( \tilde x\[n\]\\, e^{j 2\pi f_{c} n / f_{a}} \right),
//! $$
//!
//! with $\tilde x$ the signal resampled from $f_{s}$ to $f_{a}$
//! ([`crate::frontend::resample_rational`]); its band $f_{c} \pm f_{s}/2$ must lie
//! inside $(0, f_{a}/2)$. A recording $r$ comes back to baseband as ([`downconvert`])
//!
//! $$
//! y\[m\] = 2\\, \mathrm{LPF}\\!\left\\{ r\[n\]\\, e^{-j 2\pi f_{c} n / f_{a}} \right\\}
//! \ \text{resampled to } f_{s},
//! $$
//!
//! the channel selection of a software radio on a real capture
//! ([`crate::frontend::select_channel`]): the low-pass filter removes the image at
//! $-2 f_{c}$ and the factor 2 restores the amplitude. What comes back is the signal
//! sent with an unknown delay, the phase of the carrier at that delay, the difference
//! between the clocks of the two cards (tens of parts per million, on the carrier and
//! on the symbol rate alike) and the echoes of the room: the receiver that follows has
//! the synchronisation of any radio link to do.
//!
//! [`wav_encode`] and [`wav_decode`] are the RIFF/WAVE container: mono 16-bit PCM
//! written, PCM of 8 to 32 bits and IEEE floats of 32 and 64 bits read, every channel.
//!
//! # Examples
//! ```
//! use comnum_dsp::audio::{downconvert, upconvert, wav_decode, wav_encode};
//! use comnum_math::C64;
//! // a constant at baseband, sampled at 2.4 kHz: a tone of 6 kHz on the card
//! let x = vec![C64::new(0.5, 0.0); 240];
//! let s = upconvert(&x, 2400.0, 6000.0, 48_000.0)?;
//! assert_eq!(s.len(), 4800);
//! let wav = wav_decode(&wav_encode(&s, 48_000)?)?;
//! assert_eq!((wav.sample_rate, wav.channels.len()), (48_000, 1));
//! let y = downconvert(&wav.channels[0], 48_000.0, 6000.0, 2400.0)?;
//! assert!((y[120] - x[120]).norm() < 1e-3);
//! # Ok::<(), comnum_core::Error>(())
//! ```

use crate::frontend::{resample_rational, select_channel};
use comnum_core::{Error, Result, ensure};
use comnum_math::{C64, cis};
use std::f64::consts::TAU;

/// Taps per input sample of the resampling filters of [`upconvert`] and [`downconvert`].
const TAPS: usize = 65;

/// `Ok` when the band $f_{c} \pm f_{s}/2$ lies inside $(0, f_{a}/2)$.
fn check_band(fs: f64, carrier_hz: f64, audio_rate: f64) -> Result<()> {
    ensure(fs > 0.0 && carrier_hz - fs / 2.0 >= 0.0 && carrier_hz + fs / 2.0 <= audio_rate / 2.0, || {
        format!(
            "the band {carrier_hz} ± {} Hz does not fit in the 0 to {} Hz of a card at {audio_rate} Hz",
            fs / 2.0,
            audio_rate / 2.0
        )
    })
}

/// The real signal that carries the complex baseband `x` (sampled at `fs`) on the
/// carrier `carrier_hz` at the audio rate `audio_rate` (see the module); its amplitude
/// is that of `x`, with the ringing of the resampling filter where `x` starts or stops
/// abruptly (keep it within $\pm 1$ for [`wav_encode`], which clips).
///
/// # Errors
/// [`Error::Config`] for rates that are not positive whole numbers of hertz or a band
/// that does not fit under half the audio rate.
pub fn upconvert(x: &[C64], fs: f64, carrier_hz: f64, audio_rate: f64) -> Result<Vec<f64>> {
    check_band(fs, carrier_hz, audio_rate)?;
    let resampled = resample_rational(x, fs, audio_rate, fs / 2.0, TAPS)?;
    let step = TAU * carrier_hz / audio_rate;
    Ok(resampled.iter().enumerate().map(|(n, v)| (v * cis(step * n as f64)).re).collect())
}

/// The complex baseband at the rate `fs` of the band around `carrier_hz` of the real
/// recording `r` sampled at `audio_rate` (see the module).
///
/// # Errors
/// As [`upconvert`].
pub fn downconvert(r: &[f64], audio_rate: f64, carrier_hz: f64, fs: f64) -> Result<Vec<C64>> {
    check_band(fs, carrier_hz, audio_rate)?;
    let complex: Vec<C64> = r.iter().map(|&v| C64::new(2.0 * v, 0.0)).collect();
    select_channel(&complex, audio_rate, carrier_hz, fs, fs / 2.0, TAPS)
}

/// The samples of a WAV file: its rate and its channels, in $\[-1, 1\]$.
#[derive(Clone, Debug, PartialEq)]
pub struct Wav {
    /// Samples per second.
    pub sample_rate: u32,
    /// The samples of every channel.
    pub channels: Vec<Vec<f64>>,
}

/// The bytes of a mono WAV file of 16-bit PCM at `sample_rate`: the `samples`, clipped
/// to $\[-1, 1\]$, on the scale $\pm 32767$.
///
/// # Errors
/// [`Error::Config`] for a rate of zero or more samples than a WAV file holds (2 GiB).
pub fn wav_encode(samples: &[f64], sample_rate: u32) -> Result<Vec<u8>> {
    ensure(sample_rate > 0, || "the sample rate of a WAV file must be positive".to_string())?;
    let data_len = u32::try_from(samples.len() * 2)
        .ok()
        .filter(|n| *n < u32::MAX - 36)
        .ok_or_else(|| Error::Config(format!("{} samples do not fit in a WAV file", samples.len())))?;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend(b"RIFF");
    out.extend((36 + data_len).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes()); // PCM
    out.extend(1u16.to_le_bytes()); // mono
    out.extend(sample_rate.to_le_bytes());
    out.extend((sample_rate * 2).to_le_bytes()); // bytes per second
    out.extend(2u16.to_le_bytes()); // bytes per frame
    out.extend(16u16.to_le_bytes()); // bits per sample
    out.extend(b"data");
    out.extend(data_len.to_le_bytes());
    for s in samples {
        let v = if s.is_nan() { 0.0 } else { s.clamp(-1.0, 1.0) };
        out.extend(((v * 32767.0).round() as i16).to_le_bytes());
    }
    Ok(out)
}

/// A little-endian integer of `bytes` (1 to 4 of them).
fn le(bytes: &[u8]) -> u32 {
    bytes.iter().rev().fold(0, |acc, b| (acc << 8) | u32::from(*b))
}

/// One sample of `width` bytes in the format `float` or PCM, in $\[-1, 1\]$.
fn sample(bytes: &[u8], float: bool) -> f64 {
    match (float, bytes.len()) {
        (true, 4) => f64::from(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        (true, _) => {
            let mut b = [0u8; 8];
            b.copy_from_slice(bytes);
            f64::from_le_bytes(b)
        }
        // 8-bit PCM is unsigned, the wider ones are signed
        (false, 1) => (f64::from(bytes[0]) - 128.0) / 128.0,
        (false, w) => {
            let shift = 32 - 8 * w as u32;
            f64::from(((le(bytes) << shift) as i32) >> shift) / f64::from(1u32 << (8 * w as u32 - 1))
        }
    }
}

/// The samples of the WAV file `bytes` (see the module for the formats read).
///
/// # Errors
/// [`Error::Config`] for bytes that are not a RIFF/WAVE file, a file without its `fmt`
/// or `data` chunk, or a sample format that is not PCM or IEEE float.
pub fn wav_decode(bytes: &[u8]) -> Result<Wav> {
    let bad = |what: &str| Error::Config(format!("not a WAV file that can be read: {what}"));
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("no RIFF/WAVE header"));
    }
    let mut format = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let size = le(&bytes[pos + 4..pos + 8]) as usize;
        // a data chunk cut short (a recording interrupted) is read up to the end
        let body = &bytes[pos + 8..bytes.len().min(pos + 8 + size)];
        match &bytes[pos..pos + 4] {
            b"fmt " if body.len() >= 16 => {
                let mut tag = le(&body[..2]);
                if tag == 0xFFFE && body.len() >= 26 {
                    tag = le(&body[24..26]); // WAVE_FORMAT_EXTENSIBLE: the sub-format
                }
                format = Some((tag, le(&body[2..4]) as usize, le(&body[4..8]), le(&body[14..16]) as usize));
            }
            b"data" => {
                let (tag, n_channels, sample_rate, bits) =
                    format.ok_or_else(|| bad("the data come before the format"))?;
                let float = match (tag, bits) {
                    (1, 8 | 16 | 24 | 32) => false,
                    (3, 32 | 64) => true,
                    _ => return Err(bad(&format!("format {tag} of {bits} bits"))),
                };
                if n_channels == 0 || sample_rate == 0 {
                    return Err(bad("no channel or a rate of zero"));
                }
                let width = bits / 8;
                let mut channels = vec![Vec::with_capacity(body.len() / (width * n_channels)); n_channels];
                for frame in body.chunks_exact(width * n_channels) {
                    for (c, s) in frame.chunks_exact(width).enumerate() {
                        channels[c].push(sample(s, float));
                    }
                }
                return Ok(Wav { sample_rate, channels });
            }
            _ => {}
        }
        pos += 8 + size + (size & 1); // chunks are padded to an even size
    }
    Err(bad("no data chunk"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_baseband_signal_survives_the_carrier_and_the_file() {
        // a slow complex exponential at 4.8 kHz, on a carrier of 9 kHz at 48 kHz
        let (fs, fc, fa) = (4800.0, 9000.0, 48_000.0);
        let x: Vec<C64> = (0..960).map(|n| cis(TAU * 300.0 * n as f64 / fs) * 0.6).collect();
        let s = upconvert(&x, fs, fc, fa).unwrap();
        assert_eq!(s.len(), 9600);
        // the amplitude of the signal, but for the ringing of the filter at its two ends
        assert!(s[200..9400].iter().all(|v| v.abs() <= 0.601) && s.iter().all(|v| v.abs() < 0.75));
        // the tone of the card is at 9.3 kHz: its power in that bin of a one-second DFT grid
        let tone: C64 = s[2400..7200].iter().enumerate().map(|(n, v)| cis(-TAU * 9300.0 * n as f64 / fa) * *v).sum();
        assert!((tone.norm() / 4800.0 - 0.3).abs() < 1e-3);
        let wav = wav_decode(&wav_encode(&s, 48_000).unwrap()).unwrap();
        let y = downconvert(&wav.channels[0], fa, fc, fs).unwrap();
        assert_eq!(y.len(), 960);
        // away from the ends, the signal sent to the 16 bits of the file
        let worst = (100..860).map(|n| (y[n] - x[n]).norm()).fold(0.0, f64::max);
        assert!(worst < 2e-3, "{worst}");
    }

    #[test]
    fn the_band_must_fit_under_half_the_audio_rate() {
        let x = [C64::new(1.0, 0.0); 8];
        assert!(upconvert(&x, 8000.0, 3000.0, 48_000.0).is_err()); // below 0 Hz
        assert!(upconvert(&x, 8000.0, 21_000.0, 48_000.0).is_err()); // above 24 kHz
        assert!(upconvert(&x, 8000.5, 6000.0, 48_000.0).is_err()); // not whole hertz
        assert!(downconvert(&[0.0; 8], 48_000.0, 30_000.0, 8000.0).is_err());
        assert!(upconvert(&x, 8000.0, 4000.0, 48_000.0).is_ok());
    }

    #[test]
    fn the_wav_container_reads_what_it_writes_and_the_usual_formats() {
        let samples = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, f64::NAN];
        let bytes = wav_encode(&samples, 44_100).unwrap();
        assert_eq!(bytes.len(), 44 + 14);
        let wav = wav_decode(&bytes).unwrap();
        assert_eq!(wav.sample_rate, 44_100);
        let expected = [0.0, 0.5, -0.5, 1.0, -1.0, 1.0, 0.0];
        assert!(wav.channels[0].iter().zip(expected).all(|(a, b)| (a - b * 32767.0 / 32768.0).abs() < 2e-5));
        // stereo 32-bit float behind an unknown chunk of odd size, then stereo 24-bit PCM
        let header = |tag: u16, bits: u16, data: &[u8]| {
            let mut b = b"RIFF\0\0\0\0WAVELIST\x03\0\0\0abc\0fmt \x10\0\0\0".to_vec();
            b.extend(tag.to_le_bytes());
            b.extend(2u16.to_le_bytes());
            b.extend(8000u32.to_le_bytes());
            b.extend([0; 6]);
            b.extend(bits.to_le_bytes());
            b.extend(b"data");
            b.extend((data.len() as u32).to_le_bytes());
            b.extend(data);
            b
        };
        let floats: Vec<u8> = [0.25f32, -0.75, 0.5, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let wav = wav_decode(&header(3, 32, &floats)).unwrap();
        assert_eq!(wav.channels, vec![vec![0.25, 0.5], vec![-0.75, 1.0]]);
        let pcm24 = [0x00, 0x00, 0x40, 0x00, 0x00, 0xC0];
        let wav = wav_decode(&header(1, 24, &pcm24)).unwrap();
        assert_eq!(wav.channels, vec![vec![0.5], vec![-0.5]]);
        assert!(wav_decode(b"not a wav file").is_err());
        assert!(wav_decode(&header(2, 16, &[0; 4])).is_err());
        assert!(wav_decode(&bytes[..40]).is_err());
        assert!(wav_encode(&[0.0], 0).is_err());
    }
}
