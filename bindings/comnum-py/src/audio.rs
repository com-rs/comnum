//! A sound card as a channel: text as bits, HDLC frames, a baseband signal on an
//! audio carrier and the WAV files.

use crate::{BArr, CArr, FArr, IntoPy, bvec, carr, cvec, farr, fvec};
use comnum::dsp::audio;
use comnum::systems::hdlc;
use numpy::IntoPyArray;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;

/// The bits of the UTF-8 bytes of a text, every byte the most significant bit first.
///
/// Args:
///     text (str): the text.
///
/// Returns:
///     numpy.ndarray: eight bits per byte (uint8).
///
/// Examples:
///     >>> import comnum
///     >>> comnum.text_to_bits("A").tolist()
///     [0, 1, 0, 0, 0, 0, 0, 1]
///     >>> comnum.bits_to_text(comnum.text_to_bits("été 2026"))
///     'été 2026'
#[pyfunction]
fn text_to_bits<'py>(py: Python<'py>, text: &str) -> BArr<'py> {
    comnum::math::bits::text_to_bits(text).into_pyarray(py)
}

/// The text of bits read eight at a time as UTF-8 bytes, the inverse of
/// `text_to_bits`: the bits after the last whole byte are dropped, and bytes that are
/// not UTF-8 (a bit received in error) become the replacement character `�`.
///
/// Args:
///     bits (array_like): the bits.
///
/// Returns:
///     str: the text.
///
/// Raises:
///     ValueError: for values other than 0 and 1.
#[pyfunction]
fn bits_to_text(bits: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(comnum::math::bits::bits_to_text(&bvec(bits)?))
}

/// The bits of one HDLC frame (ISO/IEC 13239) of a payload: the flag `01111110`, the
/// payload and its frame check sequence (CRC-16/X-25), least significant bit first and
/// bit-stuffed, and the closing flag. A receiver finds the frame in a stream of bits
/// by its flags and knows by the check sequence whether it is intact.
///
/// Args:
///     payload (bytes | str): the bytes of the frame (a text is sent as UTF-8).
///
/// Returns:
///     numpy.ndarray: the bits of the frame (uint8).
///
/// Raises:
///     TypeError: for a payload that is neither bytes nor a text.
///
/// Examples:
///     >>> import comnum
///     >>> bits = comnum.hdlc_frame("hello")
///     >>> comnum.hdlc_frames(bits)
///     [(b'hello', True)]
#[pyfunction]
fn hdlc_frame<'py>(py: Python<'py>, payload: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
    let bytes: Vec<u8> = match payload.extract::<String>() {
        Ok(text) => text.into_bytes(),
        Err(_) => payload.extract()?,
    };
    Ok(hdlc::frame(&bytes).into_pyarray(py))
}

/// The HDLC frames of a stream of bits: what lies between two flags, unstuffed, with
/// the verdict of its frame check sequence.
///
/// Args:
///     bits (array_like): the received bits.
///     min_bytes (int): the smallest payload kept.
///
/// Returns:
///     list[tuple[bytes, bool]]: the payload of every frame and whether its check
///     sequence is correct.
///
/// Raises:
///     ValueError: for values other than 0 and 1.
#[pyfunction]
#[pyo3(signature = (bits, *, min_bytes=1))]
fn hdlc_frames<'py>(
    py: Python<'py>,
    bits: &Bound<'py, PyAny>,
    min_bytes: usize,
) -> PyResult<Vec<(Bound<'py, PyBytes>, bool)>> {
    let frames = hdlc::frames_of_at_least(&bvec(bits)?, min_bytes);
    Ok(frames.iter().map(|f| (PyBytes::new(py, &f.payload), f.fcs_ok)).collect())
}

/// The real signal of a sound card that carries a complex baseband signal on an audio
/// carrier: :math:`\operatorname{Re}(\tilde{x}[n] \exp(j 2\pi f_c n / f_a))`, :math:`\tilde{x}`
/// the signal resampled to the audio rate. The band :math:`f_c \pm f_s/2` must lie between
/// 0 and half the audio rate.
///
/// Args:
///     x (array_like): the complex baseband samples.
///     fs (float): their sample rate (Hz, a whole number).
///     carrier_hz (float): the carrier.
///     audio_rate (float): the rate of the card (Hz).
///
/// Returns:
///     numpy.ndarray: the real samples at `audio_rate`, of the amplitude of `x`.
///
/// Raises:
///     ValueError: for a band that does not fit or rates that are not whole hertz.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> s = comnum.audio_upconvert(np.full(240, 0.5 + 0j), 2400, 6000)
///     >>> s.size, bool(np.allclose(comnum.audio_downconvert(s, 6000, 2400)[60:180], 0.5, atol=1e-3))
///     (4800, True)
#[pyfunction]
#[pyo3(signature = (x, fs, carrier_hz, audio_rate=48_000.0))]
fn audio_upconvert<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    fs: f64,
    carrier_hz: f64,
    audio_rate: f64,
) -> PyResult<FArr<'py>> {
    Ok(farr(py, audio::upconvert(&cvec(x)?, fs, carrier_hz, audio_rate).py()?))
}

/// The complex baseband of the band around a carrier of a real recording: the mixing
/// to 0 Hz, the low-pass filter and the resampling to `fs`. What comes back has an
/// unknown delay and phase, and the clock difference of the two cards.
///
/// Args:
///     r (array_like): the real samples of the recording.
///     carrier_hz (float): the carrier.
///     fs (float): the sample rate of the baseband signal wanted (Hz, a whole number).
///     audio_rate (float): the rate of the recording (Hz).
///
/// Returns:
///     numpy.ndarray: the complex baseband samples at `fs`.
///
/// Raises:
///     ValueError: for a band that does not fit or rates that are not whole hertz.
#[pyfunction]
#[pyo3(signature = (r, carrier_hz, fs, audio_rate=48_000.0))]
fn audio_downconvert<'py>(
    py: Python<'py>,
    r: &Bound<'py, PyAny>,
    carrier_hz: f64,
    fs: f64,
    audio_rate: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, audio::downconvert(&fvec(r)?, audio_rate, carrier_hz, fs).py()?))
}

/// Write real samples to a mono WAV file of 16-bit PCM. The samples are clipped to
/// :math:`[-1, 1]`: scale the signal first (a peak of 0.5 leaves headroom).
///
/// Args:
///     path (str | os.PathLike): the file.
///     samples (array_like): the real samples.
///     sample_rate (int): samples per second.
///
/// Raises:
///     ValueError: for a rate of zero.
///     OSError: if the file cannot be written.
#[pyfunction]
#[pyo3(signature = (path, samples, sample_rate=48_000))]
fn write_wav(path: PathBuf, samples: &Bound<'_, PyAny>, sample_rate: u32) -> PyResult<()> {
    let bytes = audio::wav_encode(&fvec(samples)?, sample_rate).py()?;
    std::fs::write(&path, bytes).map_err(|e| PyOSError::new_err(format!("{}: {e}", path.display())))
}

/// Read a WAV file: PCM of 8 to 32 bits or IEEE floats, every channel.
///
/// Args:
///     path (str | os.PathLike): the file.
///
/// Returns:
///     tuple[numpy.ndarray, int]: the samples in :math:`[-1, 1]`, `(n,)` for a mono file and
///     `(channels, n)` otherwise, and the sample rate.
///
/// Raises:
///     ValueError: for a file that is not a WAV file of these formats.
///     OSError: if the file cannot be read.
#[pyfunction]
fn read_wav(py: Python<'_>, path: PathBuf) -> PyResult<(Py<PyAny>, u32)> {
    let bytes = std::fs::read(&path).map_err(|e| PyOSError::new_err(format!("{}: {e}", path.display())))?;
    let mut wav = audio::wav_decode(&bytes).py()?;
    let samples = if wav.channels.len() == 1 {
        wav.channels.remove(0).into_pyarray(py).into_any().unbind()
    } else {
        let n = wav.channels[0].len();
        ndarray::Array2::from_shape_vec((wav.channels.len(), n), wav.channels.concat())
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    Ok((samples, wav.sample_rate))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(text_to_bits, m)?)?;
    m.add_function(wrap_pyfunction!(bits_to_text, m)?)?;
    m.add_function(wrap_pyfunction!(hdlc_frame, m)?)?;
    m.add_function(wrap_pyfunction!(hdlc_frames, m)?)?;
    m.add_function(wrap_pyfunction!(audio_upconvert, m)?)?;
    m.add_function(wrap_pyfunction!(audio_downconvert, m)?)?;
    m.add_function(wrap_pyfunction!(write_wav, m)?)?;
    m.add_function(wrap_pyfunction!(read_wav, m)?)?;
    Ok(())
}
