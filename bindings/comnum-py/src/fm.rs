//! FM broadcasting: the modulator, the discriminator, the emphasis, the stereophonic
//! multiplex and the radio data system (RDS).

use crate::{BArr, CArr, FArr, IntoPy, bvec, carr, cvec, farr, fvec};
use comnum::dsp::fm::{self, Emphasis};
use comnum::systems::fm::{mpx, rds};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// The complex envelope of the frequency modulation of the message `m` (sampled at
/// `fs`) with the peak deviation `deviation_hz` for `|m| = 1`: the phase sums
/// `2π Δf m[n] / fs`.
///
/// Args:
///     m (array_like): the real message.
///     fs (float): the sampling rate (Hz).
///     deviation_hz (float): the deviation of `m = 1` (Hz); 1 for a message already in
///         hertz of deviation (the multiplex of :func:`fm_multiplex`).
///
/// Returns:
///     numpy.ndarray: unit-amplitude samples (complex128).
///
/// Raises:
///     ValueError: for an input that is not a 1-D array of real numbers.
#[pyfunction]
fn fm_modulate<'py>(py: Python<'py>, m: &Bound<'py, PyAny>, fs: f64, deviation_hz: f64) -> PyResult<CArr<'py>> {
    Ok(carr(py, fm::fm_modulate(&fvec(m)?, fs, deviation_hz)))
}

/// The instantaneous frequency (Hz) of `x` by the polar discriminator,
/// `fs/(2π) arg(x[n] x*[n-1])` (0 for the first sample): exact within `±fs/2`,
/// insensitive to the amplitude.
///
/// Args:
///     x (array_like): complex samples at baseband.
///     fs (float): the sampling rate (Hz).
///
/// Returns:
///     numpy.ndarray: the frequency of every sample (float64, Hz).
///
/// Raises:
///     ValueError: for an input that is not a 1-D array of numbers.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> x = comnum.fm_modulate(np.full(100, 0.5), 228e3, 75e3)
///     >>> round(float(comnum.fm_discriminator(x, 228e3)[50]), 6)
///     37500.0
#[pyfunction]
fn fm_discriminator<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64) -> PyResult<FArr<'py>> {
    Ok(farr(py, fm::fm_discriminator(&cvec(x)?, fs)))
}

/// The de-emphasis `1/(1 + j2πfτ)` of the message `x` by the matched pole
/// `y[n] = a y[n-1] + (1 - a) x[n]`, `a = exp(-1/(τ fs))` (ITU-R BS.450-4 Sec. 1.2:
/// 50 µs in Europe, 75 µs in the United States).
///
/// Args:
///     x (array_like): the real message.
///     fs (float): the sampling rate (Hz).
///     tau_s (float): the time constant (s).
///
/// Returns:
///     numpy.ndarray: the de-emphasised message (float64).
///
/// Raises:
///     ValueError: for a time constant or a rate that is not positive.
#[pyfunction]
#[pyo3(signature = (x, fs, *, tau_s = 50e-6))]
fn fm_deemphasis<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64, tau_s: f64) -> PyResult<FArr<'py>> {
    Ok(farr(py, Emphasis { tau_s }.deemphasize(&fvec(x)?, fs).py()?))
}

/// The pre-emphasis of the message `x`, the exact inverse of :func:`fm_deemphasis`
/// (the FIR `(x[n] - a x[n-1]) / (1 - a)`).
///
/// Args:
///     x (array_like): the real message.
///     fs (float): the sampling rate (Hz).
///     tau_s (float): the time constant (s).
///
/// Returns:
///     numpy.ndarray: the pre-emphasised message (float64).
///
/// Raises:
///     ValueError: for a time constant or a rate that is not positive.
#[pyfunction]
#[pyo3(signature = (x, fs, *, tau_s = 50e-6))]
fn fm_preemphasis<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64, tau_s: f64) -> PyResult<FArr<'py>> {
    Ok(farr(py, Emphasis { tau_s }.preemphasize(&fvec(x)?, fs).py()?))
}

/// The stereophonic multiplex of FM broadcasting (ITU-R BS.450-4 Sec. 2.2, pilot-tone
/// system), in hertz of deviation: `Δa [M + S sin 2θ] + Δp sin θ + Δr r(t) sin 3θ`,
/// `θ = 2π 19 kHz t`, `M = (A + B)/2`, `S = (A - B)/2` after pre-emphasis, `r` the
/// biphase RDS baseband of the transmitted bits (NRSC-4 Sec. 1).
///
/// Args:
///     left (array_like): the left audio `A` at `fs`, at most 1 in magnitude.
///     right (array_like): the right audio `B`, as many samples.
///     fs (float): the sampling rate (Hz), at least 120 kHz (a multiple of 2375 Hz with
///         RDS; 228 kHz is 12 times the pilot).
///     rds_bits (array_like | None): the transmitted RDS bits (differentially encoded:
///         :func:`rds_differential_encode`), repeated to the length of the audio.
///     audio_hz (float): the deviation of the audio (90 % of 75 kHz).
///     pilot_hz (float): the deviation of the pilot (9 %).
///     rds_hz (float): the deviation of the RDS subcarrier (2 kHz).
///     tau_s (float): the time constant of the pre-emphasis (s).
///
/// Returns:
///     numpy.ndarray: the multiplex (float64, Hz of deviation).
///
/// Raises:
///     ValueError: for channels of different lengths or an invalid rate.
#[pyfunction]
#[pyo3(signature = (left, right, fs, *, rds_bits = None, audio_hz = 67.5e3, pilot_hz = 6.75e3, rds_hz = 2e3, tau_s = 50e-6))]
#[allow(clippy::too_many_arguments)]
fn fm_multiplex<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
    fs: f64,
    rds_bits: Option<&Bound<'py, PyAny>>,
    audio_hz: f64,
    pilot_hz: f64,
    rds_hz: f64,
    tau_s: f64,
) -> PyResult<FArr<'py>> {
    let bits = rds_bits.map(bvec).transpose()?;
    let levels = mpx::Levels { audio_hz, pilot_hz, rds_hz };
    let m = mpx::multiplex(&fvec(left)?, &fvec(right)?, bits.as_deref(), fs, &levels, Emphasis { tau_s }).py()?;
    Ok(farr(py, m))
}

/// The left and right channels of a multiplex (ITU-R BS.450-4 Sec. 2.2): the pilot
/// found by mixing it down (10 ms average), `S = LP{2 m sin 2θ}` and `M = LP{m}` at
/// 15 kHz, `A = M + S`, `B = M - S`, de-emphasised and resampled to `fs_audio`; mono
/// (`A = B = M`) when the pilot deviates by less than 2 kHz.
///
/// Args:
///     mpx (array_like): the multiplex in hertz of deviation (:func:`fm_discriminator`).
///     fs (float): its sampling rate (Hz), at least 120 kHz.
///     tau_s (float): the time constant of the de-emphasis (s).
///     fs_audio (float): the rate of the audio (Hz), at least 30 kHz.
///
/// Returns:
///     dict: ``left``, ``right`` (float64, full scale 1), ``fs`` (Hz), ``pilot_hz`` (the
///     deviation of the pilot), ``stereo`` (bool).
///
/// Raises:
///     ValueError: for invalid rates.
#[pyfunction]
#[pyo3(signature = (mpx, fs, *, tau_s = 50e-6, fs_audio = 48e3))]
fn fm_stereo_decode<'py>(
    py: Python<'py>,
    mpx: &Bound<'py, PyAny>,
    fs: f64,
    tau_s: f64,
    fs_audio: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let m = fvec(mpx)?;
    let a = py.detach(|| mpx::stereo_decode(&m, fs, Emphasis { tau_s }, fs_audio)).py()?;
    let d = PyDict::new(py);
    d.set_item("left", a.left.into_pyarray(py))?;
    d.set_item("right", a.right.into_pyarray(py))?;
    d.set_item("fs", a.fs)?;
    d.set_item("pilot_hz", a.pilot_hz)?;
    d.set_item("stereo", a.stereo)?;
    Ok(d)
}

/// The 104 bits of an RDS group of the four information words `blocks` (each with its
/// checkword and its offset word A, B, C or C', D; NRSC-4 Sec. 2.3, Table A.1).
///
/// Args:
///     blocks (list[int]): the four 16-bit information words.
///     version_b (bool): version B (the offset C' in block 3).
///
/// Returns:
///     numpy.ndarray: 104 bits (uint8), the first one sent first.
///
/// Raises:
///     ValueError: for another number of blocks.
#[pyfunction]
#[pyo3(signature = (blocks, *, version_b = false))]
fn rds_group_bits<'py>(py: Python<'py>, blocks: [u16; 4], version_b: bool) -> BArr<'py> {
    rds::group_bits(blocks, version_b).into_pyarray(py)
}

/// The differential encoding `d[k] = d[k-1] xor b[k]` of RDS (NRSC-4 Table 1).
///
/// Args:
///     bits (array_like): the source bits.
///
/// Returns:
///     numpy.ndarray: the transmitted bits (uint8).
///
/// Raises:
///     ValueError: for values other than 0 and 1.
#[pyfunction]
fn rds_differential_encode<'py>(py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
    Ok(rds::differential_encode(&bvec(bits)?).into_pyarray(py))
}

/// The RDS of a multiplex: the biphase symbols demodulated from 57 kHz (Gardner clock
/// recovery, Viterbi-Viterbi phase of the BPSK), differentially decoded, synchronised on
/// the offset words, and the groups assembled into the station's name, radiotext and
/// clock (NRSC-4 / EN 50067 Sec. 1-3, the characters of Annex E, the date of Annex G).
///
/// Args:
///     mpx (array_like): the multiplex in hertz of deviation (:func:`fm_discriminator`),
///         at least 0.1 s.
///     fs (float): its sampling rate (Hz), a multiple of 2375 Hz of 120 kHz at least
///         (228 kHz).
///
/// Returns:
///     dict: ``pi`` (int | None), ``pty`` (int | None), ``tp`` (bool | None), ``ps``
///     (str, 8 characters), ``radiotext`` (str), ``clock`` (dict with ``year``,
///     ``month``, ``day``, ``hour``, ``minute`` in UTC and ``offset_half_hours``, or
///     None), ``group_types`` (dict ``"0A"`` → count), ``complete_groups``,
///     ``valid_blocks``, ``blocks``, and ``groups`` (a list of four information words
///     each, None for a block whose checkword failed).
///
/// Raises:
///     ValueError: for an invalid rate or less than 0.1 s of multiplex.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> ps = b"COMNUM  "
///     >>> groups = [comnum.rds_group_bits([0xF201, s, 0, ps[2 * s] << 8 | ps[2 * s + 1]]) for s in range(4)]
///     >>> d = comnum.rds_differential_encode(np.tile(np.concatenate(groups), 4))
///     >>> silence = np.zeros(114_000)
///     >>> m = comnum.fm_multiplex(silence, silence, 228e3, rds_bits=d)
///     >>> r = comnum.rds_receive(m, 228e3)
///     >>> hex(r["pi"]), r["ps"]
///     ('0xf201', 'COMNUM  ')
#[pyfunction]
fn rds_receive<'py>(py: Python<'py>, mpx: &Bound<'py, PyAny>, fs: f64) -> PyResult<Bound<'py, PyDict>> {
    let m = fvec(mpx)?;
    let (groups, st) = py.detach(|| rds::receive(&m, fs)).py()?;
    let d = PyDict::new(py);
    d.set_item("pi", st.pi)?;
    d.set_item("pty", st.pty)?;
    d.set_item("tp", st.tp)?;
    d.set_item("ps", st.ps)?;
    d.set_item("radiotext", st.radiotext)?;
    let clock = match st.clock {
        Some(c) => {
            let cd = PyDict::new(py);
            for (k, v) in [
                ("mjd", c.mjd),
                ("year", c.year),
                ("month", c.month),
                ("day", c.day),
                ("hour", c.hour),
                ("minute", c.minute),
            ] {
                cd.set_item(k, v)?;
            }
            cd.set_item("offset_half_hours", c.offset_half_hours)?;
            cd.into_any()
        }
        None => py.None().into_bound(py),
    };
    d.set_item("clock", clock)?;
    let types = PyDict::new(py);
    for (t, counts) in st.group_types.iter().enumerate() {
        for (v, &c) in counts.iter().enumerate() {
            if c > 0 {
                types.set_item(format!("{t}{}", if v == 0 { "A" } else { "B" }), c)?;
            }
        }
    }
    d.set_item("group_types", types)?;
    d.set_item("complete_groups", st.complete_groups)?;
    d.set_item("valid_blocks", st.valid_blocks.0)?;
    d.set_item("blocks", st.valid_blocks.1)?;
    let list = PyList::empty(py);
    for g in groups {
        list.append(g.blocks.to_vec())?;
    }
    d.set_item("groups", list)?;
    Ok(d)
}

/// The samples `x` (at `fs_in`) resampled to `fs_out` by the exact rational ratio of
/// the two rates under the 80 dB Chebyshev-windowed low-pass filter of cut-off
/// `cutoff_hz` (:func:`select_channel` without the removal of the mean and the shift).
///
/// Args:
///     x (array_like): real or complex samples.
///     fs_in (float): the input rate (Hz), a whole number of hertz.
///     fs_out (float): the output rate (Hz), a whole number of hertz.
///     cutoff_hz (float): the cut-off (Hz), at most half the lower rate.
///     n_taps (int): the length of the filter at the input rate.
///
/// Returns:
///     numpy.ndarray: `ceil(len(x) fs_out / fs_in)` samples (complex128).
///
/// Raises:
///     ValueError: for rates that are not whole numbers of hertz, a ratio that reduces to
///         an interpolation by more than 1024, a cut-off out of range, or no tap.
#[pyfunction]
#[pyo3(signature = (x, fs_in, fs_out, cutoff_hz, *, n_taps = 129))]
fn resample_rational<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    fs_in: f64,
    fs_out: f64,
    cutoff_hz: f64,
    n_taps: i64,
) -> PyResult<CArr<'py>> {
    let n = crate::count(n_taps, "n_taps")?;
    Ok(carr(py, comnum::dsp::frontend::resample_rational(&cvec(x)?, fs_in, fs_out, cutoff_hz, n).py()?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fm_modulate, m)?)?;
    m.add_function(wrap_pyfunction!(fm_discriminator, m)?)?;
    m.add_function(wrap_pyfunction!(fm_deemphasis, m)?)?;
    m.add_function(wrap_pyfunction!(fm_preemphasis, m)?)?;
    m.add_function(wrap_pyfunction!(fm_multiplex, m)?)?;
    m.add_function(wrap_pyfunction!(fm_stereo_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rds_group_bits, m)?)?;
    m.add_function(wrap_pyfunction!(rds_differential_encode, m)?)?;
    m.add_function(wrap_pyfunction!(rds_receive, m)?)?;
    m.add_function(wrap_pyfunction!(resample_rational, m)?)?;
    Ok(())
}
