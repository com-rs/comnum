//! OFDM links: multipath channels, resource grid, channel estimation and
//! synchronisation.

use crate::ofdm::PyOfdm;
use crate::rng::PyRng;
use crate::{
    CArr, CMat, FArr, FMat, IArr, IntoPy, carr, cvec, farr, fvec, grid_out, iarr, mat, streams_in, streams_out,
    taps_in, taps_out, uvec,
};
use comnum::channel::fading::Fading;
use comnum::channel::multipath::{Profile, Taps, Tdl, TdlParams, rms_delay_spread};
use comnum::core::signal::Signal;
use comnum::core::units::Hertz;
use comnum::dsp::estimation::LowRank;
use comnum::waveform::ofdm::estimation::{FrequencyInterpolation, PowerDelayProfile, RandomDelayProfile, estimate};
use comnum::waveform::ofdm::grid::{PilotPattern, ResourceGrid};
use comnum::waveform::ofdm::wiener::{Correlation2d, DelaySpectrum, DopplerSpectrum, RobustDesign, Wiener2d};
use comnum::waveform::ofdm::{budget, sync};
use ndarray::Array3;
use num_complex::Complex64 as C64;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

/// The profile of a name, TDL-A to TDL-E scaled to `delay_spread_s` when it is given.
fn named_profile(name: &str, delay_spread_s: Option<f64>) -> PyResult<Profile> {
    let p = name.parse::<Profile>().py()?;
    Ok(delay_spread_s.map_or(p, |ds| p.with_delay_spread(ds)))
}

fn profile_taps(name: &str, delay_spread_s: Option<f64>) -> PyResult<Vec<(f64, f64)>> {
    named_profile(name, delay_spread_s)?.try_taps().py()
}

/// The taps of a profile given by its name (`"eva"`) or as an `(n, 2)` array of
/// `(delay in s, power in dB)`.
pub(crate) fn profile_or_taps(profile: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    if profile.is_instance_of::<PyString>() {
        profile_taps(&profile.extract::<String>()?, None)
    } else {
        taps_in(profile)
    }
}

/// The taps of a power-delay profile of the standards.
///
/// Args:
///     name (str): `"epa"`, `"eva"`, `"etu"` (3GPP TS 36.101, Annex B.2),
///         `"pedestrian_a"` or `"vehicular_a"` (ITU-R M.1225), `"tdla30"`,
///         `"tdlb100"`, `"tdlc300"`, `"tdld30"` (3GPP TS 38.101-4, Annex B.2.1), or
///         `"tdl-a"` to `"tdl-e"` (3GPP TR 38.901, Tables 7.7.2-1 to 7.7.2-5). For
///         TDL-D, TDL-E and TDLD30 the taps are the Rayleigh part; the specular (LOS)
///         row of the first tap is that of `TdlChannel.specular`.
///     delay_spread_s (float | None): the RMS delay spread of TDL-A to TDL-E (s), 100 ns
///         by default (TR 38.901 Table 7.7.3-1); ignored by the other profiles.
///
/// Returns:
///     numpy.ndarray: the taps `(n_taps, 2)`, column 0 the delays (s) and column 1 the
///     powers (dB).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> taps = comnum.power_delay_profile("epa")
///     >>> taps.shape, round(float(taps[-1, 0]) * 1e9)
///     ((7, 2), 410)
#[pyfunction]
#[pyo3(signature = (name, delay_spread_s=None))]
fn power_delay_profile<'py>(py: Python<'py>, name: &str, delay_spread_s: Option<f64>) -> PyResult<FMat<'py>> {
    taps_out(py, profile_taps(name, delay_spread_s)?)
}

/// The RMS delay spread of a power-delay profile.
///
/// Args:
///     taps (array_like): the taps `(n_taps, 2)`, `(delay in s, power in dB)` (as
///         `power_delay_profile` and `cdl_profile` return them).
///
/// Returns:
///     float: the RMS delay spread (s).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.rms_delay_spread(comnum.power_delay_profile("eva")) * 1e9)
///     357
#[pyfunction]
#[pyo3(name = "rms_delay_spread")]
fn py_rms_delay_spread(taps: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(rms_delay_spread(&taps_in(taps)?))
}

/// A tapped-delay-line multipath channel with Clarke/Jakes Doppler (sum of sinusoids of
/// Zheng & Xiao 2003), SISO or MIMO: every tap is Rayleigh with its power, the delays
/// are rounded to the sampling grid and the powers normalised. The first tap of TDL-D,
/// TDL-E and TDLD30 is Ricean: a specular component of Doppler shift ``0.7 doppler_hz``
/// is added (TR 38.901 §7.7.2).
///
/// Args:
///     profile (str | array_like): a profile name (see `power_delay_profile`) or taps
///         `(n_taps, 2)` of `(delay in s, power in dB)`.
///     delay_spread_s (float | None): the RMS delay spread of TDL-A to TDL-E (s), 100 ns
///         by default.
///     fs (float): sampling rate (Hz).
///     doppler_hz (float): maximum Doppler shift (Hz), 0 for a static realisation.
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> ch = comnum.TdlChannel("eva", 15.36e6)
///     >>> y, taps = ch.fade(np.ones(64), comnum.Rng(1))
///     >>> y.shape, int(taps.delays[-1])
///     ((64,), 39)
#[pyclass(name = "TdlChannel", module = "comnum")]
pub struct PyTdl {
    inner: Tdl,
}

/// The taps of one realisation of a `TdlChannel` or a `CdlChannel`: one tap per
/// delay, with its coefficients at every sample on every link.
#[pyclass(name = "ChannelTaps", module = "comnum")]
pub struct PyTaps {
    inner: Taps,
    n_rx: usize,
    n_tx: usize,
}

impl PyTaps {
    /// Wrap the taps of a realisation between `n_tx` and `n_rx` antennas.
    pub(crate) fn new(inner: Taps, n_rx: usize, n_tx: usize) -> PyTaps {
        PyTaps { inner, n_rx, n_tx }
    }
}

#[pymethods]
impl PyTdl {
    #[new]
    #[pyo3(signature = (profile, fs, *, doppler_hz=0.0, n_rx=1, n_tx=1, delay_spread_s=None))]
    fn new(
        profile: &Bound<'_, PyAny>,
        fs: f64,
        doppler_hz: f64,
        n_rx: usize,
        n_tx: usize,
        delay_spread_s: Option<f64>,
    ) -> PyResult<Self> {
        let params = if profile.is_instance_of::<PyString>() {
            named_profile(&profile.extract::<String>()?, delay_spread_s)?.tdl_params(doppler_hz, fs, n_rx, n_tx).py()?
        } else {
            TdlParams { taps: taps_in(profile)?, specular: None, doppler_hz, sample_rate: fs, n_rx, n_tx }
        };
        Ok(PyTdl { inner: Tdl::new(&params).py()? })
    }

    /// tuple[int, float] | None: The specular component of a Ricean first tap: the index
    /// of its tap in `delays` and its normalised power (None for Rayleigh taps only).
    #[getter]
    fn specular(&self) -> Option<(usize, f64)> {
        self.inner.specular()
    }

    /// numpy.ndarray: Delays of the taps (samples, int64).
    #[getter]
    fn delays<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.delays().to_vec())
    }

    /// numpy.ndarray: Normalised powers of the Rayleigh part of the taps (unit sum with
    /// `specular`, float64).
    #[getter]
    fn powers<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.powers().to_vec())
    }

    /// int: Receive antennas.
    #[getter]
    fn n_rx(&self) -> usize {
        self.inner.n_rx
    }

    /// int: Transmit antennas.
    #[getter]
    fn n_tx(&self) -> usize {
        self.inner.n_tx
    }

    /// Apply the channel.
    ///
    /// Args:
    ///     x (array_like): the transmitted streams, `(n,)` for one transmit antenna or
    ///         `(n_tx, n)`.
    ///     rng (Rng): the generator of the realisation.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, ChannelTaps]: the received streams (`(n,)` or
    ///     `(n_rx, n)`) and the taps of the realisation.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn fade(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        mut rng: PyRefMut<'_, PyRng>,
    ) -> PyResult<(Py<PyAny>, PyTaps)> {
        let (rows, one_d) = streams_in(x)?;
        let s = Signal::from_streams(&rows, Hertz(self.inner.sample_rate())).py()?;
        let (y, taps) = self.inner.fade(&s, &mut rng.inner).py()?;
        Ok((streams_out(py, &y, one_d), PyTaps::new(taps, self.inner.n_rx, self.inner.n_tx)))
    }

    fn __repr__(&self) -> String {
        format!("TdlChannel({} taps, n_rx={}, n_tx={})", self.inner.delays().len(), self.inner.n_rx, self.inner.n_tx)
    }
}

#[pymethods]
impl PyTaps {
    /// numpy.ndarray: Delays of the taps (samples, int64).
    #[getter]
    fn delays<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.delays().to_vec())
    }

    /// int: Number of samples of the realisation.
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    /// The coefficients of every tap.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n_taps, n_samples, n_rx, n_tx)` (complex128):
    ///     `coefficients[l, n, r, t]` is tap `l` at sample `n` between transmit antenna
    ///     `t` and receive antenna `r`.
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn coefficients(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (n_taps, n, n_rx, n_tx) = (self.inner.delays().len(), self.inner.n_samples(), self.n_rx, self.n_tx);
        let a =
            ndarray::Array4::from_shape_fn((n_taps, n, n_rx, n_tx), |(l, k, r, t)| self.inner.coefficient(l, k, r, t));
        Ok(a.into_pyarray(py).into_any().unbind())
    }

    /// The frequency response at one sample on the bins of an FFT.
    ///
    /// Args:
    ///     n (int): the sample.
    ///     bins (array_like): the FFT bins (as `Ofdm.bins`).
    ///     fft_len (int): the size of the transform.
    ///
    /// Returns:
    ///     numpy.ndarray: `(len(bins), n_rx, n_tx)` (complex128), a channel matrix per bin.
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn frequency_response<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        bins: &Bound<'py, PyAny>,
        fft_len: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray3<C64>>> {
        let bins = uvec(bins)?;
        let h = self.inner.frequency_response(n, &bins, fft_len);
        let a = Array3::from_shape_vec((bins.len(), self.n_rx, self.n_tx), h)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(a.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "ChannelTaps({} taps, {} samples, n_rx={}, n_tx={})",
            self.inner.delays().len(),
            self.inner.n_samples(),
            self.n_rx,
            self.n_tx
        )
    }
}

/// An OFDM resource grid of `n_subcarriers` × `n_symbols` with pilots (Coleri et al.
/// 2002). A grid is a `(n_subcarriers, n_symbols)` array, a column per OFDM symbol, so
/// that `Ofdm.modulate(grid.map(data))` sends it; the methods that read a grid also take
/// the symbols of `Ofdm.demodulate` (1-D, symbol after symbol).
///
/// Args:
///     n_subcarriers (int): subcarriers of the grid.
///     n_symbols (int): OFDM symbols of the grid.
///     pattern (str): `"block"` (every subcarrier of one symbol in `period`), `"comb"`
///         (one subcarrier in `spacing`, every symbol) or `"lattice"` (one subcarrier
///         in `freq` of one symbol in `time`).
///     period (int): symbol period of the block pattern.
///     spacing (int): subcarrier spacing of the comb pattern.
///     freq (int): subcarrier spacing of the lattice pattern.
///     time (int): symbol spacing of the lattice pattern.
///     seed (int): seed of the pilot symbols.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> grid = comnum.ResourceGrid(12, 4, pattern="comb", spacing=4)
///     >>> grid.n_data, grid.pilot_positions().shape
///     (36, (12, 2))
#[pyclass(name = "ResourceGrid", module = "comnum")]
pub struct PyGrid {
    pub(crate) inner: ResourceGrid,
}

impl PyGrid {
    /// The resource elements of a grid, symbol after symbol (the order of the crate):
    /// a `(n_subcarriers, n_symbols)` array or a 1-D array already in that order; and
    /// whether the grid was 2-D.
    fn res_in(&self, x: &Bound<'_, PyAny>) -> PyResult<(Vec<C64>, bool)> {
        let a = crate::carray(x)?;
        let shape = [self.inner.n_subcarriers, self.inner.n_symbols];
        match a.ndim() {
            1 => Ok((a.iter().copied().collect(), false)),
            2 if a.shape() == shape => Ok((a.t().iter().copied().collect(), true)),
            _ => Err(PyValueError::new_err(format!(
                "expected a grid of shape ({}, {}) or its {} symbols, got the shape {:?}",
                shape[0],
                shape[1],
                shape[0] * shape[1],
                a.shape()
            ))),
        }
    }
}

#[pymethods]
impl PyGrid {
    #[new]
    #[pyo3(signature = (n_subcarriers, n_symbols, *, pattern="lattice", period=4, spacing=4, freq=3, time=4, seed=0))]
    #[allow(clippy::too_many_arguments, reason = "the grid and the parameters of its pilot pattern, by keyword")]
    fn new(
        n_subcarriers: usize,
        n_symbols: usize,
        pattern: &str,
        period: usize,
        spacing: usize,
        freq: usize,
        time: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let pattern = match pattern.to_ascii_lowercase().as_str() {
            "block" => PilotPattern::block(period),
            "comb" => PilotPattern::comb(spacing),
            "lattice" => PilotPattern::lattice(freq, time),
            "none" => Ok(PilotPattern::None),
            _ => Err(comnum::Error::Config(format!("unknown pilot pattern {pattern:?}: block, comb, lattice or none"))),
        }
        .py()?;
        Ok(PyGrid { inner: ResourceGrid { n_subcarriers, n_symbols, pattern, seed } })
    }

    /// int: Data REs per grid.
    #[getter]
    fn n_data(&self) -> usize {
        self.inner.n_data()
    }

    /// int: Subcarriers of the grid.
    #[getter]
    fn n_subcarriers(&self) -> usize {
        self.inner.n_subcarriers
    }

    /// int: OFDM symbols of the grid.
    #[getter]
    fn n_symbols(&self) -> usize {
        self.inner.n_symbols
    }

    /// The pilot resource elements.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n_pilots, 2)` (int64), the subcarrier `k` and the symbol `l`
    ///     of every pilot.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn pilot_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
        mat(py, self.inner.pilot_positions().into_iter().map(|(k, l)| vec![k as i64, l as i64]).collect())
    }

    /// Whether a resource element carries a pilot.
    ///
    /// Args:
    ///     k (int): the subcarrier.
    ///     l (int): the OFDM symbol.
    ///
    /// Returns:
    ///     bool: whether `(k, l)` is a pilot.
    fn is_pilot(&self, k: usize, l: usize) -> bool {
        self.inner.is_pilot(k, l)
    }

    /// The grid with `data` and the pilots.
    ///
    /// Args:
    ///     data (array_like): `n_data` data symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: the grid `(n_subcarriers, n_symbols)` (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn map<'py>(&self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<CMat<'py>> {
        grid_out(py, &self.inner.map(&cvec(data)?).py()?, self.inner.n_subcarriers)
    }

    /// The data resource elements of a grid.
    ///
    /// Args:
    ///     grid (array_like): a whole grid `(n_subcarriers, n_symbols)`, or its symbols
    ///         as `Ofdm.demodulate` gives them (symbol after symbol).
    ///
    /// Returns:
    ///     numpy.ndarray: the `n_data` data symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn extract<'py>(&self, py: Python<'py>, grid: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let (g, _) = self.res_in(grid)?;
        Ok(carr(py, self.inner.extract(&g).py()?))
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourceGrid({} × {}, {} data REs)",
            self.inner.n_subcarriers,
            self.inner.n_symbols,
            self.inner.n_data()
        )
    }
}

/// Channel estimation on a resource grid: LS at the pilots, then `"linear"`
/// interpolation, or `"lmmse"` across the subcarriers (van de Beek et al. 1995) with
/// the power-delay profile `delays`, `powers`, or the robust uniform profile over
/// `max_delay` samples (Edfors et al. 1998); linear in time.
///
/// Args:
///     grid (ResourceGrid): the grid and its pilots.
///     rx (array_like): the received grid `(n_subcarriers, n_symbols)`, or the symbols
///         of `Ofdm.demodulate` (symbol after symbol).
///     noise_var (float): noise variance per resource element.
///     method (str): `"linear"`, `"lmmse"`, or the interpolations compared by Coleri et
///         al. (2002): `"second_order"`, `"spline"`, `"lowpass"` (MATLAB `interp`) and
///         `"time_domain"` (zero padding of the IDFT, pilots on 0, L, 2L, …).
///     delays (array_like, optional): delays of the profile (samples of the IFFT of
///         `n_fft`).
///     powers (array_like, optional): powers of the profile (linear).
///     max_delay (int, optional): the length of the uniform profile (samples).
///     n_fft (int, optional): the size of the IFFT (needed by `"lmmse"`, with `delays`
///         and `powers` or `max_delay`).
///
/// Returns:
///     numpy.ndarray: the channel estimate at every resource element (complex128), in
///     the layout of `rx`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (grid, rx, noise_var, method="linear", *, delays=None, powers=None, max_delay=None, n_fft=None))]
#[allow(clippy::too_many_arguments, reason = "the grid, the method and its profile, by keyword")]
fn ofdm_estimate<'py>(
    py: Python<'py>,
    grid: &PyGrid,
    rx: &Bound<'py, PyAny>,
    noise_var: f64,
    method: &str,
    delays: Option<&Bound<'py, PyAny>>,
    powers: Option<&Bound<'py, PyAny>>,
    max_delay: Option<usize>,
    n_fft: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let pdp = match (delays, powers, max_delay, n_fft) {
        (None, None, None, None) => None,
        (Some(d), Some(p), None, Some(n)) => Some(PowerDelayProfile::new(fvec(d)?, &fvec(p)?, n).py()?),
        (None, None, Some(m), Some(n)) => Some(PowerDelayProfile::uniform(m, n).py()?),
        _ => {
            return Err(PyValueError::new_err(
                "give delays, powers and n_fft (a measured profile), or max_delay and n_fft (the uniform profile)",
            ));
        }
    };
    let m = FrequencyInterpolation::from_name(method, pdp).py()?;
    let (r, two_d) = grid.res_in(rx)?;
    let h = py.detach(|| estimate(&grid.inner, &r, noise_var, &m)).py()?;
    if two_d {
        Ok(grid_out(py, &h, grid.inner.n_subcarriers)?.into_any().unbind())
    } else {
        Ok(carr(py, h).into_any().unbind())
    }
}

/// The preamble of Schmidl & Cox (1997): two identical halves.
///
/// Args:
///     ofdm (Ofdm): the modulator.
///     seed (int): seed of the symbols.
///
/// Returns:
///     numpy.ndarray: one OFDM block with its prefix (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
#[pyfunction]
#[pyo3(signature = (ofdm, *, seed=0))]
fn ofdm_preamble<'py>(py: Python<'py>, ofdm: &PyOfdm, seed: u64) -> PyResult<CArr<'py>> {
    Ok(carr(py, sync::preamble(&ofdm.inner, seed).py()?))
}

/// The timing metric of Schmidl & Cox, `|P(d)|²/R(d)²`.
///
/// Args:
///     r (array_like): the received samples.
///     half (int): the length of a half of the preamble (samples).
///
/// Returns:
///     numpy.ndarray: the metric at every delay (float64).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
#[pyfunction]
fn schmidl_cox<'py>(py: Python<'py>, r: &Bound<'py, PyAny>, half: usize) -> PyResult<FArr<'py>> {
    Ok(farr(py, sync::schmidl_cox(&cvec(r)?, half)))
}

/// Timing and fractional frequency offset of the preamble of `ofdm` in `r` (Schmidl &
/// Cox 1997).
///
/// Args:
///     r (array_like): the received samples.
///     ofdm (Ofdm): the modulator of the preamble.
///
/// Returns:
///     dict: ``timing`` (the middle of the plateau, inside the cyclic prefix,
///     samples), ``cfo`` (subcarrier spacings), ``peak`` (the metric).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn ofdm_synchronize<'py>(py: Python<'py>, r: &Bound<'py, PyAny>, ofdm: &PyOfdm) -> PyResult<Bound<'py, PyDict>> {
    let e = sync::synchronize(&cvec(r)?, &ofdm.inner).py()?;
    let d = PyDict::new(py);
    d.set_item("timing", e.timing)?;
    d.set_item("cfo", e.cfo)?;
    d.set_item("peak", e.peak)?;
    Ok(d)
}

/// The SNR-loss budget of an OFDM receiver (Speth, Fechtel, Fock & Meyr 1999): the
/// largest residual impairments for an allowed loss of SNR.
///
/// Args:
///     snr_db (float): the SNR per subcarrier (dB).
///     loss_db (float): the allowed loss of SNR (dB).
///
/// Returns:
///     dict: ``noise`` (the largest total power of the residual noises, relative to a
///     subcarrier, eq. (51)), ``cfo`` (the largest carrier frequency offset in
///     subcarrier spacings, eq. (43)), ``doppler`` (the largest ``f_D T_u``, Fig. 7) and
///     ``estimator_gain`` (the gain a channel estimator needs, eq. (20)).
///
/// Raises:
///     ValueError: for a loss that is not positive or a non-finite SNR.
///
/// Example:
///     >>> import comnum
///     >>> b = comnum.ofdm_budget(20.0, 1.0)
///     >>> round(b["doppler"], 4), round(b["estimator_gain"], 2)
///     (0.0354, 3.86)
#[pyfunction]
fn ofdm_budget<'py>(py: Python<'py>, snr_db: f64, loss_db: f64) -> PyResult<Bound<'py, PyDict>> {
    if !(loss_db > 0.0 && loss_db.is_finite() && snr_db.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "the loss must be positive and the SNR finite, got {loss_db} dB and {snr_db} dB"
        )));
    }
    let snr = 10f64.powf(snr_db / 10.0);
    let d = PyDict::new(py);
    d.set_item("noise", budget::noise_budget(snr, loss_db))?;
    d.set_item("cfo", budget::max_cfo(snr, loss_db))?;
    d.set_item("doppler", budget::max_doppler(snr, loss_db))?;
    d.set_item("estimator_gain", budget::estimator_gain(loss_db))?;
    Ok(d)
}

/// The power of the intersymbol and intercarrier interference of an FFT window
/// ``offset`` samples late (positive) or early (negative) of the end of the guard
/// interval, relative to a subcarrier (Speth et al. 1999, eqs. (30)-(31)).
///
/// Args:
///     offset (float): the timing offset (samples).
///     n_fft (int): the FFT size.
///     guard (int): the guard interval (samples).
///     delays (array_like): the delays of the taps (samples).
///     powers (array_like): their powers (linear, of unit sum).
///
/// Returns:
///     float: the interference power.
///
/// Raises:
///     ValueError: for an empty FFT, delays and powers of different lengths, or a
///         negative delay or power.
#[pyfunction]
#[pyo3(signature = (offset, n_fft, guard=0, delays=None, powers=None))]
fn ofdm_timing_noise(
    offset: f64,
    n_fft: usize,
    guard: usize,
    delays: Option<&Bound<'_, PyAny>>,
    powers: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let d = delays.map(fvec).transpose()?.unwrap_or_else(|| vec![0.0]);
    let p = powers.map(fvec).transpose()?.unwrap_or_else(|| vec![1.0; d.len()]);
    if d.len() != p.len() {
        return Err(PyValueError::new_err(format!("{} delays and {} powers", d.len(), p.len())));
    }
    let taps: Vec<(f64, f64)> = d.into_iter().zip(p).collect();
    budget::timing_noise_power(offset, n_fft, guard, &taps).py()
}

/// The two-dimensional Wiener interpolator of pilots in time and frequency (Hoeher,
/// Kaiser & Robertson 1997), for a WSSUS channel of uniform delay spectrum: every
/// position is estimated from the ``n_taps`` nearest pilots (the weighted distance
/// ``|dl| fd_ts + |dk| tau_df``) by the linear MMSE filter of their eq. (9).
///
/// Args:
///     pilots (list[tuple[int, int]]): the ``(symbol, subcarrier)`` positions of the pilots.
///     fd_ts (float): the maximum Doppler over the symbol rate, ``f_D T_s``.
///     tau_df (float): the one-sided maximum delay over the useful duration,
///         ``tau_max Delta F`` (delays uniform on ``[-tau_max, tau_max]``).
///     noise_to_pilot (float): the noise of the LS estimates, ``sigma^2 / E_p``.
///     n_taps (int): the pilots per position.
///     doppler (str): ``"uniform"`` (their eq. (15)) or ``"jakes"``.
///
/// Raises:
///     ValueError: without pilots, for no tap or more taps than pilots, a negative noise
///         or Doppler, or an unknown Doppler spectrum.
///
/// Example:
///     >>> import comnum
///     >>> pilots = [(l, k) for l in range(0, 12, 3) for k in range(0, 24, 6)]
///     >>> w = comnum.Wiener2d(pilots, 0.0832, 0.0417, 0.01, 10)
///     >>> bool(w.mse([(0, 0)])[0] < 0.01)
///     True
#[pyclass(name = "Wiener2d", module = "comnum")]
#[derive(Clone)]
pub struct PyWiener2d {
    inner: Wiener2d,
}

#[pymethods]
impl PyWiener2d {
    #[new]
    #[pyo3(signature = (pilots, fd_ts, tau_df, noise_to_pilot, n_taps, doppler = "uniform"))]
    fn new(
        pilots: Vec<(usize, usize)>,
        fd_ts: f64,
        tau_df: f64,
        noise_to_pilot: f64,
        n_taps: usize,
        doppler: &str,
    ) -> PyResult<Self> {
        let doppler = match doppler.to_ascii_lowercase().as_str() {
            "uniform" => DopplerSpectrum::Uniform,
            "jakes" => DopplerSpectrum::Jakes,
            other => {
                return Err(PyValueError::new_err(format!("unknown Doppler spectrum {other:?}: uniform or jakes")));
            }
        };
        let corr = Correlation2d { fd_ts, doppler, delay: DelaySpectrum::Uniform { tau_df } };
        Ok(PyWiener2d { inner: Wiener2d::new(pilots, corr, noise_to_pilot, n_taps).py()? })
    }

    /// The estimates at the positions ``targets`` from the LS estimates at the pilots.
    ///
    /// Args:
    ///     h_ls (array_like): one LS estimate per pilot, in their order (complex).
    ///     targets (list[tuple[int, int]]): the ``(symbol, subcarrier)`` positions.
    ///
    /// Returns:
    ///     numpy.ndarray: the estimates (complex128).
    ///
    /// Raises:
    ///     ValueError: unless ``h_ls`` has one value per pilot.
    fn estimate<'py>(
        &self,
        py: Python<'py>,
        h_ls: &Bound<'py, PyAny>,
        targets: Vec<(usize, usize)>,
    ) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.estimate(&cvec(h_ls)?, &targets).py()?))
    }

    /// The mean square error of the estimate at every position (their eq. (12)).
    ///
    /// Args:
    ///     targets (list[tuple[int, int]]): the ``(symbol, subcarrier)`` positions.
    ///
    /// Returns:
    ///     numpy.ndarray: the MSE of every position (float64).
    ///
    /// Raises:
    ///     ValueError: for a singular system.
    fn mse<'py>(&self, py: Python<'py>, targets: Vec<(usize, usize)>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.mse(&targets).py()?))
    }

    /// The pilots (indices into the list) used for a position.
    ///
    /// Args:
    ///     symbol (int): the symbol.
    ///     subcarrier (int): the subcarrier.
    ///
    /// Returns:
    ///     list[int]: the indices of the ``n_taps`` nearest pilots.
    fn taps(&self, symbol: usize, subcarrier: usize) -> Vec<usize> {
        self.inner.taps(symbol, subcarrier)
    }

    fn __repr__(&self) -> String {
        format!("Wiener2d({} pilots, {} taps)", self.inner.pilots().len(), self.inner.n_taps())
    }
}

/// The robust channel estimator of Li, Cimini & Sollenberger (1998): designed for the
/// worst channel of delay spread ``tmax_ts`` (over the useful duration) and maximum
/// Doppler ``fmax_tf`` (over the block rate), a uniform delay profile on the taps
/// ``0`` to ``floor(K tmax_ts)`` of the DFT grid and a flat band-limited Doppler spectrum.
///
/// Args:
///     k (int): the tones.
///     tmax_ts (float): the design delay spread over the useful duration.
///     fmax_tf (float): the design Doppler over the block rate, in ``[0, 1/2]``.
///
/// Raises:
///     ValueError: without tones, for a negative spread or a Doppler outside ``[0, 1/2]``.
///
/// Example:
///     >>> import comnum
///     >>> d = comnum.RobustDesign(120, 20 / 160, 40 * 200e-6)
///     >>> d.k_o, round(10 * __import__("math").log10(d.mse(0.1)), 1)
///     (16, -27.7)
#[pyclass(name = "RobustDesign", module = "comnum")]
#[derive(Clone)]
pub struct PyRobustDesign {
    inner: RobustDesign,
}

#[pymethods]
impl PyRobustDesign {
    #[new]
    fn new(k: usize, tmax_ts: f64, fmax_tf: f64) -> PyResult<Self> {
        Ok(PyRobustDesign { inner: RobustDesign::new(k, tmax_ts, fmax_tf).py()? })
    }

    /// int: The taps of the design delay spread on the DFT grid, ``K_o``.
    #[getter]
    fn k_o(&self) -> usize {
        self.inner.k_o
    }

    /// The MSE of the infinite-length estimator (their eq. (57)), normalised to the
    /// power of the channel.
    ///
    /// Args:
    ///     rho (float): the noise to signal ratio.
    ///
    /// Returns:
    ///     float: the MSE.
    fn mse(&self, rho: f64) -> f64 {
        self.inner.mse(rho)
    }

    /// The coefficients and the MSE of the ``taps``-tap temporal filter (eqs. (69)-(74)).
    ///
    /// Args:
    ///     taps (int): the length of the filter.
    ///     rho (float): the noise to signal ratio.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, float]: the coefficients and the MSE.
    ///
    /// Raises:
    ///     ValueError: for no tap.
    fn fir<'py>(&self, py: Python<'py>, taps: usize, rho: f64) -> PyResult<(FArr<'py>, f64)> {
        let (c, mse) = self.inner.fir(taps, rho).py()?;
        Ok((farr(py, c), mse))
    }

    fn __repr__(&self) -> String {
        format!("RobustDesign(k={}, k_o={}, omega_d={})", self.inner.k, self.inner.k_o, self.inner.omega_d)
    }
}

/// The noise enhancement of LS estimates over a constellation, ``beta = E|x|^2 E|1/x|^2``
/// (Edfors et al. 1998): 1 for a constant modulus, 17/9 for 16-QAM.
///
/// Args:
///     points (array_like): the points of the constellation (complex), equiprobable.
///
/// Returns:
///     float: ``beta``.
///
/// Raises:
///     ValueError: for no point or a point at the origin.
///
/// Example:
///     >>> import comnum
///     >>> round(comnum.ls_noise_factor(comnum.Constellation("qam", 16).points) * 9, 9)
///     17.0
#[pyfunction]
fn ls_noise_factor(points: &Bound<'_, PyAny>) -> PyResult<f64> {
    comnum::dsp::estimation::ls_noise_factor(&cvec(points)?).py()
}

/// The correlation matrix of ``n`` adjacent subcarriers of a channel of ``paths``
/// impulses with delays uniform over ``[0, max_delay]`` samples (Edfors et al. 1998,
/// Sec. II): equal powers (``tau_rms=None``, their uniform channel) or ``exp(-tau /
/// tau_rms)``, the first impulse at 0 with ``first_at_zero`` (their synchronized channel).
///
/// Args:
///     n (int): the subcarriers.
///     paths (int): the impulses.
///     max_delay (float): the largest delay (samples of the IFFT).
///     tau_rms (float, optional): the decay of the powers (samples).
///     first_at_zero (bool): the first impulse at delay 0.
///     n_fft (int, optional): the IFFT size, ``n`` by default.
///
/// Returns:
///     numpy.ndarray: the ``n x n`` correlation (complex128), of unit diagonal.
///
/// Raises:
///     ValueError: for no subcarrier or impulse, a negative delay or a non-positive decay.
#[pyfunction]
#[pyo3(signature = (n, paths, max_delay, tau_rms=None, first_at_zero=false, n_fft=None))]
fn random_delay_correlation<'py>(
    py: Python<'py>,
    n: usize,
    paths: usize,
    max_delay: f64,
    tau_rms: Option<f64>,
    first_at_zero: bool,
    n_fft: Option<usize>,
) -> PyResult<CMat<'py>> {
    if n == 0 || paths == 0 || max_delay < 0.0 || tau_rms.is_some_and(|t| t <= 0.0) {
        return Err(PyValueError::new_err(
            "random_delay_correlation: n and paths must be positive, max_delay non-negative, tau_rms positive",
        ));
    }
    let p = RandomDelayProfile { paths, max_delay, tau_rms, first_at_zero, n_fft: n_fft.unwrap_or(n) };
    let r = p.correlation_matrix(n);
    mat(py, r.outer_iter().map(|row| row.to_vec()).collect())
}

/// The rank-p LMMSE estimator of Edfors et al. (1998) for the correlation ``r_hh`` of
/// ``N`` values: ``h_p = U D_p U^H h_ls``, ``D_p`` diagonal of ``delta_k = lambda_k /
/// (lambda_k + noise)`` for ``k < p`` and of 0 beyond.
///
/// Args:
///     r_hh (array_like): the ``N x N`` correlation (complex, Hermitian).
///     noise (float): the noise of the LS estimates, ``beta / SNR``.
///     rank (int): the rank ``p``.
///
/// Raises:
///     ValueError: unless ``r_hh`` is square, for a rank of 0 or above ``N``, or a
///         negative noise.
///
/// Example:
///     >>> import comnum
///     >>> r = comnum.random_delay_correlation(64, 5, 4.0)
///     >>> est = comnum.LowRank(r, 17 / 9 / 100, 8)
///     >>> round(est.mse(r, 17 / 9 / 100) / 64, 4)
///     0.002
#[pyclass(name = "LowRank", module = "comnum")]
#[derive(Clone)]
pub struct PyLowRank {
    inner: LowRank,
}

/// A square complex matrix from a 2-D array.
fn square_in(x: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<C64>> {
    let (rows, _) = streams_in(x)?;
    let n = rows.len();
    let flat: Vec<C64> = rows.into_iter().flatten().collect();
    ndarray::Array2::from_shape_vec((n, flat.len() / n.max(1)), flat)
        .map_err(|e| PyValueError::new_err(format!("a matrix: {e}")))
}

#[pymethods]
impl PyLowRank {
    #[new]
    fn new(r_hh: &Bound<'_, PyAny>, noise: f64, rank: usize) -> PyResult<Self> {
        Ok(PyLowRank { inner: LowRank::new(&square_in(r_hh)?, noise, rank).py()? })
    }

    /// numpy.ndarray: The eigenvalues of the correlation, decreasing.
    #[getter]
    fn eigenvalues<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.eigenvalues().to_vec())
    }

    /// The estimate from the LS estimates (their eq. (9)).
    ///
    /// Args:
    ///     h_ls (array_like): the ``N`` LS estimates (complex).
    ///
    /// Returns:
    ///     numpy.ndarray: the estimate (complex128).
    ///
    /// Raises:
    ///     ValueError: unless ``h_ls`` has ``N`` values.
    fn estimate<'py>(&self, py: Python<'py>, h_ls: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.estimate(&cvec(h_ls)?).py()?))
    }

    /// The mean square error ``E||h - h_p||^2`` on a channel of correlation ``r_true``
    /// with LS estimates of noise ``noise_true`` (their eq. (10)).
    ///
    /// Args:
    ///     r_true (array_like): the correlation of the channel.
    ///     noise_true (float): ``beta / SNR`` of the channel.
    ///
    /// Returns:
    ///     float: the error, summed over the ``N`` values.
    ///
    /// Raises:
    ///     ValueError: unless ``r_true`` has the size of the design.
    fn mse(&self, r_true: &Bound<'_, PyAny>, noise_true: f64) -> PyResult<f64> {
        self.inner.mse(&square_in(r_true)?, noise_true).py()
    }

    /// The floor of the error at infinite SNR (their eq. (11)).
    ///
    /// Args:
    ///     r_true (array_like): the correlation of the channel.
    ///
    /// Returns:
    ///     float: the floor, summed over the ``N`` values.
    ///
    /// Raises:
    ///     ValueError: for an input that is not a matrix of numbers, or a matrix of
    ///         another size than the design.
    fn mse_floor(&self, r_true: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.inner.mse_floor(&square_in(r_true)?).py()
    }

    fn __repr__(&self) -> String {
        format!("LowRank({} values)", self.inner.eigenvalues().len())
    }
}

/// Register the OFDM link bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTdl>()?;
    m.add_class::<PyTaps>()?;
    m.add_class::<PyGrid>()?;
    m.add_class::<PyWiener2d>()?;
    m.add_class::<PyRobustDesign>()?;
    m.add_class::<PyLowRank>()?;
    m.add_function(wrap_pyfunction!(ls_noise_factor, m)?)?;
    m.add_function(wrap_pyfunction!(random_delay_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(power_delay_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_rms_delay_spread, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_preamble, m)?)?;
    m.add_function(wrap_pyfunction!(schmidl_cox, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_budget, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_timing_noise, m)?)?;
    Ok(())
}
