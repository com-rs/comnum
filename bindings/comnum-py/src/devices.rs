//! Noise sources and optoelectronic devices: AWGN, phase noise, lasers, modulators,
//! photodiodes, the 90° hybrid and the coherent receiver.

use crate::rng::PyRng;
use crate::{CArr, FArr, IntoPy, carr, count, cvec, farr, fvec, py_bool};
use comnum::channel::awgn as awgn_ch;
use comnum::dsp::filter::LowPass;
use comnum::hardware::optical::{self as devices, Iqm, Laser, Mzm, Photodiode};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Add circular complex Gaussian noise for an SNR of `snr_db` within the bandwidth
/// `bandwidth` of a signal sampled at `fs` (OptiCommPy `awgn`).
///
/// The model [1]_ is
///
/// .. math::
///
///     y[k] = x[k] + n[k], \qquad n[k] \sim \mathcal{CN}(0, \sigma^2), \qquad
///     \sigma^2 = \frac{f_s}{B} \, \frac{P_x}{\mathrm{SNR}},
///
/// with :math:`P_x` the mean power of `x`.
///
/// Args:
///     x (array_like): the signal.
///     snr_db (float): the SNR (dB).
///     rng (Rng): the generator.
///     fs (float): sampling rate (Hz).
///     bandwidth_hz (float): the bandwidth :math:`B` of the SNR (Hz); with the default
///         `fs = 1`, `bandwidth_hz = 1`, the SNR is per sample.
///
/// Returns:
///     numpy.ndarray: the noisy signal (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// References:
///     .. [1] J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed.,
///        McGraw-Hill, 2008.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> x = np.ones(100_000, dtype=complex)
///     >>> y = comnum.awgn(x, 10.0, comnum.Rng(1))
///     >>> round(comnum.snr_db(y, x))
///     10
#[pyfunction]
#[pyo3(signature = (x, snr_db, rng, *, fs = 1.0, bandwidth_hz = 1.0))]
fn awgn<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    snr_db: f64,
    mut rng: PyRefMut<'py, PyRng>,
    fs: f64,
    bandwidth_hz: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, awgn_ch::awgn(&cvec(x)?, snr_db, fs, bandwidth_hz, &mut rng.inner)))
}

/// Add noise for an :math:`E_b/N_0` of `ebn0_db`, for :math:`b` bits per symbol at
/// :math:`\mathrm{sps}` samples per symbol.
///
/// The model [1]_ is
///
/// .. math::
///
///     y[k] = x[k] + n[k], \qquad n[k] \sim \mathcal{CN}(0, \sigma^2), \qquad
///     \sigma^2 = \frac{P_x \cdot \mathrm{sps}}{b \cdot E_b/N_0},
///
/// with :math:`P_x` the power of the signal: ``power`` when given, its expected power
/// (1 for the points of a normalised constellation), else the mean power measured on
/// `x` (OptiCommPy's convention). Measured on a short vector, the power follows the
/// symbols drawn and the noise with it: with 16 symbols of 16-QAM at 10 dB the BER is
/// 7 % above the exact one; give ``power`` for short vectors. The blocks of a chain
/// (``AwgnEbn0``) use the expected power.
///
/// Args:
///     x (array_like): the signal.
///     ebn0_db (float): :math:`E_b/N_0` (dB).
///     bits_per_symbol (float): information bits per symbol :math:`b` (fractional for a
///         coded or shaped signal).
///     rng (Rng): the generator.
///     sps (float): samples per symbol.
///     power (float, optional): the power :math:`P_x` the noise refers to (default: the
///         mean power of `x`).
///
/// Returns:
///     numpy.ndarray: the noisy signal (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// References:
///     .. [1] J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed.,
///        McGraw-Hill, 2008.
///
/// Examples:
///     >>> import comnum
///     >>> c = comnum.Constellation("qam", 4).normalized()
///     >>> b = comnum.bits(2 * 50_000, comnum.Rng(1))
///     >>> y = comnum.awgn_ebn0(c.map_bits(b), 6.0, 2, comnum.Rng(2))
///     >>> abs(comnum.ber(c.demap_bits(y), b) / comnum.theory_ber(4, 6.0) - 1) < 0.15
///     True
#[pyfunction]
#[pyo3(signature = (x, ebn0_db, bits_per_symbol, rng, *, sps = 1.0, power = None))]
fn awgn_ebn0<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    ebn0_db: f64,
    bits_per_symbol: f64,
    mut rng: PyRefMut<'py, PyRng>,
    sps: f64,
    power: Option<f64>,
) -> PyResult<CArr<'py>> {
    let x = cvec(x)?;
    let y = match power {
        Some(p) => {
            if !(p.is_finite() && p > 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("power must be positive, got {p}")));
            }
            let mut y = x;
            awgn_ch::add_noise(&mut y, awgn_ch::ebn0_noise_variance(p, ebn0_db, bits_per_symbol, sps), &mut rng.inner);
            y
        }
        None => awgn_ch::awgn_ebn0(&x, ebn0_db, bits_per_symbol, sps, &mut rng.inner),
    };
    Ok(carr(py, y))
}

/// The samples of a receiver whose sampling clock is off by ``ppm`` parts per million.
///
/// Sample :math:`m` is taken at the instant :math:`m (1 + \zeta)` of ``x`` ([1]_,
/// eqs. (12)-(13)):
///
/// .. math::
///
///     y[m] = x\big(m (1 + \zeta)\big), \qquad \zeta = 10^{-6}\, \mathrm{ppm},
///
/// interpolated by the cubic Lagrange kernel, zero beyond the end.
///
/// Args:
///     x (array_like): the samples (complex).
///     ppm (float): the offset of the sampling period, positive for a slower clock.
///
/// Returns:
///     numpy.ndarray: as many samples as ``x`` (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers.
///
/// References:
///     .. [1] M. Speth, S. A. Fechtel, G. Fock and H. Meyr, "Optimum receiver design for
///        wireless broad-band systems using OFDM — Part I," *IEEE Trans. Commun.*,
///        47(11), 1668–1677, 1999.
#[pyfunction]
fn clock_offset<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, ppm: f64) -> PyResult<CArr<'py>> {
    Ok(carr(py, comnum::hardware::oscillator::sampling_offset(&cvec(x)?, ppm * 1e-6)))
}

/// Wiener phase noise of a laser: a random walk starting at 0 whose increments have
/// the variance :math:`2\pi \Delta\nu / f_s`:
///
/// .. math::
///
///     \phi_{n+1} = \phi_n + \Delta_n, \qquad \Delta_n \sim \mathcal{N}(0, 2\pi \Delta\nu / f_s),
///     \qquad \phi_0 = 0.
///
/// Args:
///     linewidth_hz (float): the linewidth :math:`\Delta\nu` (Hz).
///     n (int): number of samples.
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator.
///
/// Returns:
///     numpy.ndarray: the phase of every sample (rad).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
#[pyfunction]
fn phase_noise<'py>(
    py: Python<'py>,
    linewidth_hz: f64,
    n: i64,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
) -> PyResult<FArr<'py>> {
    let n = count(n, "n")?;
    Ok(farr(py, comnum::hardware::oscillator::phase_noise(linewidth_hz, n, 1.0 / fs, &mut rng.inner).py()?))
}

/// The field of a laser: power, Wiener phase noise, relative intensity noise and
/// frequency offset (OptiCommPy `basicLaserModel`).
///
/// Sample :math:`k` of the field is
///
/// .. math::
///
///     E[k] = \sqrt{P (1 + r_k)}\, e^{j(\phi_k + 2\pi \Delta f k / f_s)},
///
/// with :math:`\phi_k` the Wiener phase of the linewidth (`phase_noise`), :math:`\Delta f`
/// the frequency offset and :math:`r_k \sim \mathcal{N}(0, \mathrm{rin\_var})` the relative
/// intensity noise, clipped at -1.
///
/// Args:
///     n (int): number of samples.
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator.
///     power_dbm (float): output power (dBm).
///     linewidth_hz (float): linewidth (Hz).
///     rin_var (float): variance of the relative intensity noise.
///     freq_shift_hz (float): frequency offset (Hz).
///
/// Returns:
///     numpy.ndarray: `n` samples of the field (√W, complex128).
///
/// Raises:
///     ValueError: for a negative count.
#[pyfunction]
#[pyo3(signature = (
    n,
    fs,
    rng,
    *,
    power_dbm = Laser::default().power_dbm,
    linewidth_hz = Laser::default().linewidth_hz,
    rin_var = Laser::default().rin_var,
    freq_shift_hz = Laser::default().freq_shift_hz,
))]
#[allow(clippy::too_many_arguments, reason = "the fields of the laser, by keyword")]
fn laser<'py>(
    py: Python<'py>,
    n: i64,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
    power_dbm: f64,
    linewidth_hz: f64,
    rin_var: f64,
    freq_shift_hz: f64,
) -> PyResult<CArr<'py>> {
    let n = count(n, "n")?;
    Ok(carr(py, Laser { power_dbm, linewidth_hz, rin_var, freq_shift_hz }.field(n, fs, &mut rng.inner).py()?))
}

/// Phase modulator (OptiCommPy `pm`).
///
/// The output field is
///
/// .. math::
///
///     E_{\mathrm{out}} = E\, e^{j\pi u/V_\pi}.
///
/// Args:
///     e (array_like): the optical field (√W).
///     u (array_like): the driving voltage (V), one value per sample.
///     vpi (float): the half-wave voltage :math:`V_\pi` (V).
///
/// Returns:
///     numpy.ndarray: the modulated field (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (e, u, *, vpi = 2.0))]
fn phase_modulator<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    vpi: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, devices::phase_modulator(&cvec(e)?, &fvec(u)?, vpi).py()?))
}

/// Push-pull Mach–Zehnder modulator (OptiCommPy `mzm`):
///
/// .. math::
///
///     E_{\mathrm{out}} = E \left[ c_I \cos\theta + j c_Q \sin\theta \right], \qquad
///     \theta = \frac{\pi (u + V_b)}{2 V_\pi},
///
/// with the arm coefficients of the extinction ratio,
/// :math:`c_{I,Q} = (\sqrt{1+\gamma} \pm \sqrt{1-\gamma})/2`,
/// :math:`\gamma = 2\sqrt{\mathrm{ER}}/(\mathrm{ER}+1)` (ER linear; :math:`\cos\theta` for
/// an infinite one); the bias :math:`-V_\pi` is the null (carrier suppression), :math:`-V_\pi/2` the
/// quadrature point.
///
/// Args:
///     e (array_like): the optical field (√W).
///     u (array_like): the driving voltage (V), one value per sample.
///     vpi (float): the half-wave voltage :math:`V_\pi` (V).
///     vb (float): the bias voltage :math:`V_b` (V).
///     er_db (float): the extinction ratio (dB).
///
/// Returns:
///     numpy.ndarray: the modulated field (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (e, u, *, vpi = Mzm::default().vpi, vb = Mzm::default().vb, er_db = Mzm::default().er_db))]
fn mzm<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    vpi: f64,
    vb: f64,
    er_db: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Mzm { vpi, vb, er_db }.modulate(&cvec(e)?, &fvec(u)?).py()?))
}

/// IQ modulator (OptiCommPy `iqm`): two MZMs driven by :math:`\operatorname{Re} u` and
/// :math:`\operatorname{Im} u` on the halves of the field and recombined with the phase
/// :math:`\pi V_\varphi/V_\pi`.
///
/// Args:
///     e (array_like): the optical field (√W).
///     u (array_like): the complex driving signal (V), one value per sample.
///     vpi (float): the half-wave voltage :math:`V_\pi` of both MZMs (V).
///     vb_i (float): the bias of the in-phase MZM (V).
///     vb_q (float): the bias of the quadrature MZM (V).
///     vphi (float): the voltage of the phase shifter :math:`V_\varphi` (V).
///     er_i_db (float): the extinction ratio of the in-phase MZM (dB).
///     er_q_db (float): the extinction ratio of the quadrature MZM (dB).
///
/// Returns:
///     numpy.ndarray: the modulated field (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (
    e,
    u,
    *,
    vpi = Iqm::default().vpi,
    vb_i = Iqm::default().vb_i,
    vb_q = Iqm::default().vb_q,
    vphi = Iqm::default().vphi,
    er_i_db = Iqm::default().er_i_db,
    er_q_db = Iqm::default().er_q_db,
))]
#[allow(clippy::too_many_arguments, reason = "the fields of the modulator, by keyword")]
fn iqm<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    vpi: f64,
    vb_i: f64,
    vb_q: f64,
    vphi: f64,
    er_i_db: f64,
    er_q_db: f64,
) -> PyResult<CArr<'py>> {
    let m = Iqm { vpi, vb_i, vb_q, vphi, er_i_db, er_q_db };
    Ok(carr(py, m.modulate(&cvec(e)?, &cvec(u)?).py()?))
}

/// A PIN photodiode (OptiCommPy `photodiode`): responsivity, shot noise, thermal noise,
/// and a low-pass FIR of bandwidth :math:`B`.
///
/// The photocurrent and the variances of the noises are
///
/// .. math::
///
///     I = R |E|^2, \qquad \sigma^2_{\mathrm{shot}} = 2q(I + I_d)B, \qquad
///     \sigma^2_{\mathrm{th}} = \frac{4 k_B T B}{R_L}.
///
/// Every attribute is readable and writable; `ideal=True` gives :math:`R|E|^2` without
/// noise nor filter.
///
/// Args:
///     responsivity (float): responsivity (A/W).
///     temperature_k (float): temperature (K).
///     dark_current_a (float): dark current (A).
///     load_ohm (float): load resistance (Ω).
///     bandwidth_hz (float): electrical bandwidth :math:`B` (Hz).
///     n_taps (int): taps of the bandwidth-limiting filter (made odd).
///     filter_kind (str): the filter, `"rect"` or `"gauss"` (see `low_pass_fir`).
///     shot_noise (bool): add shot noise.
///     thermal_noise (bool): add thermal noise.
///     bandwidth_limitation (bool): filter to the bandwidth.
///     ideal (bool): ideal square-law detection (overrides the three flags above).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
#[pyclass(name = "Photodiode", module = "comnum")]
#[derive(Clone)]
pub struct PyPhotodiode {
    inner: Photodiode,
}

wrapper_methods! {
    PyPhotodiode {
        /// float: Responsivity (A/W) (read and write).
        responsivity / set_responsivity: f64,
        /// float: Temperature (K) (read and write).
        temperature_k / set_temperature_k: f64,
        /// float: Dark current (A) (read and write).
        dark_current_a / set_dark_current_a: f64,
        /// float: Load resistance (Ω) (read and write).
        load_ohm / set_load_ohm: f64,
        /// float: Electrical bandwidth :math:`B` (Hz) (read and write).
        bandwidth_hz / set_bandwidth_hz: f64,
        /// int: Taps of the bandwidth-limiting filter (made odd) (read and write).
        n_taps / set_n_taps: usize,
        /// bool: Add shot noise (read and write).
        shot_noise / set_shot_noise: bool,
        /// bool: Add thermal noise (read and write).
        thermal_noise / set_thermal_noise: bool,
        /// bool: Filter to the bandwidth (read and write).
        bandwidth_limitation / set_bandwidth_limitation: bool,
        /// bool: Ideal square-law detection (overrides the three flags above) (read and write).
        ideal / set_ideal: bool,
    }

    #[new]
    #[pyo3(signature = (
        *,
        responsivity = Photodiode::default().responsivity,
        temperature_k = Photodiode::default().temperature_k,
        dark_current_a = Photodiode::default().dark_current_a,
        load_ohm = Photodiode::default().load_ohm,
        bandwidth_hz = Photodiode::default().bandwidth_hz,
        n_taps = Photodiode::default().n_taps,
        filter_kind = Photodiode::default().filter.name(),
        shot_noise = Photodiode::default().shot_noise,
        thermal_noise = Photodiode::default().thermal_noise,
        bandwidth_limitation = Photodiode::default().bandwidth_limitation,
        ideal = Photodiode::default().ideal,
    ))]
    #[allow(clippy::too_many_arguments, reason = "one keyword argument per field of the photodiode")]
    fn new(
        responsivity: f64,
        temperature_k: f64,
        dark_current_a: f64,
        load_ohm: f64,
        bandwidth_hz: f64,
        n_taps: usize,
        filter_kind: &str,
        shot_noise: bool,
        thermal_noise: bool,
        bandwidth_limitation: bool,
        ideal: bool,
    ) -> PyResult<Self> {
        Ok(PyPhotodiode {
            inner: Photodiode {
                responsivity,
                temperature_k,
                dark_current_a,
                load_ohm,
                bandwidth_hz,
                n_taps,
                filter: filter_kind.parse().py()?,
                shot_noise,
                thermal_noise,
                bandwidth_limitation,
                ideal,
            },
        })
    }

    /// str: The filter, `"rect"` or `"gauss"` (read and write; a name is checked when
    /// it is assigned).
    #[getter]
    fn filter_kind(&self) -> &'static str {
        self.inner.filter.name()
    }

    #[setter]
    fn set_filter_kind(&mut self, value: &str) -> PyResult<()> {
        self.inner.filter = value.parse::<LowPass>().py()?;
        Ok(())
    }

    /// Photocurrent of an optical field (one polarisation).
    ///
    /// Args:
    ///     e (array_like): the field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng): the generator of the noises.
    ///
    /// Returns:
    ///     numpy.ndarray: the photocurrent (A, float64).
    ///
    /// Raises:
    ///     ValueError: if the detector is not ideal and :math:`f_s < 2B`.
    fn detect<'py>(
        &self,
        py: Python<'py>,
        e: &Bound<'py, PyAny>,
        fs: f64,
        mut rng: PyRefMut<'py, PyRng>,
    ) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.detect(&cvec(e)?, fs, &mut rng.inner).py()?))
    }

    /// Photocurrent of an optical power.
    ///
    /// Args:
    ///     p (array_like): the optical power of every sample (W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng): the generator of the noises.
    ///
    /// Returns:
    ///     numpy.ndarray: the photocurrent (A, float64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn detect_power<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        fs: f64,
        mut rng: PyRefMut<'py, PyRng>,
    ) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.detect_power(&fvec(p)?, fs, &mut rng.inner).py()?))
    }

    fn __repr__(&self) -> String {
        let d = &self.inner;
        format!(
            "comnum.Photodiode(responsivity={}, temperature_k={}, dark_current_a={:e}, load_ohm={}, bandwidth_hz={:e}, \
             n_taps={}, filter_kind='{}', shot_noise={}, thermal_noise={}, bandwidth_limitation={}, ideal={})",
            d.responsivity,
            d.temperature_k,
            d.dark_current_a,
            d.load_ohm,
            d.bandwidth_hz,
            d.n_taps,
            d.filter.name(),
            py_bool(d.shot_noise),
            py_bool(d.thermal_noise),
            py_bool(d.bandwidth_limitation),
            py_bool(d.ideal)
        )
    }

    fn __getnewargs_ex__<'py>(&self, py: Python<'py>) -> PyResult<((), Bound<'py, PyDict>)> {
        let d = &self.inner;
        let kw = PyDict::new(py);
        kw.set_item("responsivity", d.responsivity)?;
        kw.set_item("temperature_k", d.temperature_k)?;
        kw.set_item("dark_current_a", d.dark_current_a)?;
        kw.set_item("load_ohm", d.load_ohm)?;
        kw.set_item("bandwidth_hz", d.bandwidth_hz)?;
        kw.set_item("n_taps", d.n_taps)?;
        kw.set_item("filter_kind", d.filter.name())?;
        kw.set_item("shot_noise", d.shot_noise)?;
        kw.set_item("thermal_noise", d.thermal_noise)?;
        kw.set_item("bandwidth_limitation", d.bandwidth_limitation)?;
        kw.set_item("ideal", d.ideal)?;
        Ok(((), kw))
    }
}

fn photodiode_or_ideal(pd: Option<&PyPhotodiode>) -> Photodiode {
    pd.map_or_else(Photodiode::ideal, |p| p.inner.clone())
}

/// Balanced detection: the difference of the photocurrents of two photodiodes,
///
/// .. math::
///
///     i = i(E_1) - i(E_2).
///
/// Args:
///     e1 (array_like): the field on the first photodiode (√W).
///     e2 (array_like): the field on the second photodiode (√W).
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator of the noises.
///     photodiode (Photodiode, optional): the photodiodes (ideal by default).
///
/// Returns:
///     numpy.ndarray: the photocurrent (A, float64).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (e1, e2, fs, rng, *, photodiode = None))]
fn balanced_photodiode<'py>(
    py: Python<'py>,
    e1: &Bound<'py, PyAny>,
    e2: &Bound<'py, PyAny>,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
    photodiode: Option<&PyPhotodiode>,
) -> PyResult<FArr<'py>> {
    let pd = photodiode_or_ideal(photodiode);
    Ok(farr(py, devices::balanced_photodiode(&cvec(e1)?, &cvec(e2)?, &pd, fs, &mut rng.inner).py()?))
}

/// The 90° optical hybrid (OptiCommPy `hybrid_2x4_90deg`): the four outputs
///
/// .. math::
///
///     \frac{E_s - E_{lo}}{2}, \quad \frac{j(E_s + E_{lo})}{2}, \quad
///     \frac{jE_s - E_{lo}}{2}, \quad \frac{-E_s + jE_{lo}}{2}.
///
/// Args:
///     es (array_like): the signal field (√W).
///     elo (array_like): the field of the local oscillator (√W).
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: the four
///     output fields.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[allow(clippy::type_complexity, reason = "the four output fields of the hybrid")]
fn hybrid_2x4<'py>(
    py: Python<'py>,
    es: &Bound<'py, PyAny>,
    elo: &Bound<'py, PyAny>,
) -> PyResult<(CArr<'py>, CArr<'py>, CArr<'py>, CArr<'py>)> {
    let [a, b, c, d] = devices::hybrid_2x4(&cvec(es)?, &cvec(elo)?).py()?;
    Ok((carr(py, a), carr(py, b), carr(py, c), carr(py, d)))
}

/// Single-polarisation coherent receiver (OptiCommPy `coherentReceiver`): the hybrid
/// and two balanced photodiodes. With ideal photodiodes its output is
///
/// .. math::
///
///     i = R\, E_s E_{lo}^*.
///
/// Args:
///     es (array_like): the signal field (√W).
///     elo (array_like): the field of the local oscillator (√W).
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator of the noises.
///     photodiode (Photodiode, optional): the photodiodes (ideal by default).
///
/// Returns:
///     numpy.ndarray: the in-phase and quadrature photocurrents as one complex signal
///     (A).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (es, elo, fs, rng, *, photodiode = None))]
fn coherent_receiver<'py>(
    py: Python<'py>,
    es: &Bound<'py, PyAny>,
    elo: &Bound<'py, PyAny>,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
    photodiode: Option<&PyPhotodiode>,
) -> PyResult<CArr<'py>> {
    let pd = photodiode_or_ideal(photodiode);
    Ok(carr(py, devices::coherent_receiver(&cvec(es)?, &cvec(elo)?, &pd, fs, &mut rng.inner).py()?))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhotodiode>()?;
    for f in [
        wrap_pyfunction!(awgn, m)?,
        wrap_pyfunction!(awgn_ebn0, m)?,
        wrap_pyfunction!(phase_noise, m)?,
        wrap_pyfunction!(clock_offset, m)?,
        wrap_pyfunction!(laser, m)?,
        wrap_pyfunction!(phase_modulator, m)?,
        wrap_pyfunction!(mzm, m)?,
        wrap_pyfunction!(iqm, m)?,
        wrap_pyfunction!(balanced_photodiode, m)?,
        wrap_pyfunction!(hybrid_2x4, m)?,
        wrap_pyfunction!(coherent_receiver, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
