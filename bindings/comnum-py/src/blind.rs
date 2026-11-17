//! Blind estimation: cumulants, modulation classification, the M2M4 SNR, the symbol
//! rate by cyclostationarity.

use crate::comm::PyConstellation;
use crate::{IntoPy, cvec, map_real};
use comnum::dsp::cyclo;
use comnum::metrics::blind::{self, Classifier, Cumulants};
use num_complex::Complex64 as C64;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn cumulant_dict<'py>(py: Python<'py>, c: &Cumulants) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("c20", c.c20)?;
    d.set_item("c21", c.c21)?;
    d.set_item("c40", c.c40)?;
    d.set_item("c41", c.c41)?;
    d.set_item("c42", c.c42)?;
    d.set_item("c40_normalized", c.c40_normalized())?;
    d.set_item("c42_normalized", c.c42_normalized())?;
    Ok(d)
}

/// The sample cumulants of order two and four of a zero-mean signal, and the normalised
/// :math:`C_{40}/C_{21}^2`, :math:`C_{42}/C_{21}^2`.
///
/// With the moments :math:`M_{pq} = \mathbb{E}[r^{p-q} (r^*)^q]`, replaced by sample means,
/// the cumulants are [1]_
///
/// .. math::
///
///     \begin{aligned}
///     C_{20} &= M_{20}, \quad C_{21} = M_{21}, \quad C_{40} = M_{40} - 3 M_{20}^2, \\
///     C_{41} &= M_{41} - 3 M_{20} M_{21}, \quad C_{42} = M_{42} - |M_{20}|^2 - 2 M_{21}^2,
///     \end{aligned}
///
/// and :math:`\tilde{C}_{4k} = C_{4k}/C_{21}^2`.
///
/// Args:
///     x (array_like): the samples (complex).
///
/// Returns:
///     dict: ``c20``, ``c21``, ``c40``, ``c41``, ``c42``, ``c40_normalized``,
///     ``c42_normalized``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] A. Swami and B. M. Sadler, "Hierarchical digital modulation classification
///        using cumulants," *IEEE Trans. Commun.*, 48(3), 416–429, 2000.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> c = comnum.Constellation("qam", 4).normalized()
///     >>> round(comnum.cumulants(np.tile(c.points, 100))["c42_normalized"], 6)
///     -1.0
#[pyfunction]
fn cumulants<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    cumulant_dict(py, &Cumulants::estimate(&cvec(x)?).py()?)
}

/// The cumulants of an alphabet (its theoretical features), with the probabilities of
/// its points.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///
/// Returns:
///     dict: as :func:`cumulants`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_cumulants(comnum.Constellation("qam", 16))["c42_normalized"], 2)
///     -0.68
#[pyfunction]
#[pyo3(signature = (constellation, *, px=None))]
fn theory_cumulants<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let px = px.map(crate::fvec).transpose()?;
    cumulant_dict(py, &Cumulants::of_constellation(&constellation.inner, px.as_deref()).py()?)
}

/// Blind modulation classification at one sample per symbol, timing and carrier
/// recovered.
///
/// Methods: `"ml"` (the likelihood, the scale and the noise estimated by M2M4 [1]_ under
/// every candidate), `"ml_known"` (the likelihood with the noise variance ``noise_var``
/// known and the symbols of unit energy, [2]_), `"cumulants"` (the nearest
/// fourth-order features :math:`|C_{40}|/C_{21}^2`, :math:`C_{42}/C_{21}^2`, [3]_; with
/// ``noise_var``, normalised by :math:`C_{21} - \mathrm{noise\_var}`).
///
/// The score of a candidate alphabet :math:`\{a\}` of :math:`M` points under `"ml_known"` is
/// the mean log-likelihood of the :math:`L` samples in noise of variance :math:`N`:
///
/// .. math::
///
///     \frac{1}{L} \sum_{n} \ln \frac{1}{M} \sum_{a} e^{-|r_n - a|^2/N},
///
/// up to a constant common to the candidates.
///
/// Args:
///     x (array_like): the samples (complex).
///     candidates (list[Constellation]): the candidate alphabets (scaled to unit energy).
///     method (str): the classifier.
///     noise_var (float, optional): the noise variance, for `"ml_known"` and
///         `"cumulants"`.
///
/// Returns:
///     dict: ``index`` (the decided candidate) and ``scores`` (one per candidate, the
///     largest wins).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] D. R. Pauluzzi and N. C. Beaulieu, "A comparison of SNR estimation techniques
///        for the AWGN channel," *IEEE Trans. Commun.*, 48(10), 1681–1691, 2000.
///     .. [2] W. Wei and J. M. Mendel, "Maximum-likelihood classification for digital
///        amplitude-phase modulations," *IEEE Trans. Commun.*, 48(2), 189–193, 2000.
///     .. [3] A. Swami and B. M. Sadler, "Hierarchical digital modulation classification
///        using cumulants," *IEEE Trans. Commun.*, 48(3), 416–429, 2000.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> rng = comnum.Rng(1)
///     >>> cands = [comnum.Constellation(k, m).normalized() for k, m in [("psk", 2), ("qam", 4), ("qam", 16)]]
///     >>> x = cands[2].points[rng.integers(16, 2000)] + rng.complex_noise(2000, 0.02)
///     >>> comnum.classify_modulation(x, cands)["index"]
///     2
#[pyfunction]
#[pyo3(signature = (x, candidates, method="ml", *, noise_var=None))]
fn classify_modulation<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    candidates: Vec<PyRef<'py, PyConstellation>>,
    method: &str,
    noise_var: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = cvec(x)?;
    let cands: Vec<_> = candidates.iter().map(|c| c.inner.normalized()).collect();
    let classifier = Classifier::from_name(method, noise_var).py()?;
    let d = py.detach(|| classifier.classify(&x, &cands)).py()?;
    let out = PyDict::new(py);
    out.set_item("index", d.index)?;
    out.set_item("scores", d.scores)?;
    Ok(out)
}

/// The M2M4 estimate of the SNR, generalised to alphabets of several amplitudes by their
/// kurtosis; it degrades at high SNR for QAM.
///
/// With :math:`M_2 = \mathbb{E}|r|^2` and :math:`M_4 = \mathbb{E}|r|^4` (sample means) and
/// the kurtosis :math:`k_a = \mathbb{E}|a|^4/(\mathbb{E}|a|^2)^2` of the alphabet (1 for
/// PSK), the signal and noise powers are
///
/// .. math::
///
///     \hat{S} = \sqrt{\frac{2 M_2^2 - M_4}{2 - k_a}}, \qquad \hat{N} = M_2 - \hat{S},
///
/// and the SNR is :math:`\hat{S}/\hat{N}` ([1]_, eq. (37), with the kurtosis 2 of
/// complex noise; eq. (39) for PSK).
///
/// Args:
///     x (array_like): the samples (complex), at one sample per symbol.
///     constellation (Constellation): the alphabet.
///
/// Returns:
///     float: the SNR (dB); ``inf`` when the moments give no noise.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] D. R. Pauluzzi and N. C. Beaulieu, "A comparison of SNR estimation techniques
///        for the AWGN channel," *IEEE Trans. Commun.*, 48(10), 1681–1691, 2000.
///
/// Examples:
///     >>> import comnum
///     >>> rng = comnum.Rng(2)
///     >>> c = comnum.Constellation("psk", 8).normalized()
///     >>> x = c.points[rng.integers(8, 100_000)] + rng.complex_noise(100_000, 0.1)
///     >>> round(comnum.snr_m2m4_db(x, c), 1)
///     10.0
#[pyfunction]
fn snr_m2m4_db(x: &Bound<'_, PyAny>, constellation: &PyConstellation) -> PyResult<f64> {
    let snr = blind::snr_m2m4(&cvec(x)?, &constellation.inner).py()?;
    Ok(comnum::math::lin2db(snr))
}

/// The Cramér–Rao bound on the normalised mean squared error of an unbiased SNR
/// estimator from known PSK symbols.
///
/// The bound is ([1]_, eq. (65) in complex AWGN, eq. (67), twice it, in real AWGN)
///
/// .. math::
///
///     \mathrm{NMSE}(\hat{\rho}) \ge \frac{2}{\rho N_{\mathrm{sym}}} + \frac{1}{N_{\mathrm{ss}} N_{\mathrm{sym}}},
///
/// with :math:`\rho` the SNR, :math:`N_{\mathrm{sym}}` = `n_symbols` and
/// :math:`N_{\mathrm{ss}}` = `sps`.
///
/// Args:
///     snr_db (float | array_like): the true SNR (dB).
///     n_symbols (int): the symbols of a block.
///     sps (int): independent samples per symbol.
///     real (bool): real AWGN instead of complex.
///
/// Returns:
///     float | numpy.ndarray: the bound on :math:`\mathbb{E}[(\hat{\rho} - \rho)^2] / \rho^2`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] D. R. Pauluzzi and N. C. Beaulieu, "A comparison of SNR estimation techniques
///        for the AWGN channel," *IEEE Trans. Commun.*, 48(10), 1681–1691, 2000.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.snr_crb_nmse(30.0, 64), 4)  # tends to 1/64
///     0.0157
#[pyfunction]
#[pyo3(signature = (snr_db, n_symbols, *, sps=1, real=false))]
fn snr_crb_nmse(snr_db: &Bound<'_, PyAny>, n_symbols: usize, sps: usize, real: bool) -> PyResult<Py<PyAny>> {
    map_real(snr_db, |s| blind::snr_crb_nmse(comnum::math::db2lin(s), n_symbols, sps, real).py())
}

/// The symbol rate of a linearly modulated signal from the strongest line of the
/// spectrum of its instantaneous power (cyclostationarity); the pulse must
/// have an excess bandwidth.
///
/// At :math:`\tau = 0` the cyclic autocorrelation [1]_ (`cyclic_autocorrelation`),
///
/// .. math::
///
///     R_x^\alpha(0) = \frac{1}{N} \sum_n |x[n]|^2\, e^{-j 2\pi \alpha n/f_s},
///
/// has a line at :math:`\alpha = 1/T`; the estimate is the strongest line of the spectrum
/// of :math:`|x|^2` (its mean removed) in the search band, refined by a parabola through
/// the log-magnitudes of the three bins around the peak.
///
/// Args:
///     x (array_like): the samples (complex).
///     fs (float): the sampling rate (Hz).
///     f_min_hz (float): the lowest symbol rate searched (Hz).
///     f_max_hz (float): the highest symbol rate searched (Hz, at most :math:`f_s/2`).
///
/// Returns:
///     float: the symbol rate (Hz).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] W. A. Gardner, "Exploitation of spectral redundancy in cyclostationary
///        signals," *IEEE Signal Process. Mag.*, 8(2), 14–36, 1991.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> n = np.arange(4096)
///     >>> x = np.sqrt(1 + 0.3 * np.cos(np.pi * n / 4)).astype(complex)
///     >>> round(comnum.estimate_symbol_rate(x, 1.0, 0.05, 0.2), 6)
///     0.125
#[pyfunction]
fn estimate_symbol_rate(x: &Bound<'_, PyAny>, fs: f64, f_min_hz: f64, f_max_hz: f64) -> PyResult<f64> {
    cyclo::estimate_symbol_rate(&cvec(x)?, fs, f_min_hz, f_max_hz).py()
}

/// The cyclic autocorrelation
///
/// .. math::
///
///     R^\alpha(\tau) = \operatorname{mean}_n\, x[n+\tau]\, x^*[n]\, e^{-j2\pi \alpha n/f_s}.
///
/// The definition is that of [1]_.
///
/// Args:
///     x (array_like): the samples (complex).
///     alpha (float): the cycle frequency (Hz).
///     lag (int): the lag :math:`\tau` (samples).
///     fs (float): the sampling rate (Hz).
///
/// Returns:
///     complex: the cyclic autocorrelation.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// References:
///     .. [1] W. A. Gardner, "Exploitation of spectral redundancy in cyclostationary
///        signals," *IEEE Signal Process. Mag.*, 8(2), 14–36, 1991.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> x = np.ones(100, dtype=complex)
///     >>> abs(comnum.cyclic_autocorrelation(x, 0.0, lag=0, fs=1.0))
///     1.0
#[pyfunction]
#[pyo3(signature = (x, alpha, *, lag=0, fs=1.0))]
fn cyclic_autocorrelation(x: &Bound<'_, PyAny>, alpha: f64, lag: usize, fs: f64) -> PyResult<C64> {
    Ok(cyclo::cyclic_autocorrelation(&cvec(x)?, alpha, lag, fs))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cumulants, m)?)?;
    m.add_function(wrap_pyfunction!(theory_cumulants, m)?)?;
    m.add_function(wrap_pyfunction!(classify_modulation, m)?)?;
    m.add_function(wrap_pyfunction!(snr_m2m4_db, m)?)?;
    m.add_function(wrap_pyfunction!(snr_crb_nmse, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_symbol_rate, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_autocorrelation, m)?)?;
    Ok(())
}
