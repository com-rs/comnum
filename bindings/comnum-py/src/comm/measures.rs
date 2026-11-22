//! The measured metrics: error rates, SNR, EVM and the information rates.

use super::*;

/// Bit error rate of two bit sequences of the same length.
///
/// Args:
///     rx_bits (array_like): the decided bits.
///     tx_bits (array_like): the transmitted bits.
///
/// Returns:
///     float: the fraction of different bits.
///
/// Raises:
///     ValueError: for sequences of different lengths.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.ber([0, 1, 1, 0], [0, 1, 0, 0])
///     0.25
#[pyfunction]
fn ber(rx_bits: &Bound<'_, PyAny>, tx_bits: &Bound<'_, PyAny>) -> PyResult<f64> {
    metrics::ber(&bvec(rx_bits)?, &bvec(tx_bits)?).py()
}

/// The worst-case opening of the :math:`M - 1` eyes between adjacent levels of
/// noiseless samples. After [1]_, eq. (7): the lowest sample of the upper level minus the
/// highest of the lower one, negative for a closed eye (``nan`` for a level without a
/// sample). The power penalty of an intersymbol interference in direct detection is
/// :math:`-10 \log_{10}(\mathrm{eye}/Y)` with :math:`Y` the opening without it.
///
/// Args:
///     samples (array_like): the samples, one per symbol.
///     levels (array_like): the index of the level of every sample, below ``m``.
///     m (int): the number of levels.
///
/// Returns:
///     ndarray: the :math:`M - 1` openings.
///
/// Raises:
///     ValueError: for lengths that differ, a level index at or above ``m``, or fewer
///         than two levels.
///
/// References:
///     .. [1] J. H. Winters, R. D. Gitlin, "Electrical signal processing techniques in
///        long-haul, fiber-optic systems," *Proc. IEEE ICC*, 397–403, 1990.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.eye_openings([0.1, 0.3, 0.8, 1.0], [0, 0, 1, 1], 2)
///     array([0.5])
#[pyfunction]
fn eye_openings<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    levels: &Bound<'py, PyAny>,
    m: usize,
) -> PyResult<FArr<'py>> {
    Ok(farr(py, metrics::eye_openings(&fvec(samples)?, &uvec(levels)?, m).py()?))
}

/// The Q factor of every eye of a sampled intensity-modulated signal, from the means
/// and standard deviations of the samples of each level (Agrawal, *Fiber-Optic
/// Communication Systems*, 3rd ed., 2002, eq. (4.5.11)):
///
/// .. math::
///
///     Q_i = \frac{\mu_{i+1} - \mu_i}{\sigma_{i+1} + \sigma_i}, \qquad i = 0, \dots, M - 2,
///
/// the BER of the eye being :math:`\frac{1}{2}\operatorname{erfc}(Q_i/\sqrt{2})`
/// (:func:`qfunc`) under Gaussian noise; the measured counterpart of
/// :meth:`DirectDetectionReceiver.q_factor`.
///
/// Args:
///     samples (array_like): one sample per symbol (real).
///     levels (array_like): the index of the level transmitted at every sample, below `m`.
///     m (int): the number of levels (2 for on–off keying, 4 for PAM4).
///
/// Returns:
///     numpy.ndarray: the `m - 1` Q factors (NaN where a level has no sample).
///
/// Raises:
///     ValueError: for sequences of different lengths, a level index at or above `m`, or
///         fewer than two levels.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.level_q_factors([0.1, -0.1, 0.9, 1.1], [0, 0, 1, 1], 2)
///     array([5.])
#[pyfunction]
fn level_q_factors<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    levels: &Bound<'py, PyAny>,
    m: usize,
) -> PyResult<FArr<'py>> {
    Ok(farr(py, metrics::level_q_factors(&fvec(samples)?, &uvec(levels)?, m).py()?))
}

/// SNR (dB) of a received sequence against the transmitted one.
///
/// The ratio of the powers is
///
/// .. math::
///
///     \mathrm{SNR} = 10 \log_{10}(P_{tx}/P_{rx-tx}),
///
/// with :math:`P_{tx}` the power of the transmitted sequence and :math:`P_{rx-tx}` that
/// of the difference.
///
/// Args:
///     rx (array_like): the received samples.
///     tx (array_like): the transmitted samples, of the same length.
///
/// Returns:
///     float: the SNR (dB).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn snr_db(rx: &Bound<'_, PyAny>, tx: &Bound<'_, PyAny>) -> PyResult<f64> {
    metrics::snr_db(&cvec(rx)?, &cvec(tx)?).py()
}

/// The standard deviation (rad) of the phase error of received symbols against the
/// transmitted ones, once the constant gain and phase of the least-squares fit are
/// removed.
///
/// With the received symbols :math:`z_k`, the transmitted ones :math:`x_k` and the
/// least-squares gain :math:`g`:
///
/// .. math::
///
///     \sigma_\theta = \sqrt{\frac{1}{N}\sum_k \arg(g z_k x_k^*)^2}, \qquad
///     g = \frac{\sum_k x_k z_k^*}{\sum_k |z_k|^2},
///
/// the metric of [1]_ for a phase-modulated signal; over AWGN at high SNR it tends to
/// :math:`1/\sqrt{2\,\mathrm{SNR}}`. Transmitted symbols at zero are skipped.
///
/// Args:
///     rx (array_like): the received symbols.
///     tx (array_like): the transmitted symbols, of the same length.
///
/// Returns:
///     float: the standard deviation of the phase error (rad).
///
/// Raises:
///     ValueError: for sequences of different lengths.
///
/// References:
///     .. [1] E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments
///        using digital backpropagation," *J. Lightw. Technol.*, 26(20), 3416–3425, 2008.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> x = np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7]))
///     >>> z = x * np.exp(1j * (0.3 + 0.01 * np.array([1, -1, 1, -1])))
///     >>> round(comnum.phase_error_std(z, x), 9)
///     0.01
#[pyfunction]
fn phase_error_std(rx: &Bound<'_, PyAny>, tx: &Bound<'_, PyAny>) -> PyResult<f64> {
    metrics::phase_error_std(&cvec(rx)?, &cvec(tx)?).py()
}

/// BER, SER and SNR of a received symbol sequence (OptiCommPy `fastBERcalc`).
///
/// The received sequence is rotated by :math:`\operatorname{mean}(tx/rx)` for QAM and PSK
/// and normalised to unit power, then decided on the normalised constellation.
///
/// Args:
///     rx (array_like): the received symbols.
///     tx (array_like): the transmitted symbols.
///     constellation (Constellation): the alphabet.
///
/// Returns:
///     tuple[float, float, float]: BER, SER and SNR (dB).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn fast_ber_calc(
    rx: &Bound<'_, PyAny>,
    tx: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
) -> PyResult<(f64, f64, f64)> {
    let r = metrics::fast_ber_calc(&cvec(rx)?, &cvec(tx)?, &constellation.inner).py()?;
    Ok((r.ber, r.ser, r.snr_db))
}

/// Error vector magnitude as a power ratio (OptiCommPy `calcEVM`).
///
/// The ratio of the power of the error vector to that of the reference :math:`\hat x`:
///
/// .. math::
///
///     \mathrm{evm} = \frac{\mathbb{E}|y - \hat x|^2}{\mathbb{E}|\hat x|^2};
///
/// the EVM in percent is :math:`100 \sqrt{\mathrm{evm}}`.
///
/// With `tx`, :math:`\hat x` is the transmitted sequence (after the same preprocessing as
/// `fast_ber_calc`); without, the nearest points of the normalised constellation.
///
/// Args:
///     symbols (array_like): the received symbols.
///     constellation (Constellation): the alphabet.
///     tx (array_like, optional): the transmitted symbols.
///
/// Returns:
///     float: the EVM as a power ratio.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (symbols, constellation, *, tx = None))]
fn evm(symbols: &Bound<'_, PyAny>, constellation: &PyConstellation, tx: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    let tx = tx.map(cvec).transpose()?;
    metrics::evm(&cvec(symbols)?, &constellation.inner, tx.as_deref()).py()
}

/// Generalised mutual information of bit-wise decoding (OptiCommPy `monteCarloGMI`).
///
/// Args:
///     rx (array_like): the received symbols.
///     tx (array_like): the transmitted symbols.
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (probabilistic
///         shaping); uniform by default. The constellation is normalised to unit energy
///         under them and the LLRs include them [1]_.
///
/// Returns:
///     tuple[float, float]: GMI (bits/symbol) and normalised GMI (per bit of
///     information, :math:`\mathrm{GMI} / H(X)`).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] G. Böcherer, "Achievable rates for shaped bit-metric decoding,"
///        arXiv:1410.8075, 2014.
#[pyfunction]
#[pyo3(signature = (rx, tx, constellation, *, px = None))]
fn monte_carlo_gmi(
    py: Python<'_>,
    rx: &Bound<'_, PyAny>,
    tx: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    px: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, f64)> {
    let px = px.map(crate::fvec).transpose()?;
    let (rx, tx, c) = (cvec(rx)?, cvec(tx)?, &constellation.inner);
    py.detach(|| metrics::monte_carlo_gmi(&rx, &tx, c, px.as_deref())).map(|g| (g.gmi, g.ngmi)).py()
}

/// Symbol-wise mutual information for a Gaussian channel of the measured noise
/// variance (OptiCommPy `monteCarloMI`; the transmitted symbols are snapped to the
/// constellation, so that the estimate never exceeds :math:`H(X)`).
///
/// Args:
///     rx (array_like): the received symbols.
///     tx (array_like): the transmitted symbols.
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (uniform by
///         default).
///
/// Returns:
///     float: the mutual information (bits/symbol).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (rx, tx, constellation, *, px = None))]
fn monte_carlo_mi(
    rx: &Bound<'_, PyAny>,
    tx: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    px: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let px = px.map(crate::fvec).transpose()?;
    metrics::monte_carlo_mi(&cvec(rx)?, &cvec(tx)?, &constellation.inner, px.as_deref()).py()
}

/// Achievable rates of a receiver with its own posterior
///
/// .. math::
///
///     q(x|z) \propto e^{-|z - x|^2/\hat{\sigma}^2}.
///
/// The rate of a decoder that uses the receiver's beliefs (mismatched decoding), not a
/// genie's:
///
/// .. math::
///
///     \mathrm{air} = H(X) - \mathbb{E}[-\log_2 q(x|z)]
///
/// on the unit-energy constellation, without any gain correction, and its bit-wise
/// version with the LLRs of :math:`q`. Scaling the exponent by :math:`s` and maximising
/// gives `air_opt` [1]_ (per block);
///
/// .. math::
///
///     \mathrm{calibration} = \hat{\sigma}^2 / \operatorname{mean}|z - x|^2
///
/// is 1 for a calibrated receiver, below 1 for an over-confident one.
///
/// Args:
///     z (array_like): the receiver's output, one sample per symbol.
///     indices (array_like): the transmitted indices.
///     constellation (Constellation): the alphabet (normalised inside).
///     noise_var (float): the noise variance :math:`\hat{\sigma}^2` the receiver believes in.
///     px (array_like, optional): prior probabilities of the points; uniform by default.
///
/// Returns:
///     dict: ``air``, ``air_bmd``, ``air_opt``, ``air_bmd_opt`` (bits/symbol), ``scale``,
///     ``scale_bmd`` (the best scalings) and ``calibration``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] G. Kaplan and S. Shamai, "Information rates and error exponents of compound
///        channels with application to antipodal signaling in a fading environment,"
///        *AEÜ*, 47(4), 228–239, 1993.
#[pyfunction]
#[pyo3(signature = (z, indices, constellation, noise_var, *, px = None))]
fn posterior_rates<'py>(
    py: Python<'py>,
    z: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let (z, idx) = (cvec(z)?, uvec(indices)?);
    let px = px.map(crate::fvec).transpose()?;
    let c = &constellation.inner;
    let r = py.detach(|| metrics::posterior_rates(&z, &idx, c, noise_var, px.as_deref())).py()?;
    let d = pyo3::types::PyDict::new(py);
    for (k, v) in [
        ("air", r.air),
        ("air_bmd", r.air_bmd),
        ("air_opt", r.air_opt),
        ("air_bmd_opt", r.air_bmd_opt),
        ("scale", r.scale),
        ("scale_bmd", r.scale_bmd),
        ("calibration", r.calibration),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Mutual information and GMI with the least-squares channel gain: consistent at every
/// SNR, unlike the power normalisation of `monte_carlo_mi` and `monte_carlo_gmi`.
///
/// The model is
///
/// .. math::
///
///     y = a x + w,
///
/// with :math:`x` the points at `indices` of the constellation normalised to unit energy
/// under the priors ``px`` (uniform by default),
///
/// .. math::
///
///     a = \frac{\sum y x^*}{\sum |x|^2}, \qquad \sigma^2 = \operatorname{mean}|y - a x|^2;
///
/// the rates use the priors [1]_.
///
/// Args:
///     rx (array_like): the received symbols.
///     indices (array_like): the transmitted indices.
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points.
///
/// Returns:
///     dict: ``mi``, ``gmi`` (bits/symbol), ``ngmi`` (per bit of information),
///     ``snr_db`` (the effective SNR, dB) and ``entropy`` (:math:`H(X)`, bits).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] G. Böcherer, "Achievable rates for shaped bit-metric decoding,"
///        arXiv:1410.8075, 2014.
#[pyfunction]
#[pyo3(signature = (rx, indices, constellation, *, px = None))]
fn information_rates<'py>(
    py: Python<'py>,
    rx: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let px = px.map(crate::fvec).transpose()?;
    let (rx, indices, c) = (cvec(rx)?, uvec(indices)?, &constellation.inner);
    let r = py.detach(|| metrics::information_rates(&rx, &indices, c, px.as_deref())).py()?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("mi", r.mi)?;
    d.set_item("gmi", r.gmi)?;
    d.set_item("ngmi", r.ngmi)?;
    d.set_item("snr_db", r.snr_db)?;
    d.set_item("entropy", r.entropy)?;
    Ok(d)
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ber, m)?)?;
    m.add_function(wrap_pyfunction!(level_q_factors, m)?)?;
    m.add_function(wrap_pyfunction!(eye_openings, m)?)?;
    m.add_function(wrap_pyfunction!(snr_db, m)?)?;
    m.add_function(wrap_pyfunction!(phase_error_std, m)?)?;
    m.add_function(wrap_pyfunction!(fast_ber_calc, m)?)?;
    m.add_function(wrap_pyfunction!(evm, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_gmi, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_mi, m)?)?;
    m.add_function(wrap_pyfunction!(information_rates, m)?)?;
    m.add_function(wrap_pyfunction!(posterior_rates, m)?)?;
    Ok(())
}
