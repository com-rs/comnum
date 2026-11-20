//! Conversions of units and special functions: decibels, dBm, the Q function and the
//! Bessel function J0.

use crate::{CArr, IntoPy, carr, map_real};
use pyo3::prelude::*;

/// Convert decibels to a linear ratio, :math:`10^{x/10}`.
///
/// Args:
///     x_db (float | array_like): the ratio (dB).
///
/// Returns:
///     float | numpy.ndarray: the linear ratio.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.db2lin(20.0), comnum.db2lin([0.0, 10.0]).tolist()
///     (100.0, [1.0, 10.0])
#[pyfunction]
fn db2lin(x_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x_db, |v| Ok(comnum::math::db2lin(v)))
}

/// Convert decibels to an amplitude ratio, :math:`10^{x/20}`.
///
/// Args:
///     x_db (float | array_like): the ratio (dB).
///
/// Returns:
///     float | numpy.ndarray: the amplitude ratio.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.db2amp(20.0), round(comnum.db2amp(-6.0), 4)
///     (10.0, 0.5012)
#[pyfunction]
fn db2amp(x_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x_db, |v| Ok(comnum::math::db2amp(v)))
}

/// Convert an amplitude ratio to decibels, :math:`20 \log_{10} x`.
///
/// Args:
///     x (float | array_like): the amplitude ratio.
///
/// Returns:
///     float | numpy.ndarray: the ratio (dB).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.amp2db(10.0), comnum.amp2db([1.0, 100.0]).tolist()
///     (20.0, [0.0, 40.0])
#[pyfunction]
fn amp2db(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x, |v| Ok(comnum::math::amp2db(v)))
}

/// The mean power of a signal, :math:`\operatorname{mean}(|x|^2)`.
///
/// Args:
///     x (array_like): the samples, real or complex (1-D).
///
/// Returns:
///     float: the mean of the squared magnitudes (0 for an empty signal).
///
/// Raises:
///     ValueError: for an input that is not a 1-D array of numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.power([1.0, 1j, -1.0, 2.0])
///     1.75
#[pyfunction]
fn power(x: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(comnum::math::power(&crate::cvec(x)?))
}

/// Convert a linear ratio to decibels, :math:`10 \log_{10} x`.
///
/// Args:
///     x (float | array_like): the linear ratio.
///
/// Returns:
///     float | numpy.ndarray: the ratio (dB).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.lin2db(100.0)
///     20.0
#[pyfunction]
fn lin2db(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x, |v| Ok(comnum::math::lin2db(v)))
}

/// Convert a power in dBm to watts, :math:`10^{(P - 30)/10}`.
///
/// Args:
///     p_dbm (float | array_like): the power (dBm).
///
/// Returns:
///     float | numpy.ndarray: the power (W).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.dbm2w(30.0)
///     1.0
#[pyfunction]
fn dbm2w(p_dbm: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p_dbm, |v| Ok(comnum::math::dbm2w(v)))
}

/// Convert a power in watts to dBm, :math:`10 \log_{10} P + 30`.
///
/// Args:
///     p_w (float | array_like): the power (W).
///
/// Returns:
///     float | numpy.ndarray: the power (dBm).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.w2dbm(1e-3)
///     0.0
#[pyfunction]
fn w2dbm(p_w: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p_w, |v| Ok(comnum::math::w2dbm(v)))
}

/// The Gaussian tail function, the probability that a standard normal variable exceeds
/// :math:`x`:
///
/// .. math::
///
///     Q(x) = \frac{1}{2} \operatorname{erfc}\!\left(\frac{x}{\sqrt{2}}\right).
///
/// Args:
///     x (float | array_like): the argument.
///
/// Returns:
///     float | numpy.ndarray: :math:`Q(x)`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.qfunc(0.0), 6), round(comnum.qfunc(3.0), 6)
///     (0.5, 0.00135)
#[pyfunction]
fn qfunc(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x, |v| Ok(comnum::math::qfunc(v)))
}

/// The inverse of the Gaussian tail function, :math:`Q^{-1}(p) = \sqrt{2}\,
/// \operatorname{erfc}^{-1}(2p)` on :math:`(0, 1)`: the Q factor of a bit error rate
/// (:math:`Q^{-1}(10^{-9}) \approx 6`).
///
/// Args:
///     p (float | array_like): the probability, in :math:`(0, 1)`.
///
/// Returns:
///     float | numpy.ndarray: :math:`Q^{-1}(p)` (:math:`\pm\infty` at the ends).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.qfuncinv(1e-9), 3)
///     5.998
#[pyfunction]
fn qfuncinv(p: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p, |v| Ok(comnum::math::qfuncinv(v)))
}

/// The Q factor (dB) of a bit error rate.
///
/// The factor is
///
/// .. math::
///
///     Q_{\mathrm{dB}} = 20 \log_{10}\left(\sqrt{2}\, \operatorname{erfc}^{-1}(2\, \mathrm{BER})\right).
///
/// Args:
///     ber (float | array_like): the bit error rate, in :math:`(0, 0.5)`.
///
/// Returns:
///     float | numpy.ndarray: the Q factor (dB).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.ber_to_q_db(1e-3), 3)
///     9.8
#[pyfunction]
fn ber_to_q_db(ber: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ber, |v| Ok(comnum::metrics::ber_to_q_db(v)))
}

/// The Bessel function :math:`J_0(x)` of the first kind and order zero.
///
/// Args:
///     x (float | array_like): the argument.
///
/// Returns:
///     float | numpy.ndarray: :math:`J_0(x)`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
#[pyfunction]
fn bessel_j0(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x, |v| Ok(comnum::math::bessel_j0(v)))
}

/// The Zadoff–Chu sequence of root :math:`u` and odd length :math:`N` (constant
/// amplitude, zero periodic autocorrelation at every nonzero lag; the PSS of LTE is the
/// one of length 63, its middle element punctured).
///
/// Its values are ([1]_; [2]_, Sec. 6.11.1.1)
///
/// .. math::
///
///     x_u(n) = \exp\left(-j\pi u n (n + 1) / N\right), \qquad n = 0, \dots, N - 1.
///
/// Args:
///     u (int): the root, nonzero modulo `n_zc` and prime to it.
///     n_zc (int): the odd length :math:`N`.
///
/// Returns:
///     numpy.ndarray: the :math:`N` values (complex128).
///
/// Raises:
///     ValueError: for an even length or a root that is not prime to it.
///
/// References:
///     .. [1] D. Chu, "Polyphase codes with good periodic correlation properties," *IEEE
///        Trans. Inf. Theory*, 18(4), 531–532, 1972.
///     .. [2] 3GPP TS 36.211 V18.0.2, *E-UTRA; Physical channels and modulation*.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> x = comnum.zadoff_chu(25, 63)
///     >>> bool(np.allclose(np.abs(x), 1.0)), round(float(abs(np.vdot(x, np.roll(x, 3)))), 9)
///     (True, 0.0)
#[pyfunction]
fn zadoff_chu(py: Python<'_>, u: u32, n_zc: u32) -> PyResult<CArr<'_>> {
    Ok(carr(py, comnum::math::sequences::zadoff_chu(u, n_zc).py()?))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zadoff_chu, m)?)?;
    m.add_function(wrap_pyfunction!(db2lin, m)?)?;
    m.add_function(wrap_pyfunction!(lin2db, m)?)?;
    m.add_function(wrap_pyfunction!(db2amp, m)?)?;
    m.add_function(wrap_pyfunction!(amp2db, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(dbm2w, m)?)?;
    m.add_function(wrap_pyfunction!(w2dbm, m)?)?;
    m.add_function(wrap_pyfunction!(qfunc, m)?)?;
    m.add_function(wrap_pyfunction!(qfuncinv, m)?)?;
    m.add_function(wrap_pyfunction!(ber_to_q_db, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_j0, m)?)?;
    Ok(())
}
