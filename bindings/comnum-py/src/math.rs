//! Conversions of units and special functions: decibels, dBm, the Q function and the
//! Bessel function J0.

use crate::{CArr, IntoPy, carr, map_real};
use pyo3::prelude::*;

/// Convert decibels to a linear ratio, `10^{x/10}`.
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

/// Convert a linear ratio to decibels, `10 log10 x`.
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

/// Convert a power in dBm to watts, `10^{(P - 30)/10}`.
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

/// Convert a power in watts to dBm, `10 log10 P + 30`.
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

/// The Gaussian tail function `Q(x) = erfc(x/√2)/2`.
///
/// Args:
///     x (float | array_like): the argument.
///
/// Returns:
///     float | numpy.ndarray: `Q(x)`.
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

/// The Q factor (dB) of a bit error rate, `20 log10(√2 erfc⁻¹(2 BER))`.
///
/// Args:
///     ber (float | array_like): the bit error rate, in `(0, 0.5)`.
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

/// The Bessel function `J0(x)` of the first kind and order zero.
///
/// Args:
///     x (float | array_like): the argument.
///
/// Returns:
///     float | numpy.ndarray: `J0(x)`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
#[pyfunction]
fn bessel_j0(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x, |v| Ok(comnum::math::bessel_j0(v)))
}

/// Register the bindings of this module.
/// The Zadoff–Chu sequence `x_u(n) = exp(-jπ u n (n + 1) / N)` of root `u` and odd
/// length `N` (constant amplitude, zero periodic autocorrelation at every nonzero lag;
/// the PSS of LTE is the one of length 63, its middle element punctured).
///
/// Args:
///     u (int): the root, nonzero modulo `n_zc` and prime to it.
///     n_zc (int): the odd length `N`.
///
/// Returns:
///     numpy.ndarray: the `N` values (complex128).
///
/// Raises:
///     ValueError: for an even length or a root that is not prime to it.
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zadoff_chu, m)?)?;
    m.add_function(wrap_pyfunction!(db2lin, m)?)?;
    m.add_function(wrap_pyfunction!(lin2db, m)?)?;
    m.add_function(wrap_pyfunction!(dbm2w, m)?)?;
    m.add_function(wrap_pyfunction!(w2dbm, m)?)?;
    m.add_function(wrap_pyfunction!(qfunc, m)?)?;
    m.add_function(wrap_pyfunction!(ber_to_q_db, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_j0, m)?)?;
    Ok(())
}
