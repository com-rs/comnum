//! The finite-length limits and the BPSK helpers of the code tests: the BI-AWGN capacity,
//! the normal approximation, the Shannon limit.

use super::*;

/// Capacity of the binary-input AWGN channel for a code rate.
///
/// With BPSK on real AWGN of variance :math:`\sigma^2 = 1/(2 R E_b/N_0)` per dimension,
/// the LLR of a transmitted 0 is :math:`L \sim \mathcal{N}(2/\sigma^2, 4/\sigma^2)` and
///
/// .. math::
///
///     C = 1 - \mathbb{E}\left[\log_2(1 + e^{-L})\right],
///
/// by Gauss–Hermite quadrature.
///
/// Args:
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///     rate (float): the code rate :math:`R`.
///
/// Returns:
///     float | numpy.ndarray: the capacity (bits per channel use) at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_biawgn_capacity(comnum.theory_shannon_limit_ebn0_db(0.5), 0.5), 6)
///     0.5
#[pyfunction]
fn theory_biawgn_capacity(ebn0_db: &Bound<'_, PyAny>, rate: f64) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::biawgn_capacity(bounds::bpsk_noise_variance(e, rate))))
}

/// The normal approximation of the smallest block error probability of an :math:`(n, k)`
/// binary code over BPSK and AWGN.
///
/// With the capacity :math:`C` (:func:`theory_biawgn_capacity`) and the dispersion
/// :math:`V = \operatorname{Var}\left[1 - \log_2(1 + e^{-L})\right]` of the binary-input
/// AWGN channel at the rate :math:`R = k/n` [1]_:
///
/// .. math::
///
///     \varepsilon \approx Q\left(\frac{n C - k + \tfrac{1}{2}\log_2 n}{\sqrt{n V}}\right).
///
/// Args:
///     n (int): block length.
///     k (int): information bits.
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///
/// Returns:
///     float | numpy.ndarray: the block error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// References:
///     .. [1] Y. Polyanskiy, H. V. Poor and S. Verdú, "Channel coding rate in the finite
///        blocklength regime," *IEEE Trans. Inf. Theory*, 56(5), 2307–2359, 2010.
///
/// Examples:
///     >>> import comnum
///     >>> p = comnum.theory_normal_approximation(256, 128, [1.0, 3.0])
///     >>> bool(p[0] > p[1])
///     True
#[pyfunction]
fn theory_normal_approximation(n: usize, k: usize, ebn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::normal_approximation(n, k, e)))
}

/// The smallest :math:`E_b/N_0` at which BPSK supports a code rate.
///
/// The limit (:math:`n \to \infty`) [1]_ is the solution of
///
/// .. math::
///
///     C = R,
///
/// with :math:`C` the capacity of the binary-input AWGN channel at
/// :math:`\sigma^2 = 1/(2 R E_b/N_0)` (:func:`theory_biawgn_capacity`).
///
/// Args:
///     rate (float | array_like): the code rate :math:`R`.
///
/// Returns:
///     float | numpy.ndarray: the limit (dB).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// References:
///     .. [1] C. E. Shannon, "A mathematical theory of communication," *Bell Syst. Tech.
///        J.*, 27, 379–423 and 623–656, 1948.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_shannon_limit_ebn0_db(0.5), 3)
///     0.187
#[pyfunction]
fn theory_shannon_limit_ebn0_db(rate: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(rate, |r| Ok(bounds::shannon_limit_ebn0_db(r)))
}

/// The noise variance per dimension of BPSK at :math:`E_b/N_0` for a code of rate `rate`
/// (unit-energy symbols):
///
/// .. math::
///
///     \sigma^2 = \frac{1}{2 R E_b/N_0}.
///
/// Args:
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///     rate (float): the code rate :math:`R`.
///
/// Returns:
///     float | numpy.ndarray: :math:`\sigma^2` at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.bpsk_noise_variance(0.0, 0.5)
///     1.0
#[pyfunction]
fn bpsk_noise_variance(ebn0_db: &Bound<'_, PyAny>, rate: f64) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::bpsk_noise_variance(e, rate)))
}

/// The LLRs of BPSK (:math:`0 \mapsto +1`, :math:`1 \mapsto -1`) received in real Gaussian
/// noise of variance `noise_var` per dimension.
///
/// For the sample :math:`y` and the variance :math:`\sigma^2`:
///
/// .. math::
///
///     L = \frac{2y}{\sigma^2},
///
/// positive for a 0, as every decoder of the package expects.
///
/// Args:
///     y (array_like): the received real samples.
///     noise_var (float): the noise variance per dimension :math:`\sigma^2`.
///
/// Returns:
///     numpy.ndarray: one LLR per sample (float64).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.bpsk_llr([1.0, -0.5], 0.5).tolist()
///     [4.0, -2.0]
#[pyfunction]
fn bpsk_llr<'py>(py: Python<'py>, y: &Bound<'py, PyAny>, noise_var: f64) -> PyResult<FArr<'py>> {
    Ok(farr(py, comnum::mapping::demapper::bpsk_llr(&fvec(y)?, noise_var)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theory_biawgn_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(bpsk_noise_variance, m)?)?;
    m.add_function(wrap_pyfunction!(bpsk_llr, m)?)?;
    m.add_function(wrap_pyfunction!(theory_normal_approximation, m)?)?;
    m.add_function(wrap_pyfunction!(theory_shannon_limit_ebn0_db, m)?)?;
    Ok(())
}
