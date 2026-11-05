//! MIMO channel estimation: the estimators of one training block, and the BLUE combination
//! of several trainings.

use super::*;

/// The unbiased estimates of a linear detector and their noise variances.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received streams, `(n_rx, n)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): `"mr"`, `"zf"` or `"mmse"`.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the estimates and their variances, both
///     `(n_tx, n)` (or `(n_tx,)`).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (h, y, constellation, noise_var, method="mmse"))]
pub(super) fn mimo_estimate(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let lin = match method.parse::<DetectorKind>().py()? {
        DetectorKind::Mr => Linear::mr(constellation.inner.clone()),
        DetectorKind::Zf => Linear::zf(constellation.inner.clone()),
        DetectorKind::Mmse => Linear::mmse(constellation.inner.clone()),
        _ => return Err(PyValueError::new_err(format!("method must be mr, zf or mmse, got {method:?}"))),
    };
    let rows = py
        .detach(|| {
            per_vector(&hs, &ys, |h, y| {
                lin.estimate(h, y, noise_var).map(|e| e.symbols.into_iter().zip(e.noise_var).collect())
            })
        })
        .py()?;
    let (z, v): (Vec<C64>, Vec<f64>) = rows.data.into_iter().unzip();
    let width = rows.width;
    Ok((rows_to_array(py, Rows { data: z, width }, single)?, rows_to_array(py, Rows { data: v, width }, single)?))
}

/// A complex matrix from a 2-D array.
pub(super) fn matrix_in(x: &Bound<'_, PyAny>, what: &str) -> PyResult<Array2<C64>> {
    carray(x)?.into_dimensionality().map_err(|_| PyValueError::new_err(format!("{what} must be a 2-D array")))
}

/// The estimator of a name, with its prior: the correlation `r_h` of `"mmse"`, the trace
/// of `"sls"` and `"rmmse"`.
fn channel_estimator(name: &str, r_h: Option<Array2<C64>>, trace: Option<f64>) -> PyResult<ChannelEstimator> {
    let known =
        |what: &str| trace.map(TracePrior::Known).ok_or_else(|| PyValueError::new_err(format!("{what} needs trace=")));
    Ok(match name {
        "ls" => ChannelEstimator::Ls,
        "sls" => ChannelEstimator::ScaledLs(known("sls")?),
        "ls-sls" => ChannelEstimator::ScaledLs(TracePrior::FromLs),
        "rmmse" => ChannelEstimator::RelaxedMmse(known("rmmse")?),
        "ls-rmmse" => ChannelEstimator::RelaxedMmse(TracePrior::FromLs),
        "mmse" => ChannelEstimator::Mmse(r_h.ok_or_else(|| PyValueError::new_err("mmse needs r_h="))?),
        _ => {
            return Err(PyValueError::new_err(format!(
                "estimator must be ls, sls, ls-sls, mmse, rmmse or ls-rmmse, got {name:?}"
            )));
        }
    })
}

/// A training matrix ``P`` (``t x N``) for the estimation of a flat MIMO channel from
/// ``S = H P + V`` (Biguesh & Gershman 2006).
///
/// The estimators are linear, ``H_hat = S A``: ``"ls"`` (no prior), ``"sls"`` (scaled LS,
/// the trace of ``R_H = E[H^H H]`` and the noise), ``"mmse"`` (the LMMSE of ``R_H``),
/// ``"rmmse"`` (the LMMSE of ``R_H`` replaced by ``tr(R_H)/t I``), and ``"ls-sls"`` and
/// ``"ls-rmmse"``, whose trace is estimated from the LS estimate. ``mse`` is the
/// closed-form mean square error ``E||H - H_hat||_F^2``.
///
/// Args:
///     p (array_like): the ``(t, N)`` training, ``N >= t``.
///
/// Raises:
///     ValueError: for an empty matrix, fewer columns than rows or a non-finite entry.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> p = comnum.MimoTraining.orthogonal(4, total_power=10.0)
///     >>> r_h = 2 * np.eye(4)
///     >>> round(p.mse("ls", r_h, 0.5, 2), 6)   # sigma^2 t^2 r / P, eq. (12)
///     1.6
///     >>> round(p.mse("sls", r_h, 0.5, 2) / 8, 4)   # t / (P / sigma^2 + t), eq. (20)
///     0.1667
#[pyclass(name = "MimoTraining", module = "comnum")]
#[derive(Clone)]
pub struct PyMimoTraining {
    inner: Training,
}

#[pymethods]
impl PyMimoTraining {
    #[new]
    fn new(p: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyMimoTraining { inner: Training::new(matrix_in(p, "the training")?).py()? })
    }

    /// The DFT training of orthogonal rows, ``P P^H = (total_power / t) I`` (eq. (10)).
    ///
    /// Args:
    ///     n_tx (int): the transmit antennas ``t``.
    ///     n (int | None): the training symbols ``N >= t`` (``t`` by default).
    ///     total_power (float): ``||P||_F^2``.
    ///
    /// Returns:
    ///     MimoTraining: the training.
    ///
    /// Raises:
    ///     ValueError: for ``n < n_tx``, no antenna or a power that is not positive.
    #[staticmethod]
    #[pyo3(signature = (n_tx, n=None, total_power=1.0))]
    fn orthogonal(n_tx: usize, n: Option<usize>, total_power: f64) -> PyResult<Self> {
        Ok(PyMimoTraining { inner: Training::orthogonal(n_tx, n.unwrap_or(n_tx), total_power).py()? })
    }

    /// The training that minimises the MSE of the LMMSE estimator: the power
    /// water-filled on the eigenvectors of ``r_h`` (eqs. (41)-(44)).
    ///
    /// Args:
    ///     r_h (array_like): the ``(t, t)`` correlation ``E[H^H H]``.
    ///     noise_var (float): the noise variance per entry.
    ///     n_rx (int): the receive antennas ``r``.
    ///     total_power (float): ``||P||_F^2``.
    ///     n (int | None): the training symbols ``N >= t`` (``t`` by default).
    ///
    /// Returns:
    ///     MimoTraining: the training.
    ///
    /// Raises:
    ///     ValueError: for a correlation that is not square, ``n < t``, or a noise or a
    ///         power that is not positive.
    #[staticmethod]
    #[pyo3(signature = (r_h, noise_var, n_rx, total_power, n=None))]
    fn mmse_optimal(
        r_h: &Bound<'_, PyAny>,
        noise_var: f64,
        n_rx: usize,
        total_power: f64,
        n: Option<usize>,
    ) -> PyResult<Self> {
        let r = matrix_in(r_h, "r_h")?;
        let n = n.unwrap_or(r.nrows());
        Ok(PyMimoTraining { inner: Training::mmse_optimal(&r.view(), noise_var, n_rx, total_power, n).py()? })
    }

    /// The ``(t, N)`` training symbols ``P``.
    #[getter]
    fn symbols<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<C64>> {
        self.inner.symbols().clone().into_pyarray(py)
    }

    /// The power ``||P||_F^2``.
    #[getter]
    fn total_power(&self) -> f64 {
        self.inner.total_power()
    }

    /// The estimate ``H_hat = S A`` of the channel from the received training.
    ///
    /// Args:
    ///     s (array_like): the ``(r, N)`` received training.
    ///     estimator (str): ``"ls"``, ``"sls"``, ``"ls-sls"``, ``"mmse"``, ``"rmmse"`` or
    ///         ``"ls-rmmse"``.
    ///     noise_var (float): the noise variance per entry (unused by LS).
    ///     r_h (array_like | None): the correlation ``E[H^H H]`` of ``"mmse"``.
    ///     trace (float | None): ``tr(R_H)`` of ``"sls"`` and ``"rmmse"``.
    ///
    /// Returns:
    ///     numpy.ndarray: the ``(r, t)`` estimate.
    ///
    /// Raises:
    ///     ValueError: for an unknown estimator, a missing prior, inconsistent shapes or
    ///         LS on a training of rank below ``t``.
    #[pyo3(signature = (s, estimator="ls", noise_var=0.0, *, r_h=None, trace=None))]
    fn estimate<'py>(
        &self,
        py: Python<'py>,
        s: &Bound<'py, PyAny>,
        estimator: &str,
        noise_var: f64,
        r_h: Option<&Bound<'py, PyAny>>,
        trace: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let r = r_h.map(|r| matrix_in(r, "r_h")).transpose()?;
        let e = channel_estimator(estimator, r, trace)?;
        Ok(self.inner.estimate(&matrix_in(s, "s")?.view(), &e, noise_var).py()?.into_pyarray(py))
    }

    /// The closed-form MSE ``E||H - H_hat||_F^2`` of ``estimator`` on a channel of
    /// correlation ``r_h`` (eqs. (7), (13), (28), (46) of Biguesh & Gershman).
    ///
    /// Args:
    ///     estimator (str): ``"ls"``, ``"sls"``, ``"mmse"`` or ``"rmmse"`` (``"ls-sls"`` and
    ///         ``"ls-rmmse"`` have no closed form).
    ///     r_h (array_like): the true ``(t, t)`` correlation ``E[H^H H]``.
    ///     noise_var (float): the noise variance per entry.
    ///     n_rx (int): the receive antennas ``r``.
    ///     prior (array_like | float | None): the prior the estimator assumes, a
    ///         correlation for ``"mmse"``, a trace for ``"sls"`` and ``"rmmse"``; the true
    ///         one by default.
    ///
    /// Returns:
    ///     float: the error, summed over the ``r t`` entries.
    ///
    /// Raises:
    ///     ValueError: for an estimator without a closed form, or inconsistent shapes.
    #[pyo3(signature = (estimator, r_h, noise_var, n_rx, prior=None))]
    fn mse(
        &self,
        estimator: &str,
        r_h: &Bound<'_, PyAny>,
        noise_var: f64,
        n_rx: usize,
        prior: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let r = matrix_in(r_h, "r_h")?;
        let (assumed, trace) = match (estimator, prior) {
            ("mmse", Some(p)) => (Some(matrix_in(p, "prior")?), None),
            (_, Some(p)) => (None, Some(p.extract::<f64>()?)),
            (_, None) => (Some(r.clone()), Some((0..r.nrows().min(r.ncols())).map(|i| r[(i, i)].re).sum())),
        };
        let e = channel_estimator(estimator, assumed, trace)?;
        self.inner.mse(&e, &r.view(), noise_var, n_rx).py()
    }

    fn __repr__(&self) -> String {
        format!("MimoTraining({} x {}, power {})", self.inner.n_tx(), self.inner.n_symbols(), self.inner.total_power())
    }
}

fn trainings(list: &[PyRef<'_, PyMimoTraining>]) -> Vec<Training> {
    list.iter().map(|t| t.inner.clone()).collect()
}

/// The weights of the best linear unbiased combination of the LS estimates of several
/// training periods (Biguesh & Gershman, Lemma 4); ``P_k / P_tot`` for orthogonal
/// trainings.
///
/// Args:
///     trainings (list[MimoTraining]): the training of every period.
///
/// Returns:
///     numpy.ndarray: the weights, summing to 1.
///
/// Raises:
///     ValueError: for no period, a training of rank below ``t`` or periods of
///         different ``t``.
#[pyfunction]
pub(super) fn blue_weights<'py>(py: Python<'py>, trainings: Vec<PyRef<'py, PyMimoTraining>>) -> PyResult<FArr<'py>> {
    Ok(farr(py, est::blue_weights(&self::trainings(&trainings)).py()?))
}

/// The combination ``sum_k w_k H_k`` of channel estimates (eq. (52)).
///
/// Args:
///     estimates (list[array_like]): the ``(r, t)`` estimates.
///     weights (array_like): one weight per estimate (``blue_weights``).
///
/// Returns:
///     numpy.ndarray: the combined ``(r, t)`` estimate.
///
/// Raises:
///     ValueError: for no estimate, estimates of different shapes, or as many weights
///         as estimates.
#[pyfunction]
pub(super) fn blue_combine<'py>(
    py: Python<'py>,
    estimates: Vec<Bound<'py, PyAny>>,
    weights: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let hs = estimates.iter().map(|h| matrix_in(h, "an estimate")).collect::<PyResult<Vec<_>>>()?;
    Ok(est::blue_combine(&hs, &crate::fvec(weights)?).py()?.into_pyarray(py))
}

/// The MSE ``sigma^2 r / sum_k 1/tr((P_k P_k^H)^-1)`` of the BLUE combination of the LS
/// estimates of several periods (eqs. (59), (61)).
///
/// Args:
///     trainings (list[MimoTraining]): the training of every period.
///     noise_var (float): the noise variance per entry.
///     n_rx (int): the receive antennas ``r``.
///
/// Returns:
///     float: the error, summed over the ``r t`` entries.
///
/// Raises:
///     ValueError: for no period, a training of rank below ``t``, a negative noise or no
///         receive antenna.
#[pyfunction]
pub(super) fn blue_mse(trainings: Vec<PyRef<'_, PyMimoTraining>>, noise_var: f64, n_rx: usize) -> PyResult<f64> {
    est::blue_mse(&self::trainings(&trainings), noise_var, n_rx).py()
}
