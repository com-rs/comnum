//! MIMO channel estimation: the estimators of one training block, and the BLUE combination
//! of several trainings.

use super::*;

/// The unbiased estimates of a linear detector and their noise variances.
///
/// On the link :math:`\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{w}`, the estimates are
/// :math:`\hat x_{k} = x_{k} + n_{k}`, with the filters
///
/// .. math::
///
///     \mathbf{W}_{\mathrm{zf}, \mathrm{mmse}}
///     = (\mathbf{H}^{\mathsf{H}} \mathbf{H} + \lambda \mathbf{I})^{-1} \mathbf{H}^{\mathsf{H}},
///     \qquad
///     \mathbf{W}_{\mathrm{mr}} = \mathbf{D}^{-1} \mathbf{H}^{\mathsf{H}},
///
/// :math:`\lambda = 0` for ZF and :math:`\sigma^2/E_s` for MMSE (the MMSE estimate then
/// divided by its bias :math:`1 - E_{kk}/E_s`,
/// :math:`\mathbf{E} = \sigma^2 (\mathbf{H}^{\mathsf{H}} \mathbf{H} + \lambda \mathbf{I})^{-1}`),
/// :math:`\mathbf{D} = \operatorname{diag}(\lVert \mathbf{h}_k \rVert^2)`, and
/// :math:`\sigma^2 = \mathbb{E}|w_i|^2`.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received streams, `(n_rx, n)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna :math:`\mathbb{E}|w_i|^2`.
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

/// A training matrix for the estimation of a flat MIMO channel.
///
/// The :math:`r \times t` channel :math:`\mathbf{H}` is probed by the :math:`N \ge t`
/// columns of the :math:`t \times N` training :math:`\mathbf{P}`, and every estimator is
/// linear ([1]_, eqs. (2), (4), (25)):
///
/// .. math::
///
///     \mathbf{S} = \mathbf{H}\mathbf{P} + \mathbf{V}, \qquad
///     \hat{\mathbf{H}} = \mathbf{S}\mathbf{A},
///
/// with :math:`\mathbf{V}` white of variance :math:`\sigma_n^2` per entry and the
/// :math:`N \times t` matrix :math:`\mathbf{A}`: ``"ls"`` (no prior,
/// :math:`\mathbf{A} = \mathbf{P}^{\mathsf{H}} (\mathbf{P}\mathbf{P}^{\mathsf{H}})^{-1}`),
/// ``"sls"`` (scaled LS, the trace of
/// :math:`\mathbf{R}_H = \mathbb{E}[\mathbf{H}^{\mathsf{H}}\mathbf{H}]` and the noise), ``"mmse"``
/// (the LMMSE of :math:`\mathbf{R}_H`,
/// :math:`\mathbf{A} = (\mathbf{P}^{\mathsf{H}} \mathbf{R}_H \mathbf{P} + \sigma_n^2 r
/// \mathbf{I})^{-1} \mathbf{P}^{\mathsf{H}} \mathbf{R}_H`), ``"rmmse"`` (the LMMSE of
/// :math:`\mathbf{R}_H` replaced by :math:`\operatorname{tr}(\mathbf{R}_H)/t \, \mathbf{I}`),
/// and ``"ls-sls"`` and ``"ls-rmmse"``, whose trace is estimated from the LS estimate.
/// ``mse`` is the closed-form mean square error
/// :math:`\mathbb{E}\lVert \mathbf{H} - \hat{\mathbf{H}} \rVert_F^2`.
///
/// Args:
///     p (array_like): the ``(t, N)`` training, :math:`N \ge t`.
///
/// Raises:
///     ValueError: for an empty matrix, fewer columns than rows or a non-finite entry.
///
/// References:
///     .. [1] M. Biguesh and A. B. Gershman, "Training-based MIMO channel estimation: a
///        study of estimator tradeoffs and optimal training signals,"
///        *IEEE Trans. Signal Process.*, 54(3), 884–893, 2006.
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

    /// The DFT training of orthogonal rows (eq. (10)), of entries of equal magnitude:
    ///
    /// .. math::
    ///
    ///     \mathbf{P}\mathbf{P}^{\mathsf{H}} = \frac{\lVert \mathbf{P} \rVert_F^2}{t} \mathbf{I}.
    ///
    /// Args:
    ///     n_tx (int): the transmit antennas :math:`t`.
    ///     n (int | None): the training symbols :math:`N \ge t` (:math:`t` by default).
    ///     total_power (float): :math:`\lVert \mathbf{P} \rVert_F^2`.
    ///
    /// Returns:
    ///     MimoTraining: the training.
    ///
    /// Raises:
    ///     ValueError: for :math:`N < t`, no antenna or a power that is not positive.
    #[staticmethod]
    #[pyo3(signature = (n_tx, n=None, total_power=1.0))]
    fn orthogonal(n_tx: usize, n: Option<usize>, total_power: f64) -> PyResult<Self> {
        Ok(PyMimoTraining { inner: Training::orthogonal(n_tx, n.unwrap_or(n_tx), total_power).py()? })
    }

    /// The training that minimises the MSE of the LMMSE estimator: the power
    /// water-filled on the eigenvectors of ``r_h`` (eqs. (41)-(44)).
    ///
    /// Args:
    ///     r_h (array_like): the ``(t, t)`` correlation
    ///         :math:`\mathbb{E}[\mathbf{H}^{\mathsf{H}}\mathbf{H}]`.
    ///     noise_var (float): the noise variance per entry.
    ///     n_rx (int): the receive antennas :math:`r`.
    ///     total_power (float): :math:`\lVert \mathbf{P} \rVert_F^2`.
    ///     n (int | None): the training symbols :math:`N \ge t` (:math:`t` by default).
    ///
    /// Returns:
    ///     MimoTraining: the training.
    ///
    /// Raises:
    ///     ValueError: for a correlation that is not square, :math:`N < t`, or a noise or a
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

    /// The ``(t, N)`` training symbols :math:`\mathbf{P}`.
    #[getter]
    fn symbols<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<C64>> {
        self.inner.symbols().clone().into_pyarray(py)
    }

    /// The power :math:`\lVert \mathbf{P} \rVert_F^2`.
    #[getter]
    fn total_power(&self) -> f64 {
        self.inner.total_power()
    }

    /// The estimate of the channel from the received training:
    ///
    /// .. math::
    ///
    ///     \hat{\mathbf{H}} = \mathbf{S}\mathbf{A},
    ///
    /// with the :math:`N \times t` matrix :math:`\mathbf{A}` of the estimator (see the class).
    ///
    /// Args:
    ///     s (array_like): the ``(r, N)`` received training.
    ///     estimator (str): ``"ls"``, ``"sls"``, ``"ls-sls"``, ``"mmse"``, ``"rmmse"`` or
    ///         ``"ls-rmmse"``.
    ///     noise_var (float): the noise variance per entry (unused by LS).
    ///     r_h (array_like | None): the correlation
    ///         :math:`\mathbb{E}[\mathbf{H}^{\mathsf{H}}\mathbf{H}]` of ``"mmse"``.
    ///     trace (float | None): :math:`\operatorname{tr}(\mathbf{R}_H)` of ``"sls"`` and
    ///         ``"rmmse"``.
    ///
    /// Returns:
    ///     numpy.ndarray: the ``(r, t)`` estimate.
    ///
    /// Raises:
    ///     ValueError: for an unknown estimator, a missing prior, inconsistent shapes or
    ///         LS on a training of rank below :math:`t`.
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

    /// The closed-form MSE of ``estimator`` on a channel of correlation ``r_h`` (eqs. (7),
    /// (13), (28), (46) of Biguesh & Gershman).
    ///
    /// With :math:`\mathbf{B} = \mathbf{I} - \mathbf{P}\mathbf{A}`,
    ///
    /// .. math::
    ///
    ///     J = \mathbb{E}\lVert \mathbf{H} - \hat{\mathbf{H}} \rVert_F^2
    ///     = \operatorname{tr}(\mathbf{B}^{\mathsf{H}} \mathbf{R}_H \mathbf{B})
    ///     + \sigma_n^2 r \operatorname{tr}(\mathbf{A}^{\mathsf{H}} \mathbf{A}),
    ///
    /// also when the prior of the estimator is not the true :math:`\mathbf{R}_H`.
    ///
    /// Args:
    ///     estimator (str): ``"ls"``, ``"sls"``, ``"mmse"`` or ``"rmmse"`` (``"ls-sls"`` and
    ///         ``"ls-rmmse"`` have no closed form).
    ///     r_h (array_like): the true ``(t, t)`` correlation
    ///         :math:`\mathbb{E}[\mathbf{H}^{\mathsf{H}}\mathbf{H}]`.
    ///     noise_var (float): the noise variance per entry.
    ///     n_rx (int): the receive antennas :math:`r`.
    ///     prior (array_like | float | None): the prior the estimator assumes, a
    ///         correlation for ``"mmse"``, a trace for ``"sls"`` and ``"rmmse"``; the true
    ///         one by default.
    ///
    /// Returns:
    ///     float: the error, summed over the :math:`rt` entries.
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
/// training periods.
///
/// The weights are ([1]_, Lemma 4, eq. (54))
///
/// .. math::
///
///     \alpha_{k} = \frac{1 / \kappa_{k}}{\sum_{l} 1 / \kappa_{l}}, \qquad
///     \kappa_{k} = \operatorname{tr}\big((\mathbf{P}_{k} \mathbf{P}_{k}^{\mathsf{H}})^{-1}\big),
///
/// :math:`P_k / P_{\mathrm{tot}}` for orthogonal trainings.
///
/// Args:
///     trainings (list[MimoTraining]): the training of every period.
///
/// Returns:
///     numpy.ndarray: the weights, summing to 1.
///
/// Raises:
///     ValueError: for no period, a training of rank below :math:`t` or periods of
///         different :math:`t`.
///
/// References:
///     .. [1] M. Biguesh and A. B. Gershman, "Training-based MIMO channel estimation: a
///        study of estimator tradeoffs and optimal training signals,"
///        *IEEE Trans. Signal Process.*, 54(3), 884–893, 2006.
#[pyfunction]
pub(super) fn blue_weights<'py>(py: Python<'py>, trainings: Vec<PyRef<'py, PyMimoTraining>>) -> PyResult<FArr<'py>> {
    Ok(farr(py, est::blue_weights(&self::trainings(&trainings)).py()?))
}

/// The combination of channel estimates.
///
/// The combined estimate is ([1]_, eq. (52))
///
/// .. math::
///
///     \hat{\mathbf{H}} = \sum_k w_k \hat{\mathbf{H}}_k.
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
///
/// References:
///     .. [1] M. Biguesh and A. B. Gershman, "Training-based MIMO channel estimation: a
///        study of estimator tradeoffs and optimal training signals,"
///        *IEEE Trans. Signal Process.*, 54(3), 884–893, 2006.
#[pyfunction]
pub(super) fn blue_combine<'py>(
    py: Python<'py>,
    estimates: Vec<Bound<'py, PyAny>>,
    weights: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let hs = estimates.iter().map(|h| matrix_in(h, "an estimate")).collect::<PyResult<Vec<_>>>()?;
    Ok(est::blue_combine(&hs, &crate::fvec(weights)?).py()?.into_pyarray(py))
}

/// The MSE of the BLUE combination of the LS estimates of several periods.
///
/// The error is ([1]_, eqs. (59), (61))
///
/// .. math::
///
///     J = \frac{\sigma^2 r}
///     {\sum_k 1 / \operatorname{tr}\big((\mathbf{P}_k \mathbf{P}_k^{\mathsf{H}})^{-1}\big)},
///
/// :math:`\sigma^2` the noise variance per entry and :math:`r` the receive antennas.
///
/// Args:
///     trainings (list[MimoTraining]): the training of every period.
///     noise_var (float): the noise variance per entry.
///     n_rx (int): the receive antennas :math:`r`.
///
/// Returns:
///     float: the error, summed over the :math:`rt` entries.
///
/// Raises:
///     ValueError: for no period, a training of rank below :math:`t`, a negative noise or no
///         receive antenna.
///
/// References:
///     .. [1] M. Biguesh and A. B. Gershman, "Training-based MIMO channel estimation: a
///        study of estimator tradeoffs and optimal training signals,"
///        *IEEE Trans. Signal Process.*, 54(3), 884–893, 2006.
#[pyfunction]
pub(super) fn blue_mse(trainings: Vec<PyRef<'_, PyMimoTraining>>, noise_var: f64, n_rx: usize) -> PyResult<f64> {
    est::blue_mse(&self::trainings(&trainings), noise_var, n_rx).py()
}
