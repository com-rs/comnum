//! Message passing detection of large MIMO (LAMA, AMP, OAMP) and its state evolution.

use super::*;

fn prior(c: &PyConstellation, px: Option<&Bound<'_, PyAny>>) -> PyResult<Prior> {
    let px = px.map(crate::fvec).transpose()?;
    Prior::with_probs(&c.inner, px.as_deref()).py()
}

fn domain(real: bool) -> Domain {
    if real { Domain::Real } else { Domain::Complex }
}

fn steps<'py>(py: Python<'py>, se: &[mp::StateEvolutionStep]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("noise_var", farr(py, se.iter().map(|s| s.noise_var).collect()))?;
    d.set_item("mse", farr(py, se.iter().map(|s| s.mse).collect()))?;
    Ok(d)
}

/// The iterations of LAMA or OAMP on one received vector: the MMSE estimate of every
/// iteration, and the decoupled channel of the last one (Jeon et al. 2018; Ma & Ping
/// 2017).
///
/// Args:
///     h (array_like): the channel `(n_rx, n_tx)`.
///     y (array_like): the received vector `(n_rx,)`.
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): `"amp"` (LAMA) or `"oamp"`.
///     iterations (int): iterations.
///     linear (str): the linear estimator of OAMP, `"lmmse"`, `"pinv"` or `"mf"`.
///     variance (str): the variance of LAMA, `"postulated"` (Algorithm 2) or
///         `"residual"` (eq. 38, better in small systems).
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model (real channel, symbols and noise).
///
/// Returns:
///     dict: ``estimates`` (`(iterations, n_tx)`, the MMSE estimate of every iteration),
///     ``z`` (`(n_tx,)`, the input of the last denoiser: the symbols in Gaussian noise),
///     ``noise_var`` (its variance) and ``noise_vars`` (the variance of every iteration).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> c = comnum.Constellation("qam", 4).normalized()
///     >>> rng = comnum.Rng(1)
///     >>> h = comnum.mimo_channel(64, 32, rng)
///     >>> x = c.points[rng.integers(4, 32)]
///     >>> out = comnum.message_passing(h, h @ x, c, 1e-4, "amp", iterations=15)
///     >>> bool(np.allclose(out["estimates"][-1], x, atol=1e-3))
///     True
#[pyfunction]
#[pyo3(signature = (
    h, y, constellation, noise_var, method="amp", *, iterations=DetectorKind::DEFAULT_ITERATIONS,
    linear="lmmse", variance="postulated", px=None, real=false
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
pub(super) fn message_passing<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    iterations: usize,
    linear: &str,
    variance: &str,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let Channels::One(h) = channels(h)? else {
        return Err(PyValueError::new_err("message_passing takes one channel matrix (n_rx, n_tx)"));
    };
    let y = crate::cvec(y)?;
    let config = MessagePassing { prior: prior(constellation, px)?, domain: domain(real), iterations };
    let trace = match method.parse::<DetectorKind>().py()? {
        DetectorKind::Amp(_) => {
            let v: AmpVariance = variance.parse().py()?;
            py.detach(|| mp::lama(&h.view(), &y, noise_var, &config, v)).py()?
        }
        DetectorKind::Oamp(_) => {
            let l: OampLinear = linear.parse().py()?;
            py.detach(|| mp::oamp(&h.view(), &y, noise_var, &config, l)).py()?
        }
        _ => return Err(PyValueError::new_err(format!("unknown method {method:?}: amp or oamp"))),
    };
    let n = h.ncols();
    let flat: Vec<C64> = trace.estimates.iter().flatten().copied().collect();
    let est =
        Array2::from_shape_vec((trace.estimates.len(), n), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("estimates", est.into_pyarray(py))?;
    d.set_item("z", trace.z.clone().into_pyarray(py))?;
    d.set_item("noise_var", trace.noise_var)?;
    d.set_item("noise_vars", farr(py, trace.noise_vars))?;
    Ok(d)
}

/// The state evolution of LAMA (Jeon et al. 2018, Corollary 2): the variance of the
/// decoupled channel and the predicted MSE of every iteration, in the large-system limit
/// with columns of unit norm.
///
/// Args:
///     constellation (Constellation): the alphabet of every stream.
///     beta (float): the system ratio `n_tx / n_rx`.
///     noise_var (float): noise variance per receive antenna, relative to columns of unit
///         norm (`β Es / SNR`).
///     iterations (int): iterations.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model.
///
/// Returns:
///     dict: ``noise_var`` and ``mse``, one value per iteration.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comnum
///     >>> c = comnum.Constellation("qam", 4).normalized()
///     >>> se = comnum.amp_state_evolution(c, 0.5, 0.05, 20)
///     >>> bool(se["mse"][-1] < se["mse"][0])
///     True
#[pyfunction]
#[pyo3(signature = (constellation, beta, noise_var, iterations, *, px=None, real=false))]
pub(super) fn amp_state_evolution<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    beta: f64,
    noise_var: f64,
    iterations: usize,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let config = MessagePassing { prior: prior(constellation, px)?, domain: domain(real), iterations };
    let se = py.detach(|| mp::amp_state_evolution(&config, beta, noise_var));
    steps(py, &se)
}

/// The recovery thresholds of LAMA (Jeon et al. 2018, Table II): below ``beta_min`` LAMA
/// is optimal at every SNR, below ``beta_max`` it recovers the noiseless signal;
/// ``n0_min`` and ``n0_max`` are the critical noise variances at those ratios.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model.
///
/// Returns:
///     dict: ``beta_min``, ``beta_max``, ``n0_min`` and ``n0_max``.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comnum
///     >>> t = comnum.amp_thresholds(comnum.Constellation("qam", 16).normalized())
///     >>> round(t["beta_min"], 3), round(t["beta_max"], 3)
///     (0.983, 1.363)
#[pyfunction]
#[pyo3(signature = (constellation, *, px=None, real=false))]
pub(super) fn amp_thresholds<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = prior(constellation, px)?;
    let t = py.detach(|| mp::amp_thresholds(&p, domain(real)));
    let d = PyDict::new(py);
    for (k, v) in [("beta_min", t.beta_min), ("beta_max", t.beta_max), ("n0_min", t.n0_min), ("n0_max", t.n0_max)] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// The empirical state evolution of OAMP with the optimal denoiser (Ma & Ping 2017,
/// eqs. 32 and 39) for the singular values of a channel.
///
/// Args:
///     constellation (Constellation): the alphabet of every stream.
///     singular_values (array_like): the singular values of the channel.
///     n_tx (int): the number of columns of the channel.
///     noise_var (float): noise variance per receive antenna.
///     iterations (int): iterations.
///     linear (str): the linear estimator, `"lmmse"`, `"pinv"` or `"mf"`.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model.
///
/// Returns:
///     dict: ``noise_var`` (`τ²` of every iteration) and ``mse``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> c = comnum.Constellation("qam", 4).normalized()
///     >>> h = comnum.mimo_channel(128, 64, comnum.Rng(1)) / np.sqrt(128)
///     >>> lam = np.linalg.svd(h, compute_uv=False)
///     >>> se = comnum.oamp_state_evolution(c, lam, 64, 0.3, 10)
///     >>> bool(se["mse"][-1] < se["mse"][0])
///     True
#[pyfunction]
#[pyo3(signature = (
    constellation, singular_values, n_tx, noise_var, iterations, *, linear="lmmse", px=None, real=false
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
pub(super) fn oamp_state_evolution<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    singular_values: &Bound<'py, PyAny>,
    n_tx: usize,
    noise_var: f64,
    iterations: usize,
    linear: &str,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let config = MessagePassing { prior: prior(constellation, px)?, domain: domain(real), iterations };
    let sv = crate::fvec(singular_values)?;
    let l: OampLinear = linear.parse().py()?;
    let se = py.detach(|| mp::oamp_state_evolution(&config, &sv, n_tx, noise_var, l));
    steps(py, &se)
}
