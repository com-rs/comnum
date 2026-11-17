//! Power amplifiers: the memoryless models of Rapp and Saleh, the memory polynomial,
//! their Bussgang decomposition and the digital predistorter.

use crate::{CArr, IntoPy, carr, cvec};
use comnum::dsp::compensation::dpd::MemoryPolynomialDpd;
use comnum::dsp::nonlinear;
use comnum::hardware::amplifier::{MemoryPolynomial, Model, Rapp, Saleh};
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Bussgang's decomposition of a memoryless amplifier driven by a circular complex
/// Gaussian signal: the complex gain and the power of the distortion.
///
/// The output is written [1]_
///
/// .. math::
///
///     y = K x + d,
///
/// with :math:`K` the complex gain and :math:`d` a distortion uncorrelated with :math:`x`,
/// of power :math:`\sigma_d^2`.
///
/// The parameters of each model default to those of :func:`pa_rapp` and
/// :func:`pa_saleh` (Rapp [2]_ with :math:`p = 2`, the fit of [3]_).
///
/// Args:
///     model (str): `"rapp"` (gain 1, `saturation`, `smoothness`) or
///         `"saleh"` (`alpha_a`, `beta_a`, `alpha_p`, `beta_p`).
///     power (float): the input power.
///     saturation (float): Rapp's saturation amplitude.
///     smoothness (float): Rapp's smoothness :math:`p`.
///     alpha_a (float): Saleh's AM/AM gain.
///     beta_a (float): Saleh's AM/AM compression.
///     alpha_p (float): Saleh's AM/PM gain (rad).
///     beta_p (float): Saleh's AM/PM compression.
///
/// Returns:
///     tuple[complex, float]: :math:`K` and :math:`\sigma_d^2`.
///
/// Raises:
///     ValueError: for an unknown model.
///
/// References:
///     .. [1] J. J. Bussgang, "Crosscorrelation functions of amplitude-distorted Gaussian
///        signals," MIT RLE Tech. Rep. 216, 1952.
///     .. [2] C. Rapp, "Effects of HPA-nonlinearity on a 4-DPSK/OFDM-signal for a digital
///        sound broadcasting system," *Proc. 2nd European Conf. on Satellite
///        Communications*, Liège, 179–184, 1991.
///     .. [3] A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear
///        models of TWT amplifiers," *IEEE Trans. Commun.*, 29(11), 1715–1720, 1981.
///     .. [4] H. Bouhadda, H. Shaiek, D. Roviras, R. Zayani, Y. Medjahdi and R.
///        Bouallegue, "Theoretical analysis of BER performance of nonlinearly amplified
///        FBMC/OQAM and OFDM signals," *EURASIP J. Adv. Signal Process.*, 2014:60, 2014.
///
/// Examples:
///     The TWTA of [4]_ (Table 3) at 4 dB of input back-off,
///     :math:`\phi_0 = \pi/6`:
///
///     >>> import comnum, math
///     >>> k, d = comnum.amplifier_bussgang("saleh", 10 ** -0.4, alpha_a=1.0, beta_a=1.0,
///     ...                                 alpha_p=math.pi / 6, beta_p=1.0)
///     >>> round(k.real, 4), round(k.imag, 4)
///     (0.5917, 0.1106)
#[pyfunction]
#[pyo3(signature = (
    model,
    power,
    *,
    saturation = Rapp::default().saturation,
    smoothness = Rapp::default().smoothness,
    alpha_a = Saleh::default().alpha_a,
    beta_a = Saleh::default().beta_a,
    alpha_p = Saleh::default().alpha_p,
    beta_p = Saleh::default().beta_p,
))]
#[allow(clippy::too_many_arguments, reason = "the parameters of both amplifier models, by keyword")]
fn amplifier_bussgang(
    model: &str,
    power: f64,
    saturation: f64,
    smoothness: f64,
    alpha_a: f64,
    beta_a: f64,
    alpha_p: f64,
    beta_p: f64,
) -> PyResult<(C64, f64)> {
    let b = match model.parse::<Model>().py()? {
        Model::Rapp => nonlinear::bussgang_of(&Rapp { saturation, smoothness, ..Rapp::default() }, power),
        Model::Saleh => nonlinear::bussgang_of(&Saleh { alpha_a, beta_a, alpha_p, beta_p }, power),
        other => return Err(PyValueError::new_err(format!("no Bussgang decomposition for {other:?}"))),
    };
    Ok((b.gain, b.distortion))
}

/// Rapp's solid-state amplifier, no AM/PM.
///
/// The AM/AM conversion [1]_ is
///
/// .. math::
///
///     A(r) = \frac{g r}{\bigl(1 + (g r / A_{\mathrm{sat}})^{2p}\bigr)^{1/(2p)}}.
///
/// Args:
///     x (array_like): the input envelope.
///     gain (float): the small-signal gain :math:`g`.
///     saturation (float): the saturation amplitude :math:`A_{\mathrm{sat}}`.
///     smoothness (float): the smoothness :math:`p`.
///     obo_db (float | None): when given, the input is first scaled so that the mean
///         output power of the block sits `obo_db` dB below the saturation power
///         (the output back-off of [2]_, eq. (3)); by bisection.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape,
///         a back-off that is not positive, or a silent input with a back-off.
///
/// References:
///     .. [1] C. Rapp, "Effects of HPA-nonlinearity on a 4-DPSK/OFDM-signal for a digital
///        sound broadcasting system," *Proc. 2nd European Conf. on Satellite
///        Communications*, Liège, 179–184, 1991.
///     .. [2] A. Saul, "Peak reduction for OFDM by shaping the clipping noise," *Proc. IEEE
///        VTC2004-Fall*, vol. 1, 443–447, 2004.
#[pyfunction]
#[pyo3(signature = (x, *, gain=Rapp::default().gain, saturation=Rapp::default().saturation, smoothness=Rapp::default().smoothness, obo_db=None))]
fn pa_rapp<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    gain: f64,
    saturation: f64,
    smoothness: f64,
    obo_db: Option<f64>,
) -> PyResult<CArr<'py>> {
    let pa = Rapp { gain, saturation, smoothness };
    let mut v = cvec(x)?;
    if let Some(obo) = obo_db {
        let scale = pa.scale_for_output_backoff(&v, obo).py()?;
        v.iter_mut().for_each(|s| *s *= scale);
    }
    Ok(carr(py, pa.amplify(&v).py()?))
}

/// The static predistorter of Rapp's amplifier: the inverse AM/AM, up to `max_output`
/// times the saturation (the cascade is then an ideal envelope limiter).
///
/// Args:
///     x (array_like): the input envelope.
///     gain (float): the small-signal gain of the amplifier.
///     saturation (float): its saturation amplitude.
///     smoothness (float): its smoothness.
///     max_output (float): the largest output, relative to the saturation.
///
/// Returns:
///     numpy.ndarray: the predistorted input (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
#[pyfunction]
#[pyo3(signature = (
    x,
    *,
    gain = Rapp::default().gain,
    saturation = Rapp::default().saturation,
    smoothness = Rapp::default().smoothness,
    max_output = 0.99,
))]
fn rapp_predistort<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    gain: f64,
    saturation: f64,
    smoothness: f64,
    max_output: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Rapp { gain, saturation, smoothness }.predistort(&cvec(x)?, max_output).py()?))
}

/// Saleh's travelling-wave tube amplifier (AM/AM and AM/PM, default coefficients of his
/// fit).
///
/// In the model [1]_, an input :math:`x = r e^{j\theta}` gives
/// :math:`y = A(r)\, e^{j(\theta + \Phi(r))}` with
///
/// .. math::
///
///     A(r) = \frac{\alpha_a r}{1 + \beta_a r^2}, \qquad
///     \Phi(r) = \frac{\alpha_\phi r^2}{1 + \beta_\phi r^2}.
///
/// Args:
///     x (array_like): the input envelope.
///     alpha_a (float): AM/AM coefficient :math:`\alpha_a`.
///     beta_a (float): AM/AM coefficient :math:`\beta_a`.
///     alpha_p (float): AM/PM coefficient :math:`\alpha_\phi`.
///     beta_p (float): AM/PM coefficient :math:`\beta_\phi`.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// References:
///     .. [1] A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear
///        models of TWT amplifiers," *IEEE Trans. Commun.*, 29(11), 1715–1720, 1981.
#[pyfunction]
#[pyo3(signature = (
    x,
    *,
    alpha_a = Saleh::default().alpha_a,
    beta_a = Saleh::default().beta_a,
    alpha_p = Saleh::default().alpha_p,
    beta_p = Saleh::default().beta_p,
))]
fn pa_saleh<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    alpha_a: f64,
    beta_a: f64,
    alpha_p: f64,
    beta_p: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Saleh { alpha_a, beta_a, alpha_p, beta_p }.amplify(&cvec(x)?).py()?))
}

/// The memory-polynomial amplifier: orders 1, 3, 5, memory 3.
///
/// The output of the amplifier of [1]_ is
///
/// .. math::
///
///     y[n] = \sum_{k} \sum_{m} a_{km}\, x[n-m]\, |x[n-m]|^{k-1},
///
/// with the coefficients :math:`a_{km}` of the paper.
///
/// Args:
///     x (array_like): the input envelope.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// References:
///     .. [1] L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim and C. R.
///        Giardina, "A robust digital baseband predistorter constructed using memory
///        polynomials," *IEEE Trans. Commun.*, 52(1), 159–165, 2004.
#[pyfunction]
fn pa_memory_polynomial<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    Ok(carr(py, MemoryPolynomial::ding2004().amplify(&cvec(x)?)))
}

/// A memory-polynomial digital predistorter, trained by indirect learning.
///
/// The predistorter placed before the amplifier is the memory polynomial ([1]_, [2]_,
/// [3]_)
///
/// .. math::
///
///     z[n] = \sum_{k \in \mathcal{K}} \sum_{m=0}^{M-1} c_{km}\, x[n-m]\, |x[n-m]|^{k-1},
///
/// with :math:`\mathcal{K}` the orders, so that the cascade with the amplifier is linear,
/// :math:`F(P(x)) \approx G x`.
///
/// Args:
///     orders (list[int]): nonlinearity orders (odd: 1, 3, 5, …).
///     memory (int): memory depth :math:`M` (samples).
///
/// References:
///     .. [1] C. Eun and E. J. Powers, "A new Volterra predistorter based on the indirect
///        learning architecture," *IEEE Trans. Signal Process.*, 45(1), 223–227, 1997.
///     .. [2] L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim and C. R.
///        Giardina, "A robust digital baseband predistorter constructed using memory
///        polynomials," *IEEE Trans. Commun.*, 52(1), 159–165, 2004.
///     .. [3] D. R. Morgan, Z. Ma, J. Kim, M. G. Zierdt and J. Pastalan, "A generalized
///        memory polynomial model for digital predistortion of RF power amplifiers,"
///        *IEEE Trans. Signal Process.*, 54(10), 3852–3860, 2006.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> g = 1.0513 + 0.0904j                      # small-signal gain of the amplifier
///     >>> x = 0.2 * comnum.Rng(1).complex_noise(20_000, 1.0)
///     >>> dpd = comnum.MemoryPolynomialDpd([1, 3, 5], 3)
///     >>> dpd.train_indirect(x, comnum.pa_memory_polynomial, gain=g, iterations=2)
///     >>> nmse = lambda y: float(np.mean(np.abs(y / g - x) ** 2) / np.mean(np.abs(x) ** 2))
///     >>> nmse(comnum.pa_memory_polynomial(dpd.apply(x))) < nmse(comnum.pa_memory_polynomial(x)) / 5
///     True
///
/// The drive must stay where the amplifier is monotonic: beyond, the post-inverse that
/// indirect learning fits has no meaning and the iterations diverge.
#[pyclass(name = "MemoryPolynomialDpd", module = "comnum")]
pub struct PyDpd {
    inner: MemoryPolynomialDpd,
}

#[pymethods]
impl PyDpd {
    fn __repr__(&self) -> String {
        format!("MemoryPolynomialDpd(orders={:?}, memory={})", self.inner.orders(), self.inner.memory())
    }

    #[new]
    #[pyo3(signature = (
        orders = MemoryPolynomialDpd::DEFAULT_ORDERS.to_vec(),
        memory = MemoryPolynomialDpd::DEFAULT_MEMORY,
    ))]
    fn new(orders: Vec<u32>, memory: usize) -> Self {
        PyDpd { inner: MemoryPolynomialDpd::new(orders, memory) }
    }

    /// numpy.ndarray: The coefficients, order after order, tap after tap (complex128).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.coefficients())
    }

    /// Predistort a block.
    ///
    /// Args:
    ///     x (array_like): the input envelope.
    ///
    /// Returns:
    ///     numpy.ndarray: the predistorted envelope (complex128).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn apply<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let v = cvec(x)?;
        Ok(carr(py, py.detach(|| self.inner.apply(&v))))
    }

    /// Fit the polynomial mapping `x` to `y` by regularised least squares.
    ///
    /// With the basis matrix :math:`\Phi(x)` of the polynomial (one column per term
    /// :math:`k, m`), the coefficients solve
    ///
    /// .. math::
    ///
    ///     \left(\Phi^H \Phi + \lambda I\right) c = \Phi^H y,
    ///
    /// with :math:`\lambda` the regularisation.
    ///
    /// Args:
    ///     x (array_like): the input of the polynomial.
    ///     y (array_like): the output it should give.
    ///     regularization (float): the Tikhonov regularisation.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (x, y, *, regularization=1e-9))]
    fn fit(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, regularization: f64) -> PyResult<()> {
        self.inner.fit(&cvec(x)?, &cvec(y)?, regularization).py()
    }

    /// Indirect learning against an amplifier.
    ///
    /// Args:
    ///     x (array_like): the training signal.
    ///     pa (callable): the amplifier, an array to an array.
    ///     gain (complex): the target complex gain of the cascade.
    ///     iterations (int): iterations of the indirect learning.
    ///     regularization (float): the regularisation of every fit.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (x, pa, *, gain=C64::new(1.0, 0.0), iterations=3, regularization=1e-9))]
    fn train_indirect(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        pa: &Bound<'_, PyAny>,
        gain: C64,
        iterations: usize,
        regularization: f64,
    ) -> PyResult<()> {
        let x = cvec(x)?;
        // the exception raised by `pa`, passed through the Rust error
        let mut raised = None;
        let amplify = |z: &[C64]| {
            pa.call1((carr(py, z.to_vec()),)).and_then(|y| cvec(&y)).map_err(|e| {
                raised = Some(e);
                comnum::Error::Config("the amplifier raised an exception".into())
            })
        };
        let r = self.inner.train_indirect(&x, amplify, gain, iterations, regularization);
        match raised {
            Some(e) => Err(e),
            None => r.py(),
        }
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(amplifier_bussgang, m)?)?;
    m.add_class::<PyDpd>()?;
    m.add_function(wrap_pyfunction!(pa_rapp, m)?)?;
    m.add_function(wrap_pyfunction!(rapp_predistort, m)?)?;
    m.add_function(wrap_pyfunction!(pa_saleh, m)?)?;
    m.add_function(wrap_pyfunction!(pa_memory_polynomial, m)?)?;
    Ok(())
}
