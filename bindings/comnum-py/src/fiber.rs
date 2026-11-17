//! Fibre propagation: fibre parameters, the EDFA, the linear fibre, the split-step
//! Fourier method (one polarisation) and the Manakov equation (two polarisations).

use crate::rng::PyRng;
use crate::{CArr, CMat, IntoPy, carr, cvec, json, mat, py_bool, rng_of, streams_in};
use comnum::channel::optical::edfa::Edfa;
use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::fiber::{self as channels, Amplification, Precision, Ssfm};
use comnum::channel::optical::manakov::ManakovSsfm;
use comnum::channel::optical::pmd::{ManakovPmd, PmdLink};
use comnum::channel::optical::raman::Raman;
use comnum::core::signal::DualPol;
use comnum::dsp::compensation::dbp::{Dbp, NonlinearLength, power_filter};
use comnum::math::nlse::{NonlinearStep, StepDistribution, StepRule, StepUpdate};
use comnum::spec::blocks::BuildContext;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The parameters of a fibre link of identical spans, each followed by an amplifier.
///
/// Engineering units of OptiCommPy: attenuation in dB/km, dispersion parameter :math:`D`
/// in ps/(nm·km), nonlinear coefficient :math:`\gamma` in 1/(W·km), lengths in km,
/// carrier frequency in Hz, noise figure in dB. The defaults describe standard
/// single-mode fibre at 193.1 THz with 80 km spans. Every attribute is readable and
/// writable.
///
/// Args:
///     alpha_db_km (float): attenuation (dB/km).
///     d_ps_nm_km (float): dispersion parameter :math:`D` (ps/(nm·km)).
///     gamma_w_km (float): nonlinear coefficient :math:`\gamma` (1/(W·km)).
///     span_km (float): span length (km).
///     n_spans (int): number of spans.
///     fc_hz (float): carrier frequency (Hz).
///     nf_db (float): noise figure of the amplifiers (dB).
///
/// Examples:
///     >>> import comnum
///     >>> f = comnum.FiberParams(n_spans=10)
///     >>> f.total_km
///     800.0
///     >>> round(f.beta2_ps2_km, 2)
///     -21.75
#[pyclass(name = "FiberParams", module = "comnum")]
#[derive(Clone)]
pub struct PyFiber {
    pub(crate) inner: FiberParams,
}

wrapper_methods! {
    PyFiber {
        /// float: Attenuation (dB/km) (read and write).
        alpha_db_km / set_alpha_db_km: f64,
        /// float: Dispersion parameter :math:`D` (ps/(nm·km)) (read and write).
        d_ps_nm_km / set_d_ps_nm_km: f64,
        /// float: Nonlinear coefficient :math:`\gamma` (1/(W·km)) (read and write).
        gamma_w_km / set_gamma_w_km: f64,
        /// float: Span length (km) (read and write).
        span_km / set_span_km: f64,
        /// int: Number of spans (read and write).
        n_spans / set_n_spans: usize,
        /// float: Carrier frequency (Hz) (read and write).
        fc_hz / set_fc_hz: f64,
        /// float: Noise figure of the amplifiers (dB) (read and write).
        nf_db / set_nf_db: f64,
    }

    #[new]
    #[allow(clippy::too_many_arguments, reason = "the fields of FiberParams, by keyword")]
    #[pyo3(signature = (
        *,
        alpha_db_km = FiberParams::default().alpha_db_km,
        d_ps_nm_km = FiberParams::default().d_ps_nm_km,
        gamma_w_km = FiberParams::default().gamma_w_km,
        span_km = FiberParams::default().span_km,
        n_spans = FiberParams::default().n_spans,
        fc_hz = FiberParams::default().fc_hz,
        nf_db = FiberParams::default().nf_db,
        slope_ps_nm2_km = 0.0,
    ))]
    fn new(
        alpha_db_km: f64,
        d_ps_nm_km: f64,
        gamma_w_km: f64,
        span_km: f64,
        n_spans: usize,
        fc_hz: f64,
        nf_db: f64,
        slope_ps_nm2_km: f64,
    ) -> Self {
        PyFiber {
            inner: FiberParams { alpha_db_km, d_ps_nm_km, gamma_w_km, span_km, n_spans, fc_hz, nf_db, slope_ps_nm2_km },
        }
    }

    /// float: Total length (km).
    #[getter]
    fn total_km(&self) -> f64 {
        self.inner.total_km()
    }

    /// float: Power attenuation :math:`\alpha` (neper/km).
    #[getter]
    fn alpha_np_km(&self) -> f64 {
        self.inner.alpha_np_km()
    }

    /// float: Wavelength :math:`c/f_c` (m).
    #[getter]
    fn wavelength_m(&self) -> f64 {
        self.inner.wavelength_m()
    }

    /// float: Group-velocity dispersion :math:`\beta_2` (s²/km):
    ///
    /// .. math::
    ///
    ///     \beta_2 = -\frac{D\lambda^2}{2\pi c}.
    ///
    #[getter]
    fn beta2_s2_km(&self) -> f64 {
        self.inner.beta2_s2_km()
    }

    /// float: Group-velocity dispersion :math:`\beta_2` (ps²/km).
    #[getter]
    fn beta2_ps2_km(&self) -> f64 {
        self.inner.beta2_ps2_km()
    }

    /// float: Linear power gain :math:`e^{\alpha L}` that compensates the loss of one span.
    #[getter]
    fn span_gain(&self) -> f64 {
        self.inner.span_gain()
    }

    /// float: Span gain (dB).
    #[getter]
    fn span_gain_db(&self) -> f64 {
        self.inner.span_gain_db()
    }

    /// float: Effective length of a span (km):
    ///
    /// .. math::
    ///
    ///     L_{\mathrm{eff}} = \frac{1 - e^{-\alpha L}}{\alpha}.
    ///
    #[getter]
    fn effective_length_km(&self) -> f64 {
        self.inner.effective_length_km()
    }

    /// float: Accumulated dispersion :math:`D \cdot L_{\mathrm{total}}` (ps/nm).
    #[getter]
    fn accumulated_dispersion_ps_nm(&self) -> f64 {
        self.inner.accumulated_dispersion_ps_nm()
    }

    /// Dispersion memory of the link for a signal of bandwidth `bandwidth_hz`.
    ///
    /// The number of samples at :math:`f_s` spanned by the dispersion is
    ///
    /// .. math::
    ///
    ///     N = 2\pi |\beta_2| L B f_s,
    ///
    /// with :math:`L` the total length and :math:`B` the bandwidth.
    ///
    /// Args:
    ///     bandwidth_hz (float): bandwidth of the signal (Hz).
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     float: the memory (samples at `fs`).
    fn dispersion_memory_samples(&self, bandwidth_hz: f64, fs: f64) -> f64 {
        self.inner.dispersion_memory_samples(bandwidth_hz, fs)
    }

    fn __repr__(&self) -> String {
        let f = &self.inner;
        format!(
            "comnum.FiberParams(alpha_db_km={}, d_ps_nm_km={}, gamma_w_km={}, span_km={}, n_spans={}, fc_hz={:e}, nf_db={})",
            f.alpha_db_km, f.d_ps_nm_km, f.gamma_w_km, f.span_km, f.n_spans, f.fc_hz, f.nf_db
        )
    }

    fn __getnewargs_ex__<'py>(&self, py: Python<'py>) -> PyResult<((), Bound<'py, PyDict>)> {
        let f = &self.inner;
        let kw = PyDict::new(py);
        kw.set_item("alpha_db_km", f.alpha_db_km)?;
        kw.set_item("d_ps_nm_km", f.d_ps_nm_km)?;
        kw.set_item("gamma_w_km", f.gamma_w_km)?;
        kw.set_item("span_km", f.span_km)?;
        kw.set_item("n_spans", f.n_spans)?;
        kw.set_item("fc_hz", f.fc_hz)?;
        kw.set_item("nf_db", f.nf_db)?;
        Ok(((), kw))
    }
}

/// An ideal erbium-doped fibre amplifier: flat gain and white ASE.
///
/// The spontaneous-emission factor :math:`n_{\mathrm{sp}}` and the variance
/// :math:`\sigma^2` of the complex Gaussian noise the ASE adds per sample at the sampling
/// rate :math:`f_s` are
///
/// .. math::
///
///     \begin{aligned}
///     n_{\mathrm{sp}} &= \frac{G \cdot \mathrm{NF} - 1}{2(G - 1)}, \\
///     \sigma^2 &= (G - 1)\, n_{\mathrm{sp}}\, h f_c f_s
///     \end{aligned}
///
/// ([1]_, eq. (54); OptiCommPy `edfa`). The defaults are those of OptiCommPy.
///
/// Args:
///     gain_db (float): gain (dB, positive).
///     nf_db (float): noise figure (dB, at least 3 dB).
///     fc_hz (float): carrier frequency (Hz).
///
/// References:
///     .. [1] Essiambre et al., "Capacity limits of optical fiber networks," *J. Lightw.
///        Technol.*, 28(4), 2010.
#[pyclass(name = "Edfa", module = "comnum")]
#[derive(Clone)]
pub struct PyEdfa {
    inner: Edfa,
}

wrapper_methods! {
    PyEdfa {
        /// float: Gain (dB) (read and write).
        gain_db / set_gain_db: f64,
        /// float: Noise figure (dB) (read and write).
        nf_db / set_nf_db: f64,
        /// float: Carrier frequency (Hz) (read and write).
        fc_hz / set_fc_hz: f64,
    }

    #[new]
    #[pyo3(signature = (*, gain_db = Edfa::default().gain_db, nf_db = Edfa::default().nf_db, fc_hz = Edfa::default().fc_hz))]
    fn new(gain_db: f64, nf_db: f64, fc_hz: f64) -> Self {
        PyEdfa { inner: Edfa { gain_db, nf_db, fc_hz } }
    }

    /// float: Spontaneous-emission factor :math:`n_{\mathrm{sp}}`.
    #[getter]
    fn n_sp(&self) -> f64 {
        self.inner.n_sp()
    }

    /// float: ASE power spectral density (W/Hz):
    ///
    /// .. math::
    ///
    ///     N_{\mathrm{ASE}} = (G - 1)\, n_{\mathrm{sp}}\, h f_c.
    ///
    #[getter]
    fn ase_psd(&self) -> f64 {
        self.inner.ase_psd()
    }

    /// Variance of the ASE per sample.
    ///
    /// Args:
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     float: the variance :math:`\mathbb{E}|w|^2` (W).
    fn noise_variance(&self, fs: f64) -> f64 {
        self.inner.noise_variance(fs)
    }

    /// Amplify a field and add the ASE.
    ///
    /// Args:
    ///     x (array_like): the field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng): the generator of the ASE.
    ///
    /// Returns:
    ///     numpy.ndarray: the amplified field (complex128).
    ///
    /// Raises:
    ///     ValueError: for a gain that is not positive or a noise figure below 3 dB.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: f64,
        mut rng: PyRefMut<'py, PyRng>,
    ) -> PyResult<CArr<'py>> {
        let mut y = cvec(x)?;
        self.inner.apply(&mut y, fs, &mut rng.inner).py()?;
        Ok(carr(py, y))
    }

    fn __repr__(&self) -> String {
        let e = &self.inner;
        format!("comnum.Edfa(gain_db={}, nf_db={}, fc_hz={:e})", e.gain_db, e.nf_db, e.fc_hz)
    }

    fn __getnewargs_ex__<'py>(&self, py: Python<'py>) -> PyResult<((), Bound<'py, PyDict>)> {
        let kw = PyDict::new(py);
        kw.set_item("gain_db", self.inner.gain_db)?;
        kw.set_item("nf_db", self.inner.nf_db)?;
        kw.set_item("fc_hz", self.inner.fc_hz)?;
        Ok(((), kw))
    }
}

/// Loss and dispersion of `length_km` of fibre on a block sampled at `fs`
/// (OptiCommPy `linearFiberChannel`).
///
/// The channel [1]_ is exact in the frequency domain on the block:
///
/// .. math::
///
///     A(L, \omega) = A(0, \omega)\, \exp\left(-\frac{\alpha L}{2}
///     + j \frac{\beta_2 \omega^2 L}{2}\right).
///
/// Args:
///     x (array_like): the field (√W).
///     fiber (FiberParams): the fibre.
///     length_km (float): the length (km).
///     fs (float): sampling rate (Hz).
///
/// Returns:
///     numpy.ndarray: the field at the output (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// References:
///     .. [1] G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013.
#[pyfunction]
fn linear_fiber_channel<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    fiber: &PyFiber,
    length_km: f64,
    fs: f64,
) -> PyResult<CArr<'py>> {
    let (x, f) = (cvec(x)?, fiber.inner.clone());
    Ok(carr(py, py.detach(|| channels::linear_fiber_channel(&x, &f, length_km, fs))))
}

/// The Raman amplification of a ``raman`` mapping (the keys of ``fiber.raman`` in a
/// specification).
pub(crate) fn raman(py: Python<'_>, v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Raman>> {
    let Some(v) = v else { return Ok(None) };
    if v.is_none() {
        return Ok(None);
    }
    let j = json(py, v)?;
    BuildContext::default().raman(Some(&serde_json::json!({ "raman": j }))).py()
}

/// The string of a nonlinear step, as the constructors parse it.
fn nonlinear_step_name(step: NonlinearStep) -> &'static str {
    match step {
        NonlinearStep::Midpoint => "midpoint",
        NonlinearStep::Trapezoidal => "trapezoidal",
        NonlinearStep::EffectiveLength => "effective_length",
        _ => "unknown",
    }
}

/// The symmetric split-step Fourier method of OptiCommPy (`ssfm`), one polarisation.
///
/// **The model.** The complex envelope :math:`A(z, t)` of the field (√W), at the distance
/// :math:`z` (km) and the time :math:`t` of the frame that moves with the group velocity,
/// follows the nonlinear Schrödinger equation
///
/// .. math::
///
///     \frac{\partial A}{\partial z} =
///     \underbrace{-\frac{\alpha}{2} A}_{\text{loss}}
///     \; \underbrace{- \, j \frac{\beta_2}{2} \frac{\partial^2 A}{\partial t^2}}_{\text{dispersion}}
///     \; + \underbrace{j \gamma |A|^2 A}_{\text{Kerr effect}},
///
/// with :math:`\alpha` the attenuation of the power (1/km, from ``alpha_db_km``),
/// :math:`\beta_2` the group-velocity dispersion (s²/km, from the dispersion parameter
/// :math:`D`, :attr:`FiberParams.beta2_s2_km`) and :math:`\gamma` the nonlinear
/// coefficient (1/(W·km)). The loss and the dispersion act on the spectrum, the Kerr
/// effect on the instantaneous power: written
///
/// .. math::
///
///     \frac{\partial A}{\partial z} = (\hat D + \hat N) A, \qquad
///     \hat N = j \gamma |A|^2,
///
/// each operator alone has an exact solution over a step, the linear one :math:`\hat D` in
/// the frequency domain and the nonlinear one :math:`\hat N` in the time domain, but not
/// their sum: the split step alternates them.
///
/// **One step.** A step of length :math:`h` applies half of the linear operator, the
/// nonlinear phase over the whole step, then the other half,
///
/// .. math::
///
///     A(z + h) \approx e^{\hat D h/2}\, e^{\hat N h}\, e^{\hat D h/2} A(z),
///
/// a symmetric splitting whose local error is of the third order in :math:`h` ([1]_,
/// Sec. 2.4). The half linear step multiplies
/// the spectrum of the block,
///
/// .. math::
///
///     \tilde A(\omega) \leftarrow \tilde A(\omega)\,
///     \exp\left(-\frac{\alpha h}{4}
///     + j \left(\frac{\beta_2}{2} + \frac{\beta_3 \omega}{6}\right) \frac{\omega^2 h}{2}\right),
///
/// with :math:`\tilde A` the discrete Fourier transform of the block (kernel
/// :math:`e^{-j\omega t}`, the block is periodic), :math:`\omega = 2\pi f` the angular
/// frequency of its bins and :math:`\beta_3` the third-order dispersion of
/// ``slope_ps_nm2_km`` (0 by default). The nonlinear step rotates every sample by a phase
/// proportional to its own power, which it leaves unchanged (hence an exact solution):
///
/// .. math::
///
///     A(t) \leftarrow A(t)\, e^{j \phi(t)}.
///
/// **The phase of the nonlinear step** (``nonlinear_step``), with
/// :math:`A_{1/2} = e^{\hat D h/2} A(z)` the field after the first half step:
///
/// .. math::
///
///     \begin{aligned}
///     \texttt{midpoint}: &\quad \phi = \gamma\, |A_{1/2}|^2\, h, \\
///     \texttt{effective\_length}: &\quad \phi = \gamma\, |A_{1/2}|^2\,
///     \frac{1 - e^{-\alpha h}}{\alpha}, \\
///     \texttt{trapezoidal}: &\quad \phi = \gamma\, \frac{|A(z)|^2 + |A(z + h)|^2}{2}\, h.
///     \end{aligned}
///
/// The midpoint rule takes the power at the middle of the step ([2]_, eq.
/// (3)). Optilux's effective length integrates the loss within the step exactly: its first
/// half step carries the dispersion only, and the attenuation of the whole step,
/// :math:`e^{-\alpha h/2}` on the field, follows the phase with the second half of the
/// dispersion [3]_. OptiCommPy's trapezoidal mean needs the field at
/// the end of the step, which depends on the phase: the step is iterated until it
/// converges.
///
/// **The steps.** Every span is cut into :math:`n = \lceil L_{\mathrm{span}}/h \rceil`
/// equal steps of :math:`\delta = L_{\mathrm{span}}/n` (:math:`h` itself when it divides
/// the span), or, with ``steps_per_span``, into the logarithmic steps of [4]_ (eq. (6)),
///
/// .. math::
///
///     \Delta z_k = -\frac{1}{\alpha} \ln \frac{1 - k\sigma}{1 - (k - 1)\sigma}, \qquad
///     \sigma = \frac{1 - e^{-\alpha L_{\mathrm{span}}}}{n}, \qquad k = 1, \dots, n,
///
/// which carry the same integrated power :math:`\int e^{-\alpha z}\, dz = \sigma/\alpha`
/// each: short at the start of the span, where the power is high, they keep the spurious
/// four-wave mixing of the method near :math:`1/n` of the true one.
///
/// **The amplifiers.** After every span:
///
/// - `"edfa"`: an EDFA of gain :math:`\alpha L` and the noise figure of the fibre (ASE
///   drawn from `rng`, noiseless without one);
/// - `"ideal"`: a noiseless gain :math:`e^{\alpha L/2}` on the field;
/// - `"none"`: nothing.
///
/// With ``raman``, the profile of every span (:func:`raman_profile`, solved for slices of
/// the spectrum of the field at its input) replaces :math:`\alpha` by a gain that depends
/// on the position and the frequency, and the amplifier after the span restores every
/// frequency to its launch power.
///
/// Args:
///     fiber (FiberParams): the link.
///     step_km (float): step size :math:`h` (km).
///     steps_per_span (int, optional): this number of logarithmic steps per span
///         instead of `step_km` ([4]_, eq. (6)): every step
///         carries the same integrated power, short at the start of the span; double
///         precision only.
///     amplification (str): `"edfa"`, `"ideal"` or `"none"`.
///     raman (dict, optional): Raman pumps and ISRS in every span, the keys of
///         ``fiber.raman`` in a specification (see :func:`raman_profile`); the amplifier
///         after a span then restores every frequency to its launch power.
///     nonlinear_step (str): ``"midpoint"`` (the default), ``"effective_length"``
///         (Optilux) or ``"trapezoidal"`` (OptiCommPy's ``manakovSSF``, iterated); double
///         precision only for the last two.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013.
///     .. [2] O. V. Sinkin, R. Holzlöhner, J. Zweck and C. R. Menyuk, "Optimization of
///        the split-step Fourier method in modeling optical-fiber communications
///        systems," *J. Lightw. Technol.*, 21(1), 61–68, 2003.
///     .. [3] J. Shao, X. Liang and S. Kumar, "Comparison of split-step Fourier schemes
///        for simulating fiber optic communication systems," *IEEE Photonics J.*, 6(4),
///        7200515, 2014.
///     .. [4] G. Bosco, A. Carena, V. Curri, R. Gaudino, P. Poggiolini and S. Benedetto,
///        "Suppression of spurious tones induced by the split-step method in fiber
///        systems simulation," *IEEE Photon. Technol. Lett.*, 12(5), 489–491, 2000.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> link = comnum.Ssfm(comnum.FiberParams(n_spans=2), step_km=1.0, amplification="ideal")
///     >>> x = np.exp(2j * np.pi * np.arange(64) / 64) * 1e-2
///     >>> np.allclose(link.backpropagate(link.propagate(x, 32e9), 32e9, steps_per_span=80), x)
///     True
#[pyclass(name = "Ssfm", module = "comnum")]
pub struct PySsfm {
    inner: Ssfm,
}

#[pymethods]
impl PySsfm {
    fn __repr__(&self) -> String {
        let s = &self.inner;
        format!(
            "Ssfm(n_spans={}, span_km={}, {}, nonlinear_step='{}', amplification='{}', raman={})",
            s.fiber.n_spans,
            s.fiber.span_km,
            match s.rule {
                Some(StepRule::Logarithmic(n)) => format!("steps_per_span={n}"),
                _ => format!("step_km={}", s.step_km),
            },
            nonlinear_step_name(s.nonlinear),
            format!("{:?}", s.amplification).to_lowercase(),
            py_bool(s.raman.is_some())
        )
    }

    #[new]
    #[pyo3(signature = (fiber, *, step_km = 1.0, steps_per_span = None, amplification = "edfa", raman = None, nonlinear_step = "midpoint", check_sampling = true))]
    #[allow(clippy::too_many_arguments, reason = "the settings of the link, by keyword")]
    fn new(
        py: Python<'_>,
        fiber: &PyFiber,
        step_km: f64,
        steps_per_span: Option<usize>,
        amplification: &str,
        raman: Option<&Bound<'_, PyAny>>,
        nonlinear_step: &str,
        check_sampling: bool,
    ) -> PyResult<Self> {
        if steps_per_span == Some(0) {
            return Err(PyValueError::new_err("steps_per_span must be at least 1"));
        }
        Ok(PySsfm {
            inner: Ssfm {
                fiber: fiber.inner.clone(),
                step_km,
                amplification: amplification.parse::<Amplification>().py()?,
                raman: self::raman(py, raman)?,
                rule: steps_per_span.map(StepRule::Logarithmic),
                nonlinear: nonlinear_step.parse::<NonlinearStep>().py()?,
                check_sampling,
            },
        })
    }

    /// FiberParams: The link (a copy).
    #[getter]
    fn fiber(&self) -> PyFiber {
        PyFiber { inner: self.inner.fiber.clone() }
    }

    /// float: The step size (km).
    #[getter]
    fn step_km(&self) -> f64 {
        self.inner.step_km
    }

    /// Propagate a field over the link.
    ///
    /// Args:
    ///     x (array_like): the field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng, optional): the generator of the ASE (noiseless without one).
    ///     precision (str): ``"f64"``, or ``"f32"`` (about twice as fast, relative error
    ///         about :math:`10^{-6}\sqrt{\mathrm{steps}}`; the amplifiers stay in double
    ///         precision).
    ///
    /// Returns:
    ///     numpy.ndarray: the field at the end of the link (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (x, fs, *, precision = "f64", rng = None))]
    fn propagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: f64,
        precision: &str,
        rng: Option<PyRefMut<'py, PyRng>>,
    ) -> PyResult<CArr<'py>> {
        let precision: Precision = precision.parse().py()?;
        let mut y = cvec(x)?;
        let mut rng = rng;
        let r = rng_of(&mut rng);
        let s = &self.inner;
        py.detach(|| s.propagate_with(&mut y, fs, r, precision)).py()?;
        Ok(carr(py, y))
    }

    /// Digital back-propagation with `steps_per_span` symmetric steps per span.
    ///
    /// The receiver runs the split step of :class:`Ssfm` through the fibre with its
    /// coefficients negated, the spans in reverse order and the gain of each amplifier
    /// removed first:
    ///
    /// .. math::
    ///
    ///     \frac{\partial A}{\partial z} = \frac{\alpha}{2} A
    ///     + j \frac{\beta_2}{2} \frac{\partial^2 A}{\partial t^2} - j \xi \gamma |A|^2 A.
    ///
    /// With the forward steps, :math:`\xi = 1` and no noise, it inverts `propagate` exactly.
    /// With coarser steps of length :math:`\delta` the phase of a nonlinear step is
    ///
    /// .. math::
    ///
    ///     \phi(t) = -\xi \gamma\, L_{\mathrm{NL}} \sum_{k=-K}^{K} c_k |A(t - k T_s)|^2, \qquad
    ///     L_{\mathrm{NL}} = \frac{2 \sinh(\alpha \delta / 2)}{\alpha},
    ///
    /// with :math:`L_{\mathrm{NL}}` the effective length of the symmetric step (the power at
    /// its middle times :math:`L_{\mathrm{NL}}` is the power integrated over the step;
    /// ``nonlinear_length="step"`` takes :math:`\delta` itself), :math:`\xi` the scale of the
    /// optimised DBP (Ip & Kahn 2008) and :math:`c_k` the taps of the filter on the power:
    /// :math:`c_0 = 1` by default, the Gaussian weights of unit sum
    /// :math:`c_k \propto e^{-k^2/(2\sigma^2)}` with ``power_filter_taps`` :math:`= 2K + 1`
    /// (the weighted DBP of Rafique et al. 2011, eq. (2); the enhanced split step of Häger &
    /// Pfister 2021, eq. (9)): over a long step the dispersion has spread the power of a sample
    /// over its neighbours. A step that covers :math:`s` spans (``steps_per_link``) carries
    /// :math:`s` times the phase of a span, :math:`L_{\mathrm{NL}} = s(1 - e^{-\alpha L})/\alpha`,
    /// applied at its middle.
    ///
    /// Args:
    ///     x (array_like): the received field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     steps_per_span (int): steps per span (1 by default, as the ``dbp`` block).
    ///     nonlinear_scale (float): :math:`\xi`, multiplies the nonlinear phase (the
    ///         optimised DBP with few steps per span, Ip & Kahn 2008; 0 gives the linear
    ///         back-propagation).
    ///     step_distribution (str): ``"uniform"`` (equal steps), ``"logarithmic"`` (Bosco
    ///         et al. 2000: every step carries the same integrated power of the forward
    ///         span, applied from its end; better with 2 to 8 steps per span, worse with
    ///         many) or ``"modified_logarithmic"`` (Zhang et al. 2013: the attenuation
    ///         times ``step_factor``).
    ///     step_factor (float): the adjusting factor of the modified logarithmic steps.
    ///     nonlinear_length (str): ``"effective"``, the effective length of a coarse step
    ///         (the power integrated over it), or ``"step"``, the step itself (the
    ///         midpoint rule of Häger & Pfister 2021).
    ///     steps_per_link (int, optional): this number of equal steps for the whole link
    ///         instead of ``steps_per_span``, a step covering several spans (Rafique et
    ///         al. 2011); the amplifiers must restore the loss of their span.
    ///     power_filter_taps (int): the taps of a Gaussian filter on the power of every
    ///         nonlinear step (odd; 1, the default, for the instantaneous power): the
    ///         weighted DBP of Rafique et al. 2011, the enhanced split step of Häger &
    ///         Pfister 2021, eq. (9).
    ///     power_filter_sigma (float, optional): its standard deviation (samples),
    ///         needed with more than one tap.
    ///
    /// Returns:
    ///     numpy.ndarray: the back-propagated field (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (x, fs, *, steps_per_span = 1, nonlinear_scale = 1.0, step_distribution = "uniform", step_factor = 0.4, nonlinear_length = "effective", steps_per_link = None, power_filter_taps = 1, power_filter_sigma = None))]
    #[allow(clippy::too_many_arguments, reason = "the settings of the back-propagation, by keyword")]
    fn backpropagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: f64,
        steps_per_span: usize,
        nonlinear_scale: f64,
        step_distribution: &str,
        step_factor: f64,
        nonlinear_length: &str,
        steps_per_link: Option<usize>,
        power_filter_taps: usize,
        power_filter_sigma: Option<f64>,
    ) -> PyResult<CArr<'py>> {
        let x = cvec(x)?;
        let distribution = step_distribution.parse::<StepDistribution>().py()?.with_factor(step_factor);
        let dbp = Dbp {
            distribution,
            nonlinear_length: nonlinear_length.parse::<NonlinearLength>().py()?,
            steps_per_link,
            power_filter: power_filter(power_filter_taps, power_filter_sigma).py()?,
            ..self.inner.dbp(steps_per_span, nonlinear_scale)
        };
        Ok(carr(py, py.detach(|| dbp.run(&x, fs)).py()?))
    }
}

/// Dual-polarisation propagation by the Manakov equation with polarisation-mode
/// dispersion (coarse-step model).
///
/// Every span is cut into `sections_per_span` sections; a section applies the loss and
/// the dispersion, a random rotation and a differential group delay (the PMD), then the
/// averaged Kerr phase of the Manakov-PMD equation [1]_
///
/// .. math::
///
///     \phi = \frac{8}{9}\, \gamma L_{\mathrm{eff}} \left(|A_x|^2 + |A_y|^2\right).
///
/// The amplification after every span is an EDFA (ASE from `rng` if given).
///
/// Args:
///     fiber (FiberParams): the link.
///     pmd_ps_sqrt_km (float): PMD coefficient (ps/√km); 0 for none.
///     sections_per_span (int): sections per span.
///     rng (Rng, optional): draws the PMD realisation (needed when the PMD is not 0).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] D. Marcuse, C. R. Menyuk and P. K. A. Wai, "Application of the Manakov-PMD
///        equation to studies of signal propagation in optical fibers with randomly
///        varying birefringence," *J. Lightw. Technol.*, 15(9), 1735–1746, 1997.
#[pyclass(name = "ManakovPmd", module = "comnum")]
pub struct PyManakov {
    inner: ManakovPmd,
}

#[pymethods]
impl PyManakov {
    fn __repr__(&self) -> String {
        format!("ManakovPmd(n_spans={}, rms_dgd_s={:e})", self.inner.fiber.n_spans, self.inner.pmd.rms_dgd_s())
    }

    #[new]
    #[pyo3(signature = (fiber, *, pmd_ps_sqrt_km = 0.0, sections_per_span = 10, rng = None))]
    fn new(
        fiber: &PyFiber,
        pmd_ps_sqrt_km: f64,
        sections_per_span: usize,
        rng: Option<PyRefMut<'_, PyRng>>,
    ) -> PyResult<Self> {
        let f = fiber.inner.clone();
        let pmd = match (pmd_ps_sqrt_km, rng) {
            (0.0, _) => PmdLink::none(&f, sections_per_span).py()?,
            (p, Some(mut r)) => PmdLink::random(&f, p, sections_per_span, &mut r.inner).py()?,
            (_, None) => return Err(PyValueError::new_err("a PMD realisation needs an rng")),
        };
        Ok(PyManakov { inner: ManakovPmd::new(f, pmd) })
    }

    /// float: RMS differential group delay of the PMD realisation (s).
    #[getter]
    fn rms_dgd_s(&self) -> f64 {
        self.inner.pmd.rms_dgd_s()
    }

    /// Propagate the two polarisations.
    ///
    /// Args:
    ///     u (array_like): the field, `(2, n)`: the two polarisations (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng, optional): the generator of the ASE (noiseless without one).
    ///
    /// Returns:
    ///     numpy.ndarray: the field at the end of the link, `(2, n)` (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (u, fs, *, rng = None))]
    fn propagate<'py>(
        &self,
        py: Python<'py>,
        u: &Bound<'py, PyAny>,
        fs: f64,
        rng: Option<PyRefMut<'py, PyRng>>,
    ) -> PyResult<CMat<'py>> {
        let mut u = dual_pol(u)?;
        let mut rng = rng;
        let r = rng_of(&mut rng);
        let m = &self.inner;
        py.detach(|| m.propagate(&mut u, fs, r)).py()?;
        field_out(py, u)
    }

    /// Back-propagation with the known PMD realisation (inverts `propagate` without
    /// noise).
    ///
    /// Args:
    ///     u (array_like): the received field, `(2, n)` (√W).
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     numpy.ndarray: the back-propagated field, `(2, n)` (complex128).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn backpropagate_with_pmd<'py>(&self, py: Python<'py>, u: &Bound<'py, PyAny>, fs: f64) -> PyResult<CMat<'py>> {
        let mut u = dual_pol(u)?;
        let m = &self.inner;
        py.detach(|| m.backpropagate_with_pmd(&mut u, fs));
        field_out(py, u)
    }

    /// Undo the lumped PMD of the link (inverse of its frequency response), leaving the
    /// dispersion and the nonlinearity.
    ///
    /// Args:
    ///     u (array_like): the field, `(2, n)` (√W).
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     numpy.ndarray: the field without the PMD, `(2, n)` (complex128).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn equalize_pmd<'py>(&self, py: Python<'py>, u: &Bound<'py, PyAny>, fs: f64) -> PyResult<CMat<'py>> {
        let mut u = dual_pol(u)?;
        let m = &self.inner;
        py.detach(|| m.equalize_pmd(&mut u, fs));
        field_out(py, u)
    }
}

/// Dual-polarisation nonlinear propagation over the link: the Manakov equation solved
/// by the symmetric split-step method (``nonlinear_step="trapezoidal"`` is OptiCommPy's
/// ``manakovSSF``, ``"effective_length"`` Optilux's ``fiber``).
///
/// Averaged over the fast rotations of the birefringence, the two polarisations
/// :math:`A_x`, :math:`A_y` obey [1]_
///
/// .. math::
///
///     \frac{\partial A_{x,y}}{\partial z} = -\frac{\alpha}{2} A_{x,y}
///     - j \frac{\beta_2}{2} \frac{\partial^2 A_{x,y}}{\partial t^2}
///     + j \frac{8}{9} \gamma \left(|A_x|^2 + |A_y|^2\right) A_{x,y}.
///
/// The factor :math:`8/9` is the average of the self- and cross-phase modulation of the two
/// polarisations over the Poincaré sphere. The equation is solved by the symmetric split
/// step of :class:`Ssfm`: a half linear step (the same on both polarisations), the phase
/// :math:`\phi(t)` on both, then a half linear step, with
///
/// .. math::
///
///     \begin{aligned}
///     \texttt{midpoint}: &\quad \phi = \tfrac{8}{9} \gamma\, P_{1/2}\, h, \\
///     \texttt{effective\_length}: &\quad \phi = \tfrac{8}{9} \gamma\, P_{1/2}\,
///     \frac{1 - e^{-\alpha h}}{\alpha}, \\
///     \texttt{trapezoidal}: &\quad \phi = \tfrac{8}{9} \gamma\, \frac{P(z) + P(z + h)}{2}\, h,
///     \end{aligned}
///
/// :math:`P = |A_x|^2 + |A_y|^2` the total power and :math:`P_{1/2}` that after the first
/// half step. The trapezoidal rule (OptiCommPy) is iterated until the relative change of
/// the fields falls below ``tolerance``; its mean smooths the cross-phase modulation of
/// channels that walk off within a step. The effective length (Optilux) carries the
/// dispersion only in the first half step and the attenuation of the whole step after the
/// phase. The step is fixed (``step_km``) or adaptive (``step_rule``). An amplifier follows
/// every span. Polarisation-mode dispersion is neglected (see :class:`ManakovPmd`).
///
/// Args:
///     fiber (FiberParams): the link.
///     step_rule (str): ``"fwm"``, the rule of [2]_: the first step of a
///         span bounds the four-wave-mixing phase across the signal bandwidth to
///         ``fwm_phase``, the next ones grow with the loss (about 0.01 dB of error on the
///         SNR at 20 rad, whatever the power); ``"nonlinear_phase"``, OptiCommPy's
///         ``max_nl_phase`` per step; or ``"logarithmic"``, ``steps_per_span``
///         logarithmic steps ([3]_, eq. (6)).
///     steps_per_span (int): logarithmic steps per span, with
///         ``step_rule="logarithmic"``.
///     fwm_phase (float): the four-wave-mixing phase of the first step (rad).
///     bandwidth_hz (float, optional): the signal bandwidth of the rule ``"fwm"``
///         (measured on the launched field if None).
///     max_nl_phase (float): largest nonlinear phase rotation per step (rad), with
///         ``step_rule="nonlinear_phase"``.
///     step_km (float, optional): a fixed step (km) instead of the adaptive one.
///     nonlinear_step (str): ``"midpoint"``, the power at the middle of a step (the
///         scalar solver); ``"effective_length"``, Optilux's power at the start of the
///         nonlinear block over the effective length of the step; or ``"trapezoidal"``,
///         the mean of the power at its two ends, iterated (OptiCommPy): the trapezoidal
///         mean smooths the cross-phase modulation of channels that walk off within a
///         step.
///     max_iterations (int): trapezoidal iterations per step at most.
///     tolerance (float): convergence tolerance of the iterations.
///     amplification (str): ``"edfa"``, ``"ideal"`` or ``"none"``.
///     raman (dict, optional): Raman pumps and ISRS in every span (see
///         :func:`raman_profile`).
///     check_sampling (bool): refuse a field whose sampling rate is below 3/2 of its
///         bandwidth or whose block is shorter than the walk-off of the link (the
///         four-wave mixing would alias, [2]_, Appendix B).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] D. Marcuse, C. R. Menyuk and P. K. A. Wai, "Application of the Manakov-PMD
///        equation to studies of signal propagation in optical fibers with randomly
///        varying birefringence," *J. Lightw. Technol.*, 15(9), 1735–1746, 1997.
///     .. [2] S. Musetti, P. Serena and A. Bononi, "On the accuracy of split-step Fourier
///        simulations for wideband nonlinear optical communications," *J. Lightw.
///        Technol.*, 36(23), 5669–5677, 2018.
///     .. [3] G. Bosco, A. Carena, V. Curri, R. Gaudino, P. Poggiolini and S. Benedetto,
///        "Suppression of spurious tones induced by the split-step method in fiber
///        systems simulation," *IEEE Photon. Technol. Lett.*, 12(5), 489–491, 2000.
#[pyclass(name = "ManakovSsfm", module = "comnum")]
pub struct PyManakovSsfm {
    inner: ManakovSsfm,
    last_steps: Option<usize>,
}

#[pymethods]
impl PyManakovSsfm {
    fn __repr__(&self) -> String {
        let m = &self.inner;
        format!(
            "ManakovSsfm(n_spans={}, rule={:?}, nonlinear_step='{}', max_iterations={}, tolerance={:e}, amplification='{}')",
            m.fiber.n_spans,
            m.rule,
            nonlinear_step_name(m.nonlinear),
            m.max_iterations,
            m.tolerance,
            format!("{:?}", m.amplification).to_lowercase()
        )
    }

    #[new]
    #[pyo3(signature = (fiber, *, step_rule = "fwm", fwm_phase = 20.0, bandwidth_hz = None, max_nl_phase = 2e-2, steps_per_span = 100, step_km = None, nonlinear_step = "midpoint", max_iterations = 10, tolerance = 1e-5, amplification = "edfa", raman = None, check_sampling = true))]
    #[allow(clippy::too_many_arguments, reason = "the settings of the solver, by keyword")]
    fn new(
        py: Python<'_>,
        fiber: &PyFiber,
        step_rule: &str,
        fwm_phase: f64,
        bandwidth_hz: Option<f64>,
        max_nl_phase: f64,
        steps_per_span: usize,
        step_km: Option<f64>,
        nonlinear_step: &str,
        max_iterations: usize,
        tolerance: f64,
        amplification: &str,
        raman: Option<&Bound<'_, PyAny>>,
        check_sampling: bool,
    ) -> PyResult<Self> {
        let rule = match (step_km, step_rule) {
            (Some(h), _) => StepRule::Fixed(h),
            (None, "fwm") => StepRule::Fwm {
                phase_rad: fwm_phase,
                bandwidth_hz,
                update: StepUpdate::ConstantLocalError,
                max_nl_phase_rad: Some(max_nl_phase),
            },
            (None, "nonlinear_phase") => StepRule::NonlinearPhase(max_nl_phase),
            (None, "logarithmic") => StepRule::Logarithmic(steps_per_span),
            (None, other) => {
                return Err(PyValueError::new_err(format!(
                    "step_rule must be 'fwm', 'nonlinear_phase' or 'logarithmic', got {other:?}"
                )));
            }
        };
        Ok(PyManakovSsfm {
            inner: ManakovSsfm {
                fiber: fiber.inner.clone(),
                rule,
                nonlinear: nonlinear_step.parse::<NonlinearStep>().py()?,
                max_iterations,
                tolerance,
                amplification: amplification.parse::<Amplification>().py()?,
                raman: self::raman(py, raman)?,
                check_sampling,
            },
            last_steps: None,
        })
    }

    /// int | None: The number of steps of the last propagation (None before the first).
    #[getter]
    fn last_steps(&self) -> Option<usize> {
        self.last_steps
    }

    /// Propagate the two polarisations.
    ///
    /// Args:
    ///     u (array_like): the field, `(2, n)`: the two polarisations (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng, optional): the generator of the ASE (noiseless without one).
    ///
    /// Returns:
    ///     numpy.ndarray: the field at the end of the link, `(2, n)` (complex128); the
    ///     number of steps is kept in :attr:`last_steps`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (u, fs, *, rng = None))]
    fn propagate<'py>(
        &mut self,
        py: Python<'py>,
        u: &Bound<'py, PyAny>,
        fs: f64,
        rng: Option<PyRefMut<'py, PyRng>>,
    ) -> PyResult<CMat<'py>> {
        let mut u = dual_pol(u)?;
        let mut rng = rng;
        let r = rng_of(&mut rng);
        let m = &self.inner;
        let steps = py.detach(|| m.propagate(&mut u, fs, r)).py()?;
        self.last_steps = Some(steps);
        field_out(py, u)
    }

    /// Dual-polarisation digital back-propagation with `steps_per_span` symmetric steps
    /// per span (the Manakov equation with :math:`-\alpha`, :math:`-\beta_2`,
    /// :math:`-\gamma`, the amplifier gain removed before each span, the nonlinear phase
    /// of a step on its effective length).
    ///
    /// Args:
    ///     u (array_like): the received field, `(2, n)` (√W).
    ///     fs (float): sampling rate (Hz).
    ///     steps_per_span (int): steps per span.
    ///     nonlinear_scale (float): :math:`\xi`, multiplies the nonlinear phase (0:
    ///         dispersion compensation only).
    ///     steps_per_link (int, optional): this number of equal steps for the whole link
    ///         instead of ``steps_per_span``, a step covering several spans (Rafique et
    ///         al. 2011); the amplifiers must restore the loss of their span.
    ///     power_filter_taps (int): the taps of a Gaussian filter on the total power of
    ///         every nonlinear step (odd; 1, the default, for the instantaneous power):
    ///         the weighted DBP of Rafique et al. 2011, the enhanced split step of Häger &
    ///         Pfister 2021, eq. (9).
    ///     power_filter_sigma (float, optional): its standard deviation (samples),
    ///         needed with more than one tap.
    ///
    /// Returns:
    ///     numpy.ndarray: the back-propagated field, `(2, n)` (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (u, fs, *, steps_per_span = 1, nonlinear_scale = 1.0, steps_per_link = None, power_filter_taps = 1, power_filter_sigma = None))]
    #[allow(clippy::too_many_arguments, reason = "the settings of the back-propagation, by keyword")]
    fn backpropagate<'py>(
        &self,
        py: Python<'py>,
        u: &Bound<'py, PyAny>,
        fs: f64,
        steps_per_span: usize,
        nonlinear_scale: f64,
        steps_per_link: Option<usize>,
        power_filter_taps: usize,
        power_filter_sigma: Option<f64>,
    ) -> PyResult<CMat<'py>> {
        let DualPol { mut x, mut y } = dual_pol(u)?;
        let d = Dbp {
            steps_per_link,
            power_filter: power_filter(power_filter_taps, power_filter_sigma).py()?,
            ..self.inner.dbp(steps_per_span, nonlinear_scale)
        };
        py.detach(|| d.run_dual(&mut x, &mut y, fs)).py()?;
        mat(py, vec![x, y])
    }
}

/// The two polarisations of a dual-polarisation field given as a `(2, n)` array.
fn dual_pol(u: &Bound<'_, PyAny>) -> PyResult<DualPol> {
    let (rows, _) = streams_in(u)?;
    let [x, y]: [Vec<_>; 2] = rows.try_into().map_err(|r: Vec<_>| {
        PyValueError::new_err(format!("a dual-polarisation field is a (2, n) array, got {} streams", r.len()))
    })?;
    DualPol::new(x, y).py()
}

/// A dual-polarisation field as a `(2, n)` array.
fn field_out(py: Python<'_>, u: DualPol) -> PyResult<CMat<'_>> {
    mat(py, vec![u.x, u.y])
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFiber>()?;
    m.add_class::<PyEdfa>()?;
    m.add_class::<PySsfm>()?;
    m.add_class::<PyManakov>()?;
    m.add_class::<PyManakovSsfm>()?;
    m.add_function(wrap_pyfunction!(linear_fiber_channel, m)?)?;
    Ok(())
}
