//! The closed forms of the error and information rates of the constellations.

use super::*;

/// Theoretical BER over AWGN with Gray mapping (OptiCommPy `theoryBER`, exact for BPSK).
///
/// With :math:`L = \sqrt{M}` for the QAM:
///
/// .. math::
///
///     \begin{aligned}
///     P_b^{\mathrm{QAM}} &= \frac{2(1 - 1/L)}{\log_2 L}\,
///     Q\!\left(\sqrt{\frac{3 \log_2 L}{L^2 - 1} \cdot \frac{2 E_b}{N_0}}\right), \\
///     P_b^{\mathrm{PSK}} &= \frac{2}{\log_2 M}\,
///     Q\!\left(\sqrt{2 \log_2 M \cdot E_b/N_0}\, \sin(\pi/M)\right), \\
///     P_b^{\mathrm{PAM}} &= \frac{2(M - 1)}{M \log_2 M}\,
///     Q\!\left(\sqrt{\frac{6 \log_2 M}{M^2 - 1} \cdot \frac{E_b}{N_0}}\right),
///     \end{aligned}
///
/// and the exact :math:`Q(\sqrt{2 E_b/N_0})` for BPSK (OptiCommPy applies the general
/// formula of PSK, which doubles the BER of BPSK).
///
/// Args:
///     m (int): order.
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///     kind (str): `"qam"`, `"psk"` or `"pam"`.
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_ber(4, 0.0), 6)
///     0.07865
///     >>> comnum.theory_ber(16, [8.0, 12.0], "qam").shape
///     (2,)
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, kind = "qam"))]
fn theory_ber(m: usize, ebn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
    let k = self::kind(kind)?;
    map_real(ebn0_db, |e| metrics::theory_ber(k, m, e).py())
}

/// Exact BER of Gray-mapped PAM, square QAM or rectangular QAM over AWGN (K. Cho and
/// D. Yoon, *IEEE Trans. Commun.* 50(7), 2002, eqs. (9)–(22)).
///
/// For an :math:`I`-PAM (eqs. (9)–(10)):
///
/// .. math::
///
///     P_b = \frac{1}{\log_2 I} \sum_{k=1}^{\log_2 I} P_b(k), \qquad
///     P_b(k) = \frac{1}{I} \sum_i X_{i,k}\, \operatorname{erfc}\!\left((2i + 1)
///     \sqrt{\frac{3 \log_2 I \cdot E_b/N_0}{I^2 - 1}}\right),
///
/// with the coefficients :math:`X_{i,k}` of Table I of the paper. An :math:`I \times J`
/// QAM is two such PAM in quadrature (eqs. (19)–(22)):
///
/// .. math::
///
///     P_b = \frac{1}{\log_2(IJ)} \left(\sum_{k=1}^{\log_2 I} P_I(k)
///     + \sum_{l=1}^{\log_2 J} P_J(l)\right),
///
/// the same terms with the argument
/// :math:`(2i + 1)\sqrt{3 \log_2(IJ)\, E_b/N_0 / (I^2 + J^2 - 2)}`, and
/// :math:`I = J = \sqrt{M}` for a square QAM.
///
/// Args:
///     m (int): order of the PAM (`kind="pam"`), of the square QAM (`kind="qam"`), or the
///         number of in-phase levels of a rectangular QAM (`kind="qam"` with `mq`).
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///     kind (str): `"pam"` or `"qam"`.
///     mq (int | None): the number of quadrature levels of a rectangular
///         :math:`m \times \mathit{mq}` QAM.
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_ber_gray(4, 0.0, "qam"), 6)
///     0.07865
///     >>> bool(comnum.theory_ber_gray(8, 12.0, "qam", mq=4) < comnum.theory_ber_gray(64, 12.0))
///     True
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, kind = "qam", *, mq = None))]
fn theory_ber_gray(m: usize, ebn0_db: &Bound<'_, PyAny>, kind: &str, mq: Option<usize>) -> PyResult<Py<PyAny>> {
    use comnum::mapping::constellation::ConstellationKind as Kind;
    use comnum::mapping::theory as th;
    match (self::kind(kind)?, mq) {
        (Kind::Pam, None) => map_real(ebn0_db, |e| th::ber_gray_pam(m, e).py()),
        (Kind::Qam, None) => map_real(ebn0_db, |e| th::ber_gray_qam(m, e).py()),
        (Kind::Qam, Some(j)) => map_real(ebn0_db, |e| th::ber_gray_rect_qam(m, j, e).py()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "the exact Gray BER is for kind \"pam\" or \"qam\" (mq with qam only), got {kind:?}"
        ))),
    }
}

/// BER of BPSK (or Gray QPSK, per bit) with maximal-ratio combining of ``branches``
/// independent Rayleigh-faded branches (J. G. Proakis and M. Salehi, *Digital
/// Communications*, 5th ed., eq. (13.4-15)):
///
/// .. math::
///
///     P_b = \left(\frac{1 - \mu}{2}\right)^L \sum_{k<L} \binom{L-1+k}{k}
///     \left(\frac{1 + \mu}{2}\right)^k, \qquad \mu = \sqrt{\gamma/(1 + \gamma)},
///
/// with :math:`\gamma` the mean :math:`E_b/N_0` of one branch; one branch is the flat
/// Rayleigh channel, :math:`P_b = (1 - \mu)/2`.
///
/// Args:
///     ebn0_db (float | array_like): the mean :math:`E_b/N_0` of one branch (dB).
///     branches (int): the number of branches :math:`L` (at least 1).
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for no branch.
///
/// Examples:
///     >>> import comnum, math
///     >>> g = comnum.db2lin(10.0)
///     >>> round(comnum.theory_ber_rayleigh_mrc(10.0) / (0.5 * (1 - math.sqrt(g / (1 + g)))), 12)
///     1.0
///     >>> bool(comnum.theory_ber_rayleigh_mrc(10.0, branches=4) < 1e-5)
///     True
#[pyfunction]
#[pyo3(signature = (ebn0_db, *, branches = 1))]
fn theory_ber_rayleigh_mrc(ebn0_db: &Bound<'_, PyAny>, branches: usize) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| theory::ber_rayleigh_mrc(comnum::math::db2lin(e), branches).py())
}

/// A flat fading channel: the distribution of its complex gain, of unit mean power.
/// One object for the simulation (:meth:`gains`) and for the theory (:meth:`ber`,
/// :meth:`ser`, :meth:`ber_noncoherent`, :meth:`mgf`), so that a simulated curve and
/// its closed form are those of the same channel.
///
/// - ``"rayleigh"``: scattered paths only, :math:`h \sim \mathcal{CN}(0, 1)`;
/// - ``"rice"`` with the Rician factor :math:`K` (linear): a line of sight of power
///   :math:`K/(1 + K)` plus a scattered term;
/// - ``"nakagami"`` with :math:`m \ge 1/2`: :math:`|h|^2` a gamma variable of shape
///   :math:`m` and unit mean, a uniform phase;
/// - ``"hoyt"`` with :math:`0 \le q \le 1`: a Gaussian of variance :math:`1/(1 + q^2)` on
///   the real axis and :math:`q^2/(1 + q^2)` on the imaginary one;
/// - ``"awgn"``: no fading, :math:`h = 1`.
///
/// The densities are those of Table 2 of M. K. Simon and M.-S. Alouini, "A unified
/// approach to the performance analysis of digital communication over generalized
/// fading channels," Proc. IEEE 86(9), 1998; the error rates are the averages by the
/// moment generating function of the SNR (its Table 3; M.-S. Alouini and A. J.
/// Goldsmith, IEEE Trans. Commun. 47(9), 1999).
///
/// Args:
///     kind (str): ``"rayleigh"`` (the default), ``"rice"``, ``"nakagami"``, ``"hoyt"``
///         or ``"awgn"``.
///     parameter (float, optional): :math:`K` (linear) for Rice, :math:`m` for Nakagami,
///         :math:`q` for Hoyt; none for Rayleigh and AWGN.
///
/// Raises:
///     ValueError: for an unknown channel, a missing or an invalid parameter.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> ch = comnum.Fading("nakagami", 2.0)
///     >>> h = ch.gains(200_000, comnum.Rng(1))
///     >>> bool(abs(np.mean(np.abs(h) ** 2) - 1.0) < 0.01)
///     True
///     >>> round(float(ch.ber(10.0)), 6)
///     0.005528
#[pyclass(name = "Fading", module = "comnum", frozen)]
#[derive(Clone)]
pub struct PyFading {
    kind: String,
    parameter: Option<f64>,
    snr: theory::SnrFading,
}

impl PyFading {
    fn build(kind: &str, parameter: Option<f64>) -> PyResult<PyFading> {
        let kind = kind.to_ascii_lowercase();
        // the channels without a parameter ignore one
        let parameter = if matches!(kind.as_str(), "rayleigh" | "awgn") { None } else { parameter };
        let snr = theory::SnrFading::from_name(&kind, parameter).py()?;
        Ok(PyFading { kind, parameter, snr })
    }
}

#[pymethods]
impl PyFading {
    #[new]
    #[pyo3(signature = (kind = "rayleigh", parameter = None))]
    fn new(kind: &str, parameter: Option<f64>) -> PyResult<Self> {
        PyFading::build(kind, parameter)
    }

    /// str: the name of the channel.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// float | None: its parameter (:math:`K`, :math:`m` or :math:`q`).
    #[getter]
    fn parameter(&self) -> Option<f64> {
        self.parameter
    }

    fn __repr__(&self) -> String {
        match self.parameter {
            Some(p) => format!("Fading({:?}, {p:?})", self.kind),
            None => format!("Fading({:?})", self.kind),
        }
    }

    /// Gains of unit mean power, one per block of `coherence` samples.
    ///
    /// Args:
    ///     n (int): number of gains.
    ///     rng (Rng): the generator.
    ///     coherence (int): samples sharing one gain (1, the default: a new gain for
    ///         every sample).
    ///
    /// Returns:
    ///     numpy.ndarray: `n` gains (complex128); ones without fading.
    ///
    /// Raises:
    ///     ValueError: for a negative count or a zero coherence.
    #[pyo3(signature = (n, rng, *, coherence = 1))]
    fn gains<'py>(&self, py: Python<'py>, n: i64, rng: &mut PyRng, coherence: usize) -> PyResult<CArr<'py>> {
        let n = count(n, "n")?;
        Ok(carr(
            py,
            match self.snr.gain_distribution() {
                Some(f) => flat_gains(f, n, coherence, &mut rng.inner).py()?,
                None => vec![num_complex::Complex64::new(1.0, 0.0); n],
            },
        ))
    }

    /// Moment generating function of the SNR :math:`\gamma` at :math:`s \le 0`.
    ///
    /// .. math::
    ///
    ///     M(s) = \mathbb{E} \exp(s \gamma).
    ///
    /// Args:
    ///     s (float | array_like): the argument, not positive.
    ///     snr_db (float): the mean SNR (dB).
    ///
    /// Returns:
    ///     float | numpy.ndarray: the MGF at every `s`.
    ///
    /// Raises:
    ///     ValueError: for a positive `s`.
    fn mgf(&self, s: &Bound<'_, PyAny>, snr_db: f64) -> PyResult<Py<PyAny>> {
        let mean = comnum::math::db2lin(snr_db);
        map_real(s, |v| self.snr.mgf(v, mean).py())
    }

    /// Average BER of a coherent binary signal (:func:`theory_ber_binary_fading`).
    ///
    /// The conditional BER :math:`Q(\sqrt{2 g \gamma})` averaged over the SNR, with
    /// maximal-ratio combining of :math:`L` branches:
    ///
    /// .. math::
    ///
    ///     P_b = \frac{1}{\pi} \int_0^{\pi/2}
    ///     \left[M_\gamma\!\left(-\frac{g}{\sin^2 \varphi}\right)\right]^L d\varphi.
    ///
    /// Args:
    ///     ebn0_db (float | array_like): the mean :math:`E_b/N_0` of one branch (dB).
    ///     g (float): the constant of the signal: 1 for BPSK and Gray QPSK per bit, 1/2
    ///         for orthogonal BFSK.
    ///     branches (int): branches combined by maximal ratio.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (ebn0_db, *, g = 1.0, branches = 1))]
    fn ber(&self, ebn0_db: &Bound<'_, PyAny>, g: f64, branches: usize) -> PyResult<Py<PyAny>> {
        map_real(ebn0_db, |e| theory::ber_binary_fading(g, self.snr, e, branches).py())
    }

    /// Average BER of a binary signal detected noncoherently or differentially
    /// (:func:`theory_ber_noncoherent_binary_fading`).
    ///
    /// The conditional BER :math:`0.5 \exp(-g \gamma)` averaged over the SNR:
    ///
    /// .. math::
    ///
    ///     P_b = 0.5\, M_\gamma(-g).
    ///
    /// Args:
    ///     ebn0_db (float | array_like): the mean :math:`E_b/N_0` (dB).
    ///     g (float): 1 for DPSK, 1/2 for noncoherent orthogonal BFSK.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (ebn0_db, *, g = 1.0))]
    fn ber_noncoherent(&self, ebn0_db: &Bound<'_, PyAny>, g: f64) -> PyResult<Py<PyAny>> {
        map_real(ebn0_db, |e| theory::ber_noncoherent_binary_fading(g, self.snr, e).py())
    }

    /// Average SER of M-PSK, square M-QAM or M-PAM (:func:`theory_ser_fading`).
    ///
    /// Args:
    ///     m (int): the order of the constellation.
    ///     esn0_db (float | array_like): the mean :math:`E_s/N_0` (dB).
    ///     kind (str): ``"psk"``, ``"qam"`` or ``"pam"``.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the symbol error probability at every :math:`E_s/N_0`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (m, esn0_db, kind = "qam"))]
    fn ser(&self, m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
        let k = self::kind(kind)?;
        map_real(esn0_db, |e| theory::ser_fading(k, m, self.snr, e).py())
    }
}

/// The channel of the `fading` argument of the closed forms: a :class:`Fading`, a name
/// with its `parameter`, or Rayleigh fading when it is left out.
fn snr_fading(fading: Option<&Bound<'_, PyAny>>, parameter: Option<f64>) -> PyResult<theory::SnrFading> {
    let Some(fading) = fading else {
        return Ok(theory::SnrFading::Rayleigh);
    };
    if let Ok(f) = fading.cast::<PyFading>() {
        return Ok(f.get().snr);
    }
    let name: String =
        fading.extract().map_err(|_| pyo3::exceptions::PyValueError::new_err("fading must be a name or a Fading"))?;
    Ok(PyFading::build(&name, parameter)?.snr)
}

/// Average symbol error rate of M-PSK, square M-QAM or M-PAM over a flat fading
/// channel, by the moment generating function (MGF) of the SNR with Craig's form of the
/// Q function (M.-S. Alouini and A. J. Goldsmith, "A unified approach for calculating
/// error rates of linearly modulated signals over generalized fading channels," IEEE
/// Trans. Commun. 47(9), 1999, eqs. (41), (45), (48)):
///
/// .. math::
///
///     P_s = \frac{1}{\pi} \int M_\gamma\!\left(-\frac{g}{\sin^2 \varphi}\right) d\varphi
///
/// over the intervals of the family, with the MGF of Rayleigh, Rice (Rician factor
/// :math:`K`), Nakagami-m (:math:`m`) or Hoyt (:math:`q`) fading (eqs. (23), (28), (30), (26)).
///
/// Args:
///     m (int): the order of the constellation.
///     esn0_db (float | array_like): the mean :math:`E_s/N_0` (dB).
///     kind (str): ``"psk"``, ``"qam"`` or ``"pam"``.
///     fading (str | Fading, optional): a :class:`Fading`, or ``"rayleigh"`` (the
///         default), ``"rice"``, ``"nakagami"``, ``"hoyt"`` or ``"awgn"``.
///     parameter (float, optional): :math:`K` (linear) for Rice, :math:`m` for Nakagami,
///         :math:`q` for Hoyt; none for Rayleigh and AWGN.
///
/// Returns:
///     float | numpy.ndarray: the symbol error probability at every :math:`E_s/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum, math
///     >>> # 16-QAM over Rayleigh fading at 20 dB: the closed form (49) of the paper
///     >>> g, q = 0.1 * 100, 0.75
///     >>> mu = math.sqrt(g / (1 + g))
///     >>> closed = 2 * q * (1 - mu) + q * q * (4 / math.pi * mu * math.atan(1 / mu) - 1)
///     >>> round(comnum.theory_ser_fading(16, 20.0) / closed, 10)
///     1.0
///     >>> bool(comnum.theory_ser_fading(16, 20.0, fading="rice", parameter=10.0) < closed)
///     True
#[pyfunction]
#[pyo3(signature = (m, esn0_db, kind = "qam", *, fading = None, parameter = None))]
fn theory_ser_fading(
    m: usize,
    esn0_db: &Bound<'_, PyAny>,
    kind: &str,
    fading: Option<&Bound<'_, PyAny>>,
    parameter: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let k = self::kind(kind)?;
    let f = snr_fading(fading, parameter)?;
    map_real(esn0_db, |e| theory::ser_fading(k, m, f, e).py())
}

/// Average bit error rate of a coherent binary signal over a flat fading channel, with
/// maximal-ratio combining of independent branches of the same mean (M. K. Simon and
/// M.-S. Alouini, "A unified approach to the performance analysis of digital
/// communication over generalized fading channels," Proc. IEEE 86(9), 1998, eqs. (17),
/// (23) and Table 3).
///
/// The conditional BER :math:`Q(\sqrt{2 g \gamma})` at the SNR :math:`\gamma`, averaged
/// by the moment generating function of the SNR over :math:`L` branches:
///
/// .. math::
///
///     P_b = \frac{1}{\pi} \int_0^{\pi/2}
///     \left[M_\gamma\!\left(-\frac{g}{\sin^2 \varphi}\right)\right]^L d\varphi,
///
/// :math:`g = 1` for BPSK and Gray QPSK per bit, :math:`1/2` for orthogonal BFSK. Over
/// Rayleigh fading it is :func:`theory_ber_rayleigh_mrc`.
///
/// Args:
///     ebn0_db (float | array_like): the mean :math:`E_b/N_0` of one branch (dB).
///     g (float): the constant of the binary signal (1 for BPSK).
///     fading (str | Fading, optional): a :class:`Fading`, or ``"rayleigh"`` (the
///         default), ``"rice"``, ``"nakagami"``, ``"hoyt"`` or ``"awgn"``.
///     parameter (float, optional): :math:`K` (linear) for Rice, :math:`m` for Nakagami,
///         :math:`q` for Hoyt.
///     branches (int): the number of branches :math:`L` (at least 1).
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> # BPSK over Nakagami-m with an integer m is the MRC of m Rayleigh branches
///     >>> a = comnum.theory_ber_binary_fading(10.0, fading="nakagami", parameter=3.0)
///     >>> b = comnum.theory_ber_rayleigh_mrc(10.0 - 10 * __import__("math").log10(3), branches=3)
///     >>> round(a / b, 10)
///     1.0
#[pyfunction]
#[pyo3(signature = (ebn0_db, *, g = 1.0, fading = None, parameter = None, branches = 1))]
fn theory_ber_binary_fading(
    ebn0_db: &Bound<'_, PyAny>,
    g: f64,
    fading: Option<&Bound<'_, PyAny>>,
    parameter: Option<f64>,
    branches: usize,
) -> PyResult<Py<PyAny>> {
    let f = snr_fading(fading, parameter)?;
    map_real(ebn0_db, |e| theory::ber_binary_fading(g, f, e, branches).py())
}

/// Average bit error rate of a binary signal detected noncoherently or differentially
/// over a flat fading channel (Simon & Alouini 1998, eq. (53) averaged; eq. (64) over
/// Rayleigh fading).
///
/// The conditional BER :math:`0.5 \exp(-g \gamma)` at the SNR :math:`\gamma`, averaged
/// by the moment generating function of the SNR:
///
/// .. math::
///
///     P_b = 0.5\, M_\gamma(-g),
///
/// :math:`g = 1` for DPSK, :math:`1/2` for noncoherent orthogonal BFSK.
///
/// Args:
///     ebn0_db (float | array_like): the mean :math:`E_b/N_0` (dB).
///     g (float): the constant of the binary signal (1 for DPSK).
///     fading (str | Fading, optional): a :class:`Fading`, or ``"rayleigh"`` (the
///         default), ``"rice"``, ``"nakagami"``, ``"hoyt"`` or ``"awgn"``.
///     parameter (float, optional): :math:`K` (linear) for Rice, :math:`m` for Nakagami,
///         :math:`q` for Hoyt.
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> # DPSK over Rayleigh fading at 10 dB: 1 / (2 (1 + 10))
///     >>> round(comnum.theory_ber_noncoherent_binary_fading(10.0) * 22, 12)
///     1.0
#[pyfunction]
#[pyo3(signature = (ebn0_db, *, g = 1.0, fading = None, parameter = None))]
fn theory_ber_noncoherent_binary_fading(
    ebn0_db: &Bound<'_, PyAny>,
    g: f64,
    fading: Option<&Bound<'_, PyAny>>,
    parameter: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let f = snr_fading(fading, parameter)?;
    map_real(ebn0_db, |e| theory::ber_noncoherent_binary_fading(g, f, e).py())
}

/// Density of the phase error of a carrier in circular Gaussian noise (S. O. Rice 1948;
/// eq. (8) of Ahmed, Thompson and Zeidler, MILCOM 2007).
///
/// The phase error :math:`\theta = \arg r - \varphi` of :math:`r = A e^{j\varphi} + w`
/// at the SNR :math:`\rho = A^2/\mathbb{E}|w|^2` has the density
///
/// .. math::
///
///     p(\theta) = \frac{e^{-\rho}}{2\pi}\left[1 + \sqrt{4\pi\rho}\,\cos\theta\,
///     e^{\rho\cos^2\theta}\,\Phi\!\left(\sqrt{2\rho}\cos\theta\right)\right],
///     \qquad \theta \in (-\pi, \pi],
///
/// :math:`\Phi` the standard normal distribution function. It is uniform at
/// :math:`\rho = 0` and tends to :math:`\mathcal{N}(0, 1/(2\rho))` at high SNR.
///
/// Args:
///     theta (float | array_like): phase error (rad), in :math:`(-\pi, \pi]`.
///     snr_lin (float): :math:`A^2/\mathbb{E}|w|^2` (linear).
///
/// Returns:
///     float | numpy.ndarray: the density at every `theta`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum, math
///     >>> round(comnum.theory_phase_pdf(1.0, 0.0) * 2 * math.pi, 12)
///     1.0
#[pyfunction]
fn theory_phase_pdf(theta: &Bound<'_, PyAny>, snr_lin: f64) -> PyResult<Py<PyAny>> {
    map_real(theta, |t| Ok(comnum::mapping::theory::phase_pdf(t, snr_lin)))
}

/// Theoretical SER over AWGN (exact for square QAM and PAM, a tight bound for PSK).
///
/// With :math:`L = \sqrt{M}` and :math:`E_s/N_0 = \log_2 M \cdot E_b/N_0`:
///
/// .. math::
///
///     \begin{aligned}
///     P_s^{\mathrm{QAM}} &= 1 - (1 - p)^2, \qquad
///     p = 2(1 - 1/L)\, Q\!\left(\sqrt{\frac{3 E_s/N_0}{M - 1}}\right), \\
///     P_s^{\mathrm{PAM}} &= \frac{2(M - 1)}{M}\, Q\!\left(\sqrt{\frac{6 E_s/N_0}{M^2 - 1}}\right), \\
///     P_s^{\mathrm{PSK}} &\approx 2\, Q\!\left(\sqrt{2 E_s/N_0}\, \sin(\pi/M)\right),
///     \end{aligned}
///
/// the last one the nearest-neighbour bound (exact for :math:`M = 2` when halved, tight
/// for :math:`M \ge 4`).
///
/// Args:
///     m (int): order.
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///     kind (str): `"qam"`, `"psk"` or `"pam"`.
///
/// Returns:
///     float | numpy.ndarray: the symbol error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_ser(2, 0.0, "psk"), 6)
///     0.07865
///     >>> bool(comnum.theory_ser(16, 10.0) > comnum.theory_ber(16, 10.0))
///     True
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, kind = "qam"))]
fn theory_ser(m: usize, ebn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
    let k = self::kind(kind)?;
    map_real(ebn0_db, |e| metrics::theory_ser(k, m, e).py())
}

/// Theoretical SER over AWGN as a function of :math:`E_s/N_0`: :func:`theory_ser` with
///
/// .. math::
///
///     E_b/N_0 = E_s/N_0 - 10 \log_{10}(\log_2 M),
///
/// for the links whose SNR is per symbol.
///
/// Args:
///     m (int): order.
///     esn0_db (float | array_like): :math:`E_s/N_0` (dB).
///     kind (str): `"qam"`, `"psk"` or `"pam"`.
///
/// Returns:
///     float | numpy.ndarray: the symbol error probability at every :math:`E_s/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum, math
///     >>> a = comnum.theory_ser_esn0(4, 10.0 + 10 * math.log10(2))
///     >>> round(a / comnum.theory_ser(4, 10.0), 12)
///     1.0
#[pyfunction]
#[pyo3(signature = (m, esn0_db, kind = "qam"))]
fn theory_ser_esn0(m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
    let k = self::kind(kind)?;
    map_real(esn0_db, |e| theory::ser_esn0(k, m, e).py())
}

/// Mutual information of a constellation over AWGN (OptiCommPy `theoryMI`) or Rayleigh
/// fading known at the receiver, by Gauss–Hermite quadrature.
///
/// With the probabilities :math:`p_i` of the points :math:`x_i` and
/// :math:`D_{ij}(w) = (|x_i - x_j + w|^2 - |w|^2)/N_0`:
///
/// .. math::
///
///     \mathrm{MI} = -\sum_i p_i\, \mathbb{E}_w \log_2 \sum_j p_j\, e^{-D_{ij}(w)},
///
/// averaged over the gain under Rayleigh fading.
///
/// Args:
///     m (int): order.
///     snr_db (float | array_like): :math:`E_s/\sigma^2` (dB); PAM and OOK see a real channel, the
///         others a circular complex one, as in OptiCommPy.
///     kind (str): `"qam"`, `"psk"`, `"pam"` or `"ook"`.
///     px (array_like, optional): probabilities of the points.
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the mutual information (bits/symbol) at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_mi(4, 30.0), 6)
///     2.0
#[pyfunction]
#[pyo3(signature = (m, snr_db, kind = "qam", *, px = None, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_mi(
    m: usize,
    snr_db: &Bound<'_, PyAny>,
    kind: &str,
    px: Option<&Bound<'_, PyAny>>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let px = px.map(crate::fvec).transpose()?;
    let (c, f) = (Constellation::new(self::kind(kind)?, m).py()?, self::fading(fading)?);
    map_real(snr_db, |s| theory::constellation_mi(&c, s, px.as_deref(), f, points).py())
}

/// Generalised mutual information of a Gray-labelled constellation over AWGN
/// (OptiCommPy `theoryGMI`) or Rayleigh fading known at the receiver, by Gauss–Hermite
/// quadrature.
///
/// With the bits :math:`B_k` of the label of the transmitted point :math:`X`:
///
/// .. math::
///
///     \mathrm{GMI} = H(X) - \sum_k H(B_k \mid Y),
///
/// the rate of bit-metric decoding; for uniform symbols it is
/// :math:`\sum_k I(B_k; Y)`, OptiCommPy's formula.
///
/// Args:
///     m (int): order.
///     snr_db (float | array_like): :math:`E_s/\sigma^2` (dB), as `theory_mi`.
///     kind (str): `"qam"`, `"psk"`, `"pam"` or `"ook"`.
///     px (array_like, optional): probabilities of the points.
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the GMI (bits/symbol) at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (m, snr_db, kind = "qam", *, px = None, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_gmi(
    m: usize,
    snr_db: &Bound<'_, PyAny>,
    kind: &str,
    px: Option<&Bound<'_, PyAny>>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let px = px.map(crate::fvec).transpose()?;
    let (c, f) = (Constellation::new(self::kind(kind)?, m).py()?, self::fading(fading)?);
    map_real(snr_db, |s| theory::constellation_gmi(&c, s, px.as_deref(), f, points).py())
}

/// The CM capacity (bits per channel use) of a constellation with uniform inputs over
/// AWGN or Rayleigh fading known at the receiver (Caire, Taricco & Biglieri 1998, eq.
/// (10)): :func:`theory_mi` of any constellation, uniform inputs.
///
/// For :math:`M = 2^m` points :math:`\mathcal{X}` and the channel :math:`y = g x + n`:
///
/// .. math::
///
///     \tilde{C} = m - \mathbb{E}\left[\log_2 \frac{\sum_{z \in \mathcal{X}} p(y|z)}{p(y|x)}\right].
///
/// Args:
///     constellation (Constellation): the points (normalised inside).
///     snr_db (float | array_like): :math:`E_s/\sigma^2` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the capacity at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_cm_capacity(comnum.Constellation("qam", 4), 30.0), 6)
///     2.0
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_cm_capacity(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| theory::constellation_mi(&constellation.inner, s, None, f, points).py())
}

/// The BICM capacity (bits per channel use) of a labelled constellation with uniform
/// inputs (Caire, Taricco & Biglieri 1998, eq. (14)): the GMI of the bit-wise decoder,
/// labels as `constellation` gives them (Gray, or `Constellation.set_partitioning`):
/// :func:`theory_gmi` of any labelled constellation, uniform inputs.
///
/// With :math:`\mathcal{X}_b^i` the points whose bit :math:`i` is :math:`b`:
///
/// .. math::
///
///     \hat{C} = m - \sum_{i=1}^{m} \mathbb{E}_{b,y}\left[\log_2
///     \frac{\sum_{z \in \mathcal{X}} p(y|z)}{\sum_{z \in \mathcal{X}_b^i} p(y|z)}\right]
///     \le \tilde{C}
///
/// (eq. (16)), below the CM capacity of :func:`theory_cm_capacity`.
///
/// Args:
///     constellation (Constellation): the labelled points.
///     snr_db (float | array_like): :math:`E_s/\sigma^2` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the capacity at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_bicm_capacity(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| theory::constellation_gmi(&constellation.inner, s, None, f, points).py())
}

/// The CM cutoff rate (bits per channel use) with uniform inputs, in closed form
/// (Caire, Taricco & Biglieri 1998, eq. (17)).
///
/// For :math:`M = 2^m` points :math:`\mathcal{X}` of unit mean energy, the noise
/// variance :math:`\sigma^2` and the gain :math:`g`:
///
/// .. math::
///
///     \tilde{R}_0 = m - \log_2 \frac{1}{M} \sum_{x, z \in \mathcal{X}}
///     \mathbb{E}_g\, e^{-|g|^2 |x - z|^2 / (4\sigma^2)},
///
/// where the mean over Rayleigh fading is :math:`1 / (1 + |x - z|^2 / (4\sigma^2))`.
///
/// Args:
///     constellation (Constellation): the points.
///     snr_db (float | array_like): :math:`E_s/\sigma^2` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///
/// Returns:
///     float | numpy.ndarray: the cutoff rate at every SNR.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comnum, math
///     >>> r0 = comnum.theory_cm_cutoff_rate(comnum.Constellation("psk", 2), 0.0)
///     >>> round(r0 - (1 - math.log2(1 + math.exp(-1))), 12)
///     0.0
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn"))]
fn theory_cm_cutoff_rate(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| Ok(bicm::cm_cutoff_rate(&constellation.inner, s, f)))
}

/// The BICM cutoff rate (bits per channel use) of a labelled constellation (Caire,
/// Taricco & Biglieri 1998, eqs. (19), (21)).
///
/// With the Bhattacharyya factor :math:`B` and :math:`\mathcal{X}_b^i` the points whose
/// bit :math:`i` is :math:`b`:
///
/// .. math::
///
///     \hat{R}_0 = m\left(1 - \log_2(1 + B)\right), \qquad
///     B = \frac{1}{m} \sum_i \mathbb{E}_{b,y} \sqrt{\frac{\sum_{z \in \mathcal{X}_{\bar{b}}^i}
///     p(y|z)}{\sum_{z \in \mathcal{X}_b^i} p(y|z)}}.
///
/// Args:
///     constellation (Constellation): the labelled points.
///     snr_db (float | array_like): :math:`E_s/\sigma^2` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the cutoff rate at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_bicm_cutoff_rate(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| bicm::bicm_cutoff_rate(&constellation.inner, s, f, points).py())
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theory_ber, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ser, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ser_esn0, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_gray, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_rayleigh_mrc, m)?)?;
    m.add_class::<PyFading>()?;
    m.add_function(wrap_pyfunction!(theory_ser_fading, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_binary_fading, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_noncoherent_binary_fading, m)?)?;
    m.add_function(wrap_pyfunction!(theory_phase_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(theory_mi, m)?)?;
    m.add_function(wrap_pyfunction!(theory_gmi, m)?)?;
    m.add_function(wrap_pyfunction!(theory_cm_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(theory_bicm_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(theory_cm_cutoff_rate, m)?)?;
    m.add_function(wrap_pyfunction!(theory_bicm_cutoff_rate, m)?)?;
    Ok(())
}
