//! Constant-envelope OFDM: the modulator, its receivers and its closed forms.

use super::*;

/// A constant-envelope OFDM (CE-OFDM) modulator and phase demodulator.
///
/// After [1]_.
///
/// A block carries :math:`K` data symbols :math:`d` (unit mean energy) on the subcarriers
/// :math:`1, \dots, K` of a DFT of size :math:`N = 2J(K+1)`, :math:`J` the oversampling
/// factor; the real OFDM signal :math:`x` and the samples :math:`s_n` are
///
/// .. math::
///
///     \begin{aligned}
///     x &= (F d + F^* d^*)/\sqrt{2K}, \\
///     s_n &= \exp(j 2\pi h x_n),
///     \end{aligned}
///
/// with :math:`F \in \mathbb{C}^{N \times K}`, :math:`F_{nk} = e^{j2\pi kn/N}`, so that
/// :math:`x` has a unit variance and the phase has the standard deviation
/// :math:`2\pi h`, :math:`h` being the modulation index (Sec. II). An optional cyclic
/// prefix of :math:`N_{cp}` samples is prepended. The receiver (Sec. III, Fig. 1) takes
/// the phase :math:`\psi` of the samples, unwrapped along the stream or not, removes the
/// prefix and projects it on the data bins:
///
/// .. math::
///
///     \hat{d} = \frac{G \psi}{2\pi h}, \qquad G = \frac{\sqrt{2K}}{N} F^{H}.
///
/// **SNR convention.** :math:`E_s/N_0` is per data symbol: a block of energy
/// :math:`N + N_{cp}` carries :math:`K` symbols, so the noise variance per sample is
/// :math:`\sigma^2 = (N + N_{cp})/(K E_s/N_0)` (:meth:`noise_variance`;
/// :math:`\mathrm{sps} = (N + N_{cp})/K` samples per data symbol). Above the threshold of
/// the phase demodulator (a carrier-to-noise ratio near 10 dB, Sec. III-B), every data
/// subcarrier sees the SNR :math:`(2\pi h)^2 (E_s/N_0) N/(N + N_{cp})` (:meth:`snr`), and
/// the error rates are those of the constellation over AWGN at that SNR (:meth:`ser_esn0`,
/// :meth:`ber_esn0`). For :math:`M`-PAM on every real subcarrier this is eq. (15) of
/// [1]_, the nearest-neighbour term of the Gray BER:
///
/// .. math::
///
///     \mathrm{BER} = \frac{2(M-1)}{M \log_2 M}\,
///     Q\!\left(2\pi h \sqrt{\frac{6 \log_2 M}{M^2 - 1} \frac{E_b}{N_0}}\right);
///
/// :meth:`ber_esn0` uses the exact Gray BER of the square QAM of order :math:`M^2` whose
/// real and imaginary parts are the two PAM symbols (within 1 % of eq. (15) once the BER
/// is under 1e-2).
///
/// Args:
///     n_subcarriers (int): data subcarriers :math:`K` per block.
///     h (float): modulation index (phase standard deviation :math:`2\pi h`).
///     oversampling (int): :math:`J`, the DFT has :math:`N = 2J(K+1)` points.
///     n_cp (int): cyclic prefix (samples).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] S. C. Thompson, A. U. Ahmed, J. G. Proakis, J. R. Zeidler and M. J. Geile,
///        "Constant envelope OFDM," *IEEE Trans. Commun.*, 56(8), 1300–1312, 2008.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> ce = comnum.CeOfdm(16, h=0.1, oversampling=2)
///     >>> c = comnum.Constellation("qam", 16).normalized()
///     >>> _, d = comnum.symbols(c, 64, comnum.Rng(1))
///     >>> s = ce.modulate(d)
///     >>> s.size == 4 * ce.n_fft and np.allclose(np.abs(s), 1.0)
///     True
///     >>> np.allclose(ce.demodulate(s), d)
///     True
#[pyclass(name = "CeOfdm", module = "comnum")]
pub struct PyCeOfdm {
    inner: CeOfdm,
}

fn unwrap(name: &str) -> PyResult<Unwrap> {
    name.parse::<Unwrap>().py()
}

/// The receiver of a name (`arctangent`, `linear`, `enhanced`), the arctangent one with
/// the given unwrapping.
fn receiver(name: &str, unwrap: &str) -> PyResult<Receiver> {
    Ok(match name.parse::<Receiver>().py()? {
        Receiver::Arctangent(_) => Receiver::Arctangent(self::unwrap(unwrap)?),
        other => other,
    })
}

#[pymethods]
impl PyCeOfdm {
    #[new]
    #[pyo3(signature = (n_subcarriers, h, *, oversampling=4, n_cp=0))]
    fn new(n_subcarriers: usize, h: f64, oversampling: usize, n_cp: usize) -> PyResult<Self> {
        let inner = CeOfdm { n_cp, ..CeOfdm::new(n_subcarriers, oversampling, h).py()? };
        inner.check().py()?;
        Ok(PyCeOfdm { inner })
    }

    /// int: Data subcarriers :math:`K` per block.
    #[getter]
    fn n_subcarriers(&self) -> usize {
        self.inner.n_subcarriers
    }

    /// float: The modulation index :math:`h`.
    #[getter]
    fn h(&self) -> f64 {
        self.inner.h
    }

    /// int: The oversampling factor :math:`J`.
    #[getter]
    fn oversampling(&self) -> usize {
        self.inner.oversampling
    }

    /// int: The cyclic prefix :math:`N_{cp}` (samples).
    #[getter]
    fn n_cp(&self) -> usize {
        self.inner.n_cp
    }

    /// int: The DFT size :math:`N = 2J(K+1)`.
    #[getter]
    fn n_fft(&self) -> usize {
        self.inner.n_fft()
    }

    /// int: Samples per block, :math:`N + N_{cp}`.
    #[getter]
    fn block_len(&self) -> usize {
        self.inner.block_len()
    }

    /// float: Samples per data symbol, :math:`(N + N_{cp})/K`.
    #[getter]
    fn sps(&self) -> f64 {
        self.inner.sps()
    }

    /// The noise variance per sample for an :math:`E_s/N_0` per data symbol and a unit
    /// envelope:
    ///
    /// .. math::
    ///
    ///     \sigma^2 = (N + N_{cp})/(K E_s/N_0).
    ///
    /// Args:
    ///     esn0_db (float | array_like): :math:`E_s/N_0` (dB) per data symbol.
    ///
    /// Returns:
    ///     float | numpy.ndarray: :math:`\sigma^2` (for `Rng.complex_noise`).
    ///
    /// Raises:
    ///     ValueError: for values that are not real numbers.
    fn noise_variance(&self, esn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        map_real(esn0_db, |e| Ok(self.inner.noise_variance(e)))
    }

    /// The real OFDM signal of every block (no prefix):
    ///
    /// .. math::
    ///
    ///     x = (F d + F^* d^*)/\sqrt{2K},
    ///
    /// with :math:`F_{nk} = e^{j2\pi kn/N}` on the data subcarriers :math:`k = 1, \dots, K`.
    ///
    /// Args:
    ///     symbols (array_like): a multiple of :math:`K` data symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`N` real samples per block.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn ofdm_signal<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.ofdm_signal(&cvec(symbols)?).py()?))
    }

    /// The transmitted samples of every block, with the cyclic prefix.
    ///
    /// The samples are
    ///
    /// .. math::
    ///
    ///     s_n = \exp(j 2\pi h x_n),
    ///
    /// :math:`x` the real OFDM signal (:meth:`ofdm_signal`).
    ///
    /// Args:
    ///     symbols (array_like): a multiple of :math:`K` data symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`N + N_{cp}` unit-envelope samples per block.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&cvec(symbols)?).py()?))
    }

    /// The symbol estimates of every block.
    ///
    /// The arctangent receiver (Thompson et al. 2008) projects the phase :math:`\psi` of
    /// the block, prefix removed, on the data bins:
    ///
    /// .. math::
    ///
    ///     \hat{d} = \frac{G \psi}{2\pi h}, \qquad G = \frac{\sqrt{2K}}{N} F^{H};
    ///
    /// the low-complexity receivers of Ahmed and Zeidler (2015) take
    /// :math:`\operatorname{Im} r` (linear) or
    /// :math:`\operatorname{Im} r\,(2 - \operatorname{Re} r)` (enhanced) instead of the
    /// phase, divided by their Bussgang gain.
    ///
    /// Args:
    ///     samples (array_like): a multiple of :math:`N + N_{cp}` received samples.
    ///     unwrap (str): for the arctangent receiver, `"none"` (the principal phase, wraps
    ///         are errors) or `"stream"` (the phase unwrapped along the stream, the phase
    ///         unwrapper of the paper).
    ///     receiver (str): `"arctangent"` (the phase, Thompson et al.
    ///         2008), `"linear"` (:math:`\operatorname{Im} r`) or `"enhanced"`
    ///         (:math:`\operatorname{Im} r\,(2 - \operatorname{Re} r)`), the low-complexity
    ///         receivers of Ahmed and Zeidler (2015), divided by their Bussgang gain.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`K` complex estimates per block.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (samples, *, unwrap="none", receiver="arctangent"))]
    fn demodulate<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
        unwrap: &str,
        receiver: &str,
    ) -> PyResult<CArr<'py>> {
        let r = self::receiver(receiver, unwrap)?;
        Ok(carr(py, self.inner.demodulate_with_receiver(&cvec(samples)?, r).py()?))
    }

    /// The posterior means of the symbols of every block by the GAMP receiver of López
    /// Morales, Dinis and García Armada (2022), near the optimum where the phase leaves
    /// :math:`(-\pi, \pi]`.
    ///
    /// Args:
    ///     samples (array_like): a multiple of :math:`N + N_{cp}` received samples.
    ///     noise_var (float): :math:`\sigma^2` per sample (:meth:`noise_variance`).
    ///     constellation (Constellation): the symbols as transmitted (square QAM, PAM or
    ///         BPSK).
    ///     iterations (int): iterations per block.
    ///     damping (float): damping factor in :math:`(0, 1]`.
    ///     points (int): points of the quadrature of the output step.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`K` complex estimates per block.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (samples, noise_var, constellation, *, iterations=50, damping=0.5, points=64))]
    #[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python method")]
    fn gamp<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
        noise_var: f64,
        constellation: &PyConstellation,
        iterations: usize,
        damping: f64,
        points: usize,
    ) -> PyResult<CArr<'py>> {
        let g = Gamp { iterations, damping, points };
        Ok(carr(py, g.detect(&self.inner, &cvec(samples)?, &constellation.inner, noise_var).py()?))
    }

    /// The phase of the receiver before the DFT, prefix included.
    ///
    /// Args:
    ///     samples (array_like): received samples.
    ///     unwrap (str): `"none"` or `"stream"`, as :meth:`demodulate`.
    ///
    /// Returns:
    ///     numpy.ndarray: one phase (rad) per sample.
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    #[pyo3(signature = (samples, *, unwrap="none"))]
    fn received_phase<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>, unwrap: &str) -> PyResult<FArr<'py>> {
        let u = self::unwrap(unwrap)?;
        Ok(farr(py, self.inner.received_phase(&cvec(samples)?, u)))
    }

    /// The SNR of every data subcarrier in the linear regime (linear scale):
    ///
    /// .. math::
    ///
    ///     \mathrm{SNR} = (2\pi h)^2 \frac{E_s}{N_0} \frac{N}{N + N_{cp}}.
    ///
    /// Args:
    ///     esn0_db (float | array_like): :math:`E_s/N_0` (dB) per data symbol.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the SNR (linear).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn snr(&self, esn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        map_real(esn0_db, |e| theory::snr(&self.inner, e).py())
    }

    /// The SER of the linear regime: that of a constellation of `m` points over AWGN
    /// at :meth:`snr` (:func:`theory_ser`).
    ///
    /// Args:
    ///     m (int): constellation order.
    ///     esn0_db (float | array_like): :math:`E_s/N_0` (dB) per data symbol.
    ///     kind (str): `"qam"`, `"psk"` or `"pam"`.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the SER.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (m, esn0_db, kind="qam"))]
    fn ser_esn0(&self, m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
        let k = self::kind(kind)?;
        map_real(esn0_db, |e| theory::ser_esn0(&self.inner, k, m, e).py())
    }

    /// The BER of the linear regime, Gray mapping: the exact Gray BER of a square QAM
    /// (eq. (15) of Thompson et al. with its further neighbours), else
    /// :func:`theory_ber`.
    ///
    /// Args:
    ///     m (int): constellation order.
    ///     esn0_db (float | array_like): :math:`E_s/N_0` (dB) per data symbol.
    ///     kind (str): `"qam"`, `"psk"` or `"pam"`.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the BER.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (m, esn0_db, kind="qam"))]
    fn ber_esn0(&self, m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
        let k = self::kind(kind)?;
        map_real(esn0_db, |e| theory::ber_esn0(&self.inner, k, m, e).py())
    }

    /// The expected periodogram :math:`\mathbb{E}|S_k|^2/N` of a transmitted block under the
    /// Gaussian model (Guerreiro et al. 2015): the bins average to 1, bin 0 holds the carrier
    /// line.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`N` values, in the order of the DFT.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn psd<'py>(&self, py: Python<'py>) -> PyResult<FArr<'py>> {
        Ok(farr(py, theory::psd(&self.inner).py()?))
    }

    /// The SNR of every data subcarrier with the receiver: flat for the arctangent receiver
    /// above its threshold, Bussgang's analysis for the linear receivers.
    ///
    /// Args:
    ///     esn0_db (float): :math:`E_s/N_0` (dB) per data symbol.
    ///     receiver (str): `"arctangent"`, `"linear"` or `"enhanced"`.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`K` SNRs (linear).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    #[pyo3(signature = (esn0_db, *, receiver="arctangent"))]
    fn subcarrier_snr<'py>(&self, py: Python<'py>, esn0_db: f64, receiver: &str) -> PyResult<FArr<'py>> {
        let r = self::receiver(receiver, "none")?;
        Ok(farr(py, theory::subcarrier_snr(&CeOfdm { receiver: r, ..self.inner.clone() }, esn0_db).py()?))
    }

    /// The BER of a Gray-mapped constellation with the receiver, averaged over the
    /// subcarriers of :meth:`subcarrier_snr`.
    ///
    /// Args:
    ///     m (int): constellation order.
    ///     esn0_db (float | array_like): :math:`E_s/N_0` (dB) per data symbol.
    ///     kind (str): `"qam"`, `"psk"` or `"pam"`.
    ///     receiver (str): `"arctangent"`, `"linear"` or `"enhanced"`.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the BER.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (m, esn0_db, kind="qam", *, receiver="arctangent"))]
    fn ber_receiver(&self, m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str, receiver: &str) -> PyResult<Py<PyAny>> {
        let (k, r) = (self::kind(kind)?, self::receiver(receiver, "none")?);
        let ce = CeOfdm { receiver: r, ..self.inner.clone() };
        map_real(esn0_db, |e| theory::ber_receiver(&ce, k, m, e).py())
    }

    /// The change of the estimates of the arctangent receiver caused by a cycle slip of
    /// :math:`+2\pi` from sample `start` (after the prefix) to the end of the block (Ahmed et al.
    /// 2012, eqs. (11)–(14)).
    ///
    /// Args:
    ///     start (int): first sample of the slip, :math:`0 \le \mathrm{start} < N`.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`K` complex changes.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn cycle_slip_response<'py>(&self, py: Python<'py>, start: usize) -> PyResult<CArr<'py>> {
        Ok(carr(py, theory::cycle_slip_response(&self.inner, start).py()?))
    }

    fn __repr__(&self) -> String {
        let c = &self.inner;
        format!(
            "CeOfdm(n_subcarriers={}, h={}, oversampling={}, n_cp={})",
            c.n_subcarriers, c.h, c.oversampling, c.n_cp
        )
    }
}

/// The powers of the intermodulation products of a phase modulator driven by a Gaussian
/// signal.
///
/// For a Gaussian signal of standard deviation :math:`\phi = 2\pi h` ([1]_,
/// eqs. (12)–(14); [2]_, eq. (2)):
///
/// .. math::
///
///     P_i = e^{-\phi^2} \frac{\phi^{2i}}{i!},
///
/// the carrier line, the useful term, then the products of order :math:`i`.
///
/// Args:
///     two_pi_h (float): :math:`2\pi h`.
///     n (int): number of powers.
///
/// Returns:
///     numpy.ndarray: :math:`P_0 \dots P_{n-1}`.
///
/// Raises:
///     ValueError: for a negative count.
///
/// References:
///     .. [1] J. Guerreiro, R. Dinis and P. Montezuma, "Optimum performance and spectral
///        characterization of CE-OFDM signals," *Proc. IEEE VTC Fall*, 2015.
///     .. [2] R. A. Pacheco and D. Hatzinakos, "Error rate analysis of phase-modulated OFDM
///        (OFDM-PM) in AWGN channels," *Proc. IEEE ICASSP*, 2006.
///
/// Examples:
///     >>> import comnum, math
///     >>> round(float(comnum.ce_ofdm_imp_powers(1.0, 3)[0]), 6) == round(math.exp(-1), 6)
///     True
#[pyfunction]
fn ce_ofdm_imp_powers<'py>(py: Python<'py>, two_pi_h: f64, n: i64) -> PyResult<FArr<'py>> {
    Ok(farr(py, theory::imp_powers(two_pi_h, count(n, "n")?)))
}

/// The optimum asymptotic gain :math:`G` of CE-OFDM over linear OFDM in AWGN,
/// :math:`(2\pi h)^2`, or with the phase clipped to :math:`\pm\theta_{\max}`:
///
/// .. math::
///
///     G = (2\pi h)^2 \operatorname{erf}\!\left(\frac{\theta_{\max}}{\sqrt{2} \cdot 2\pi h}\right).
///
/// These are eq. (20) of [1]_ and eq. (7) of [2]_.
///
/// Args:
///     two_pi_h (float): :math:`2\pi h`.
///     theta_max (float | None): phase clipping (rad).
///
/// Returns:
///     float: :math:`G` (linear).
///
/// References:
///     .. [1] J. Guerreiro, R. Dinis and P. Montezuma, "Optimum performance and spectral
///        characterization of CE-OFDM signals," *Proc. IEEE VTC Fall*, 2015.
///     .. [2] J. Guerreiro, R. Dinis and P. Montezuma, "On the detection of CE-OFDM
///        signals," *IEEE Commun. Lett.*, 20(11), 2165–2168, 2016.
///
/// Examples:
///     >>> import comnum, math
///     >>> round(10 * math.log10(comnum.ce_ofdm_optimum_gain(1.5)), 2)
///     3.52
#[pyfunction]
#[pyo3(signature = (two_pi_h, theta_max=None))]
fn ce_ofdm_optimum_gain(two_pi_h: f64, theta_max: Option<f64>) -> f64 {
    theory::optimum_gain(two_pi_h, theta_max)
}

/// The optimum asymptotic BER of QPSK CE-OFDM in AWGN:
///
/// .. math::
///
///     P_b = Q\left(\sqrt{2 G E_b/N_0}\right).
///
/// This is eq. (8) of [1]_.
///
/// Args:
///     two_pi_h (float): :math:`2\pi h`.
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///
/// Returns:
///     float | numpy.ndarray: the BER.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// References:
///     .. [1] J. Guerreiro, R. Dinis and P. Montezuma, "On the detection of CE-OFDM
///        signals," *IEEE Commun. Lett.*, 20(11), 2165–2168, 2016.
#[pyfunction]
fn theory_ce_ofdm_ber_optimum(two_pi_h: f64, ebn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(theory::ber_optimum(two_pi_h, e)))
}

/// The occupied bandwidth :math:`B_y/(2B_m)` of a carrier phase-modulated by a Gaussian message
/// of standard deviation `sigma` with a flat spectrum, the carrier line removed.
///
/// The band follows [1]_ (eqs. (2)–(5), Fig. 1).
///
/// Args:
///     sigma (float): standard deviation of the phase (rad).
///     fraction (float): fraction of the power in the band.
///
/// Returns:
///     float: :math:`B_y/(2B_m)`.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] R. A. Pacheco and D. Hatzinakos, "Error rate analysis of phase-modulated OFDM
///        (OFDM-PM) in AWGN channels," *Proc. IEEE ICASSP*, 2006.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.pm_occupied_bandwidth(1.1), 1)
///     2.0
#[pyfunction]
#[pyo3(signature = (sigma, fraction=0.99))]
fn pm_occupied_bandwidth(sigma: f64, fraction: f64) -> PyResult<f64> {
    theory::occupied_bandwidth(sigma, fraction).py()
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCeOfdm>()?;
    m.add_function(wrap_pyfunction!(ce_ofdm_imp_powers, m)?)?;
    m.add_function(wrap_pyfunction!(ce_ofdm_optimum_gain, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ce_ofdm_ber_optimum, m)?)?;
    m.add_function(wrap_pyfunction!(pm_occupied_bandwidth, m)?)?;
    Ok(())
}
