//! Closed forms of direct-detection receivers (on–off keying): the Q factor, the
//! sensitivity of p-i-n and avalanche photodiode receivers, the quantum limit and the
//! power penalties, after Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., 2002.

use crate::{IntoPy, map_real};
use comnum::hardware::optical::direct_detection::{self as dd, DirectDetectionReceiver};
use num_complex::Complex64 as C64;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A direct-detection receiver of on–off keying: a p-i-n or avalanche photodiode, its
/// load and amplifier, in the Gaussian noise model. After [1]_, Secs. 4.4–4.5 (eqs. (4.4.9),
/// (4.4.17), (4.4.18), (4.5.10), (4.5.11), (4.5.17), (4.5.19)).
///
/// The receiver decides between the photocurrents :math:`I_1 = M R P_1` and :math:`I_0 =
/// M R P_0` of the 1 and 0 bits, each with the shot noise of its power and the thermal
/// noise of the load,
///
/// .. math::
///
///     \sigma_s^2 = 2 q M^2 F_A(M) R P \Delta f, \qquad
///     \sigma_T^2 = \frac{4 k_B T}{R_L} F_n \Delta f, \qquad
///     F_A(M) = k_A M + (1 - k_A)\left(2 - \frac{1}{M}\right),
///
/// with :math:`R` the responsivity, :math:`M` the avalanche gain (1 for a p-i-n),
/// :math:`k_A` the ionisation-coefficient ratio, :math:`F_n` the noise figure of the
/// amplifier and :math:`\Delta f` the noise bandwidth. The Q factor and the bit error
/// rate at the optimum threshold are
///
/// .. math::
///
///     Q = \frac{I_1 - I_0}{\sigma_1 + \sigma_0}, \qquad
///     \mathrm{BER} = \frac{1}{2}\operatorname{erfc}\left(\frac{Q}{\sqrt{2}}\right),
///
/// and the sensitivity, the average power of on–off keying with no power in the 0 bits
/// that gives the Q factor :math:`Q`, is
///
/// .. math::
///
///     \bar{P}_{\mathrm{rec}} = \frac{Q}{R}\left(q F_A \Delta f\, Q + \frac{\sigma_T}{M}\right),
///
/// minimised over :math:`M` at :math:`M_{\mathrm{opt}}^2 = [\sigma_T/(q \Delta f Q) -
/// (1 - k_A)]/k_A`.
///
/// Args:
///     responsivity (float): :math:`R` (A/W), before the avalanche gain.
///     apd_gain (float): :math:`M` (1 for a p-i-n photodiode).
///     ionization_ratio (float): :math:`k_A \in [0, 1]`.
///     temperature_k (float): :math:`T` (K).
///     load_ohm (float): :math:`R_L` (Ω).
///     noise_figure_db (float): :math:`F_n` (dB).
///     bandwidth_hz (float): :math:`\Delta f` (Hz), about half the bit rate.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it), when a method is
///         called.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> rx = comnum.DirectDetectionReceiver(responsivity=1.0, bandwidth_hz=0.5e9)
///     >>> round(rx.sensitivity_dbm(6.0), 1)   # Q = 6, a BER of 1e-9, at 1 Gb/s
///     -24.6
///     >>> round(rx.q_factor(2 * rx.sensitivity_w(6.0), 0.0), 9)
///     6.0
#[pyclass(name = "DirectDetectionReceiver", module = "comnum")]
#[derive(Clone)]
pub struct PyDirectDetectionReceiver {
    inner: DirectDetectionReceiver,
}

wrapper_methods! {
    PyDirectDetectionReceiver {
        /// float: Responsivity (A/W) (read and write).
        responsivity / set_responsivity: f64,
        /// float: Avalanche gain :math:`M` (read and write).
        apd_gain / set_apd_gain: f64,
        /// float: Ionisation-coefficient ratio :math:`k_A` (read and write).
        ionization_ratio / set_ionization_ratio: f64,
        /// float: Temperature (K) (read and write).
        temperature_k / set_temperature_k: f64,
        /// float: Load resistance (Ω) (read and write).
        load_ohm / set_load_ohm: f64,
        /// float: Noise figure of the amplifier (dB) (read and write).
        noise_figure_db / set_noise_figure_db: f64,
        /// float: Noise bandwidth :math:`\Delta f` (Hz) (read and write).
        bandwidth_hz / set_bandwidth_hz: f64,
    }

    #[new]
    #[pyo3(signature = (
        *,
        responsivity = DirectDetectionReceiver::default().responsivity,
        apd_gain = DirectDetectionReceiver::default().apd_gain,
        ionization_ratio = DirectDetectionReceiver::default().ionization_ratio,
        temperature_k = DirectDetectionReceiver::default().temperature_k,
        load_ohm = DirectDetectionReceiver::default().load_ohm,
        noise_figure_db = DirectDetectionReceiver::default().noise_figure_db,
        bandwidth_hz = DirectDetectionReceiver::default().bandwidth_hz,
    ))]
    #[allow(clippy::too_many_arguments, reason = "one keyword argument per field of the receiver")]
    fn new(
        responsivity: f64,
        apd_gain: f64,
        ionization_ratio: f64,
        temperature_k: f64,
        load_ohm: f64,
        noise_figure_db: f64,
        bandwidth_hz: f64,
    ) -> Self {
        PyDirectDetectionReceiver {
            inner: DirectDetectionReceiver {
                responsivity,
                apd_gain,
                ionization_ratio,
                temperature_k,
                load_ohm,
                noise_figure_db,
                bandwidth_hz,
            },
        }
    }

    /// The excess noise factor :math:`F_A(M)` of the photodiode.
    ///
    /// Returns:
    ///     float: :math:`F_A`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn excess_noise_factor(&self) -> PyResult<f64> {
        self.inner.excess_noise_factor().py()
    }

    /// The variance of the thermal noise :math:`\sigma_T^2 = 4 k_B T F_n \Delta f / R_L`.
    ///
    /// Returns:
    ///     float: the variance (A²).
    fn thermal_variance(&self) -> f64 {
        self.inner.thermal_variance()
    }

    /// The variance of the shot noise :math:`\sigma_s^2 = 2 q M^2 F_A R P \Delta f` of
    /// an optical power.
    ///
    /// Args:
    ///     power_w (float): the power :math:`P` (W).
    ///
    /// Returns:
    ///     float: the variance (A²).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn shot_variance(&self, power_w: f64) -> PyResult<f64> {
        self.inner.shot_variance(power_w).py()
    }

    /// The Q factor of the powers of the 1 and 0 bits.
    ///
    /// Args:
    ///     p1_w (float): the power of the 1 bits (W).
    ///     p0_w (float): the power of the 0 bits (W).
    ///
    /// Returns:
    ///     float: :math:`Q`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or a negative power (the message names it).
    fn q_factor(&self, p1_w: f64, p0_w: f64) -> PyResult<f64> {
        self.inner.q_factor(p1_w, p0_w).py()
    }

    /// The bit error rate :math:`\frac{1}{2}\operatorname{erfc}(Q/\sqrt{2})` of the
    /// powers of the 1 and 0 bits at the optimum threshold.
    ///
    /// Args:
    ///     p1_w (float): the power of the 1 bits (W).
    ///     p0_w (float): the power of the 0 bits (W).
    ///
    /// Returns:
    ///     float: the BER.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or a negative power (the message names it).
    fn ber(&self, p1_w: f64, p0_w: f64) -> PyResult<f64> {
        self.inner.ber(p1_w, p0_w).py()
    }

    /// The receiver sensitivity: the average power of on–off keying with no power in
    /// the 0 bits that gives a Q factor.
    ///
    /// Args:
    ///     q (float): the Q factor (6 for a BER of 1e-9).
    ///
    /// Returns:
    ///     float: the average power (W).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or a non-positive Q factor.
    fn sensitivity_w(&self, q: f64) -> PyResult<f64> {
        self.inner.sensitivity_w(q).py()
    }

    /// The receiver sensitivity in dBm (``sensitivity_w``).
    ///
    /// Args:
    ///     q (float): the Q factor.
    ///
    /// Returns:
    ///     float: the average power (dBm).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or a non-positive Q factor.
    fn sensitivity_dbm(&self, q: f64) -> PyResult<f64> {
        self.inner.sensitivity_dbm(q).py()
    }

    /// The avalanche gain that minimises the sensitivity at a Q factor,
    /// :math:`M_{\mathrm{opt}}^2 = [\sigma_T/(q \Delta f Q) - (1 - k_A)]/k_A`, at least 1.
    ///
    /// Args:
    ///     q (float): the Q factor.
    ///
    /// Returns:
    ///     float: :math:`M_{\mathrm{opt}}`.
    ///
    /// Raises:
    ///     ValueError: for an ionisation ratio of 0 (no finite optimum).
    fn optimum_apd_gain(&self, q: f64) -> PyResult<f64> {
        self.inner.optimum_apd_gain(q).py()
    }

    fn __repr__(&self) -> String {
        let d = &self.inner;
        format!(
            "comnum.DirectDetectionReceiver(responsivity={}, apd_gain={}, ionization_ratio={}, temperature_k={}, \
             load_ohm={}, noise_figure_db={}, bandwidth_hz={:e})",
            d.responsivity, d.apd_gain, d.ionization_ratio, d.temperature_k, d.load_ohm, d.noise_figure_db, d.bandwidth_hz
        )
    }

    fn __getnewargs_ex__<'py>(&self, py: Python<'py>) -> PyResult<((), Bound<'py, PyDict>)> {
        let d = &self.inner;
        let kw = PyDict::new(py);
        kw.set_item("responsivity", d.responsivity)?;
        kw.set_item("apd_gain", d.apd_gain)?;
        kw.set_item("ionization_ratio", d.ionization_ratio)?;
        kw.set_item("temperature_k", d.temperature_k)?;
        kw.set_item("load_ohm", d.load_ohm)?;
        kw.set_item("noise_figure_db", d.noise_figure_db)?;
        kw.set_item("bandwidth_hz", d.bandwidth_hz)?;
        Ok(((), kw))
    }
}

/// The Q factor of two Gaussian levels. After [1]_, eqs. (4.5.10), (4.5.11):
///
/// .. math::
///
///     Q = \frac{I_1 - I_0}{\sigma_1 + \sigma_0}.
///
/// Args:
///     i1 (float): the level of the 1 bits.
///     i0 (float): the level of the 0 bits.
///     sigma1 (float): the noise standard deviation of the 1 bits.
///     sigma0 (float): the noise standard deviation of the 0 bits.
///
/// Returns:
///     float: :math:`Q`; ``comnum.qfunc(Q)`` is the bit error rate at the optimum
///     threshold and ``comnum.qfuncinv(ber)`` the Q factor of a bit error rate.
///
/// Raises:
///     ValueError: for negative deviations, or both zero.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.q_factor(1.0, 0.0, 0.1, 0.1)
///     5.0
#[pyfunction]
fn q_factor(i1: f64, i0: f64, sigma1: f64, sigma0: f64) -> PyResult<f64> {
    dd::q_factor(i1, i0, sigma1, sigma0).py()
}

/// The excess noise factor of an avalanche photodiode. After [1]_, eq. (4.4.18):
///
/// .. math::
///
///     F_A(M) = k_A M + (1 - k_A)\left(2 - \frac{1}{M}\right).
///
/// Args:
///     m (float): the avalanche gain :math:`M \ge 1`.
///     k_a (float): the ionisation-coefficient ratio :math:`k_A \in [0, 1]`.
///
/// Returns:
///     float: :math:`F_A`.
///
/// Raises:
///     ValueError: for :math:`M < 1` or :math:`k_A` outside :math:`[0, 1]`.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.excess_noise_factor(10.0, 0.0), 6)
///     1.9
#[pyfunction]
fn excess_noise_factor(m: f64, k_a: f64) -> PyResult<f64> {
    dd::excess_noise_factor(m, k_a).py()
}

/// The average number of photons per bit of the quantum limit of on–off keying. After
/// [1]_, eq. (4.5.24): a receiver that counts photons without noise errs only when a 1 bit
/// carries no photon,
///
/// .. math::
///
///     \mathrm{BER} = \frac{1}{2} e^{-N_p}, \qquad \bar{N}_p = \frac{N_p}{2} =
///     -\frac{\ln(2\,\mathrm{BER})}{2},
///
/// 10 photons per bit for a BER of :math:`10^{-9}`.
///
/// Args:
///     ber (float | array_like): the bit error rate, in :math:`(0, 1/2)`.
///
/// Returns:
///     float | numpy.ndarray: :math:`\bar{N}_p`.
///
/// Raises:
///     ValueError: for a BER outside :math:`(0, 1/2)`.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.quantum_limit_photons_per_bit(1e-9), 1)
///     10.0
#[pyfunction]
fn quantum_limit_photons_per_bit(ber: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ber, |v| dd::quantum_limit_photons_per_bit(v).py())
}

/// The quantum limit of the receiver sensitivity. After [1]_, eq. (4.5.25):
///
/// .. math::
///
///     \bar{P}_{\mathrm{rec}} = \bar{N}_p h \nu B,
///
/// with :math:`\bar{N}_p` the photons per bit of ``quantum_limit_photons_per_bit``.
///
/// Args:
///     ber (float): the bit error rate.
///     bit_rate_hz (float): the bit rate :math:`B` (bit/s).
///     frequency_hz (float): the optical frequency :math:`\nu` (Hz), 193.4e12 at 1550 nm.
///
/// Returns:
///     float: the average power (W).
///
/// Raises:
///     ValueError: for a BER outside :math:`(0, 1/2)`, or a non-positive rate or
///         frequency.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> p = comnum.quantum_limit_w(1e-9, 10e9, 193.4e12)
///     >>> round(comnum.w2dbm(p), 1)
///     -48.9
#[pyfunction]
fn quantum_limit_w(ber: f64, bit_rate_hz: f64, frequency_hz: f64) -> PyResult<f64> {
    dd::quantum_limit_w(ber, bit_rate_hz, frequency_hz).py()
}

/// The power penalty of a finite extinction ratio of a thermal-noise-limited receiver.
/// After [1]_, eq. (4.6.4) and Fig. 4.20:
///
/// .. math::
///
///     \delta_{\mathrm{ex}} = 10 \log_{10} \frac{1 + r_{\mathrm{ex}}}{1 - r_{\mathrm{ex}}},
///     \qquad r_{\mathrm{ex}} = \frac{P_0}{P_1},
///
/// 0.87 dB at :math:`r_{\mathrm{ex}} = 0.1`.
///
/// Args:
///     r_ex (float | array_like): the extinction ratio :math:`P_0/P_1 \in [0, 1)`.
///
/// Returns:
///     float | numpy.ndarray: the penalty (dB).
///
/// Raises:
///     ValueError: for a ratio outside :math:`[0, 1)`.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.extinction_ratio_penalty_db(0.1), 2)
///     0.87
#[pyfunction]
fn extinction_ratio_penalty_db(r_ex: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(r_ex, |v| dd::extinction_ratio_penalty_db(v).py())
}

/// The power penalty of the relative intensity noise of the source at a Q factor.
/// After [1]_, eq. (4.6.11) and Fig. 4.21: the intensity noise :math:`\sigma_I = r_I R P_1` adds to the noise
/// of the 1 bits and
///
/// .. math::
///
///     \delta_I = -10 \log_{10}\left(1 - r_I^2 Q^2\right),
///
/// diverging at :math:`r_I Q = 1`.
///
/// Args:
///     r_i (float): the relative intensity noise :math:`r_I` (r.m.s. over the power).
///     q (float): the Q factor.
///
/// Returns:
///     float: the penalty (dB).
///
/// Raises:
///     ValueError: for negative arguments or :math:`r_I Q \ge 1`.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.intensity_noise_penalty_db(0.1, 6.0), 2)
///     1.94
#[pyfunction]
fn intensity_noise_penalty_db(r_i: f64, q: f64) -> PyResult<f64> {
    dd::intensity_noise_penalty_db(r_i, q).py()
}

/// The broadening factor of the r.m.s. width of a chirped Gaussian pulse after a
/// dispersive fibre, [1]_, eq. (2.4.22): for the pulse :math:`\exp[-(1 + jC) t^2 / (2
/// T_0^2)]` of r.m.s. width :math:`\sigma_0 = T_0/\sqrt{2}`,
///
/// .. math::
///
///     f_b^2 = \left(\frac{\sigma}{\sigma_0}\right)^2 =
///     \left(1 + \frac{C \beta_2 L}{T_0^2}\right)^2 + \left(\frac{\beta_2 L}{T_0^2}\right)^2
///     + (1 + C^2)^2 \left(\frac{\beta_3 L}{2 T_0^3}\right)^2.
///
/// Without :math:`\beta_3` the pulse stays Gaussian and :math:`f_b` is also the growth
/// of its half-width (eq. (2.4.17)).
///
/// Args:
///     t0_s (float): the half-width :math:`T_0` at :math:`1/e` of the intensity (s).
///     chirp (float): the chirp parameter :math:`C`.
///     beta2_s2_km (float): the group-velocity dispersion :math:`\beta_2` (s²/km),
///         ``FiberParams.beta2_s2_km``.
///     beta3_s3_km (float): the third-order dispersion :math:`\beta_3` (s³/km),
///         ``FiberParams.beta3_s3_km``, 0 for none.
///     length_km (float): the length :math:`L` (km).
///
/// Returns:
///     float: :math:`f_b`.
///
/// Raises:
///     ValueError: for a non-positive width.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.gaussian_broadening_factor(10e-12, 0.0, -20e-24, 0.0, 5.0), 6)
///     1.414214
#[pyfunction]
fn gaussian_broadening_factor(
    t0_s: f64,
    chirp: f64,
    beta2_s2_km: f64,
    beta3_s3_km: f64,
    length_km: f64,
) -> PyResult<f64> {
    dd::gaussian_broadening_factor(t0_s, chirp, beta2_s2_km, beta3_s3_km, length_km).py()
}

/// The power penalty of a Gaussian timing jitter on a thermal-noise-limited receiver of
/// raised-cosine pulses :math:`\cos^2(\pi B t/2)`. After [1]_, eqs. (4.6.17), (4.6.20)
/// and Fig. 4.22:
///
/// .. math::
///
///     \delta_j = 10 \log_{10}\frac{1 - b/2}{(1 - b/2)^2 - b^2 Q^2/2}, \qquad
///     b = \left(\frac{4\pi^2}{3} - 8\right)(B \tau_j)^2,
///
/// about 2 dB at :math:`B \tau_j = 0.16` for :math:`Q = 6`, infinite beyond about 0.2.
///
/// Args:
///     b_tau_j (float): the r.m.s. jitter :math:`B \tau_j` in bit periods.
///     q (float): the Q factor.
///
/// Returns:
///     float: the penalty (dB).
///
/// Raises:
///     ValueError: for negative arguments or a jitter that no power compensates.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.timing_jitter_penalty_db(0.16, 6.0), 2)
///     2.23
#[pyfunction]
fn timing_jitter_penalty_db(b_tau_j: f64, q: f64) -> PyResult<f64> {
    dd::timing_jitter_penalty_db(b_tau_j, q).py()
}

/// The chirp power penalty of a linearly chirped Gaussian pulse that fills the bit,
/// :math:`\sigma_0 = 1/(4B)`. After [1]_, eq. (5.4.14) and Fig. 5.11:
///
/// .. math::
///
///     \delta_c = 5 \log_{10}\left[(1 + 8 C \beta_2 B^2 L)^2 + (8 \beta_2 B^2 L)^2\right],
///
/// negative where :math:`C \beta_2 < 0` compresses the pulse first; :math:`C = 0` is the
/// dispersion penalty of a chirp-free pulse of a narrow-linewidth source.
///
/// Args:
///     chirp (float): the chirp parameter :math:`C` (negative for a directly modulated
///         laser, about minus its linewidth enhancement factor).
///     beta2_s2_km (float): the group-velocity dispersion :math:`\beta_2` (s²/km).
///     bit_rate_hz (float): the bit rate :math:`B` (bit/s).
///     length_km (float): the length :math:`L` (km).
///
/// Returns:
///     float: the penalty (dB).
///
/// Raises:
///     ValueError: for a negative bit rate or length.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.chirp_penalty_db(-6.0, -20e-24, 10e9, 100.0), 1)  # |β₂| B² L = 0.2
///     10.3
#[pyfunction]
fn chirp_penalty_db(chirp: f64, beta2_s2_km: f64, bit_rate_hz: f64, length_km: f64) -> PyResult<f64> {
    dd::chirp_penalty_db(chirp, beta2_s2_km, bit_rate_hz, length_km).py()
}

/// The power penalty of the broadening of a Gaussian pulse that fills the bit. After
/// [1]_, eq. (5.4.1):
///
/// .. math::
///
///     \delta_d = 10 \log_{10} f_b.
///
/// Args:
///     b_f (float | array_like): the broadening factor :math:`\sigma/\sigma_0 \ge 1`.
///
/// Returns:
///     float | numpy.ndarray: the penalty (dB).
///
/// Raises:
///     ValueError: for a factor below 1.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
#[pyfunction]
fn broadening_penalty_db(b_f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(b_f, |v| dd::broadening_penalty_db(v).py())
}

/// The dispersion power penalty of a source of finite spectral width. After [1]_,
/// eq. (5.4.4) and Fig. 5.7: the Gaussian output pulse fills the bit, :math:`4 B \sigma = 1`, with :math:`\sigma^2
/// = \sigma_0^2 + (D L \sigma_\lambda)^2`, and
///
/// .. math::
///
///     \delta_d = -5 \log_{10}\left[1 - (4 B L |D| \sigma_\lambda)^2\right],
///
/// diverging at :math:`4 B L |D| \sigma_\lambda = 1`.
///
/// Args:
///     bit_rate_hz (float): the bit rate :math:`B` (bit/s).
///     length_km (float): the length :math:`L` (km).
///     d_ps_nm_km (float): the dispersion :math:`D` (ps/(nm km)).
///     sigma_lambda_nm (float): the r.m.s. spectral width of the source (nm).
///
/// Returns:
///     float: the penalty (dB).
///
/// Raises:
///     ValueError: for negative arguments or a pulse that overfills the bit.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.dispersion_penalty_db(2.5e9, 50.0, 17.0, 0.05), 2)
///     0.43
#[pyfunction]
fn dispersion_penalty_db(bit_rate_hz: f64, length_km: f64, d_ps_nm_km: f64, sigma_lambda_nm: f64) -> PyResult<f64> {
    dd::dispersion_penalty_db(bit_rate_hz, length_km, d_ps_nm_km, sigma_lambda_nm).py()
}

/// The dispersion-limited length of an unchirped Gaussian pulse of optimum width. After
/// [1]_, eqs. (2.4.30) and (5.2.3): from :math:`4 B \sigma \le 1`,
///
/// .. math::
///
///     L = \frac{1}{16 B^2 |\beta_2|}.
///
/// Args:
///     bit_rate_hz (float): the bit rate :math:`B` (bit/s).
///     beta2_s2_km (float): the group-velocity dispersion :math:`\beta_2` (s²/km).
///
/// Returns:
///     float: the length (km).
///
/// Raises:
///     ValueError: for a non-positive bit rate or a zero dispersion.
///
/// References:
///     .. [1] G. P. Agrawal, *Fiber-Optic Communication Systems*, 3rd ed., Wiley, 2002.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.dispersion_limited_length_km(10e9, -21.7e-24), 1)  # standard fibre
///     28.8
#[pyfunction]
fn dispersion_limited_length_km(bit_rate_hz: f64, beta2_s2_km: f64) -> PyResult<f64> {
    dd::dispersion_limited_length_km(bit_rate_hz, beta2_s2_km).py()
}

/// The small-signal response :math:`H(f)` of a dispersive IM-DD link. After [2]_, [3]_: a
/// laser of linewidth enhancement :math:`\beta_c` and adiabatic chirp coefficient
/// :math:`\kappa` at the mean power :math:`P_0`, a fibre of dispersion :math:`\beta_2`
/// and length :math:`L`, a square-law photodiode; for the power :math:`P_0(1 + m \cos
/// \Omega t)`, :math:`m \ll 1`, the detected modulation is :math:`m\,\Re(H e^{j\Omega
/// t})` with
///
/// .. math::
///
///     H(\Omega) = \cos\theta + \beta_c \sin\theta - j\,\frac{\beta_c \kappa P_0}{\Omega}
///     \sin\theta, \qquad \theta = \frac{\beta_2 \Omega^2 L}{2},
///
/// :math:`H(0) = 1`; on standard fibre (:math:`\beta_2 < 0`) the transient chirp of a
/// positive :math:`\beta_c` brings the first fading closer and the adiabatic chirp fills
/// the notches.
///
/// Args:
///     f_hz (float): the modulation frequency (Hz).
///     chirp (float): the linewidth enhancement factor :math:`\beta_c`.
///     adiabatic_hz_w (float): the adiabatic chirp coefficient :math:`\kappa` (Hz/W), 0
///         for the transient chirp alone.
///     power_w (float): the mean power :math:`P_0` (W).
///     beta2_s2_km (float): the dispersion :math:`\beta_2` (s²/km), ``FiberParams.beta2_s2_km``.
///     length_km (float): the length :math:`L` (km).
///
/// Returns:
///     complex: :math:`H(f)`.
///
/// Raises:
///     ValueError: for a negative frequency, power or length, or a zero frequency with
///         adiabatic chirp.
///
/// References:
///     .. [2] J. Wang, K. Petermann, "Small signal analysis for dispersive optical fiber
///        communication systems," *J. Lightw. Technol.* 10(1), 96–100, 1992.
///     .. [3] F. Devaux, Y. Sorel, J. F. Kerdiles, "Simple measurement of fiber dispersion
///        and of chirp parameter of intensity modulated light emitter," *J. Lightw.
///        Technol.* 11(12), 1937–1940, 1993.
///
/// Examples:
///     >>> import comnum
///     >>> f0 = comnum.fading_frequency_hz(0, 3.0, -21.7e-24, 40.0)
///     >>> round(abs(comnum.dispersion_fading_response(f0, 3.0, 0.0, 1e-3, -21.7e-24, 40.0)), 9)
///     0.0
#[pyfunction]
fn dispersion_fading_response(
    f_hz: f64,
    chirp: f64,
    adiabatic_hz_w: f64,
    power_w: f64,
    beta2_s2_km: f64,
    length_km: f64,
) -> PyResult<C64> {
    dd::dispersion_fading_response(f_hz, chirp, adiabatic_hz_w, power_w, beta2_s2_km, length_km).py()
}

/// The fading frequency :math:`f_u` (Hz) of order :math:`u` of a dispersive IM-DD link
/// with a transient chirp :math:`\beta_c`, the zeros of :func:`dispersion_fading_response`
/// without adiabatic chirp. After [3]_:
///
/// .. math::
///
///     f_u^2 = \frac{1 + 2u - \frac{2}{\pi}\arctan\beta_c}{4\pi |\beta_2| L}
///     = \frac{c}{2 D L \lambda^2}\left(1 + 2u - \frac{2}{\pi}\arctan\beta_c\right).
///
/// Args:
///     u (int): the order, 0 for the first notch.
///     chirp (float): the linewidth enhancement factor :math:`\beta_c`.
///     beta2_s2_km (float): the dispersion :math:`\beta_2` (s²/km).
///     length_km (float): the length :math:`L` (km).
///
/// Returns:
///     float: :math:`f_u` (Hz).
///
/// Raises:
///     ValueError: for a zero dispersion or a non-positive length.
///
/// References:
///     .. [3] F. Devaux, Y. Sorel, J. F. Kerdiles, "Simple measurement of fiber dispersion
///        and of chirp parameter of intensity modulated light emitter," *J. Lightw.
///        Technol.* 11(12), 1937–1940, 1993.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.fading_frequency_hz(0, 0.0, -21.7e-24, 40.0) / 1e9, 1)  # 40 km of SSMF
///     9.6
#[pyfunction]
fn fading_frequency_hz(u: usize, chirp: f64, beta2_s2_km: f64, length_km: f64) -> PyResult<f64> {
    dd::fading_frequency_hz(u, chirp, beta2_s2_km, length_km).py()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDirectDetectionReceiver>()?;
    for f in [
        wrap_pyfunction!(q_factor, m)?,
        wrap_pyfunction!(excess_noise_factor, m)?,
        wrap_pyfunction!(quantum_limit_photons_per_bit, m)?,
        wrap_pyfunction!(quantum_limit_w, m)?,
        wrap_pyfunction!(extinction_ratio_penalty_db, m)?,
        wrap_pyfunction!(intensity_noise_penalty_db, m)?,
        wrap_pyfunction!(timing_jitter_penalty_db, m)?,
        wrap_pyfunction!(chirp_penalty_db, m)?,
        wrap_pyfunction!(gaussian_broadening_factor, m)?,
        wrap_pyfunction!(broadening_penalty_db, m)?,
        wrap_pyfunction!(dispersion_penalty_db, m)?,
        wrap_pyfunction!(dispersion_limited_length_km, m)?,
        wrap_pyfunction!(dispersion_fading_response, m)?,
        wrap_pyfunction!(fading_frequency_hz, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
