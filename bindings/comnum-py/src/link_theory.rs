//! Closed forms of OFDM links with impairments: equalisation with an imperfect channel
//! estimate over Rayleigh fading (Chang & Su 2002), differential detection over Jakes'
//! fading, and the distortion of a nonlinear amplifier driven by an OFDM signal
//! (Banelli & Cacopardi 2000).

use crate::{FArr, IntoPy, count, cvec, farr, fvec, map_real};
use comnum::dsp::nonlinear;
use comnum::mapping::constellation::Constellation;
use comnum::mapping::imperfect_csi as csi;
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The bit error rate over flat Rayleigh fading of a receiver that equalises with a
/// channel estimate jointly Gaussian with the channel, of correlations `rho1` (real) and
/// `rho2` (imaginary) and of standard deviation ratio `sigma_ratio` :math:`= \sigma_2/\sigma_1`
/// (M.-X. Chang and Y. T. Su, *IEEE Trans. Wireless Commun.* 1(4), 2002, eqs. 31, 32, 53;
/// every linear pilot-assisted estimate has this form). `rho1 = 1`, `rho2 = 0`,
/// `sigma_ratio = 1`: the perfect channel.
///
/// The received sample of a subcarrier is
///
/// .. math::
///
///     Y = HX + N,
///
/// :math:`H` zero-mean circular Gaussian of variance :math:`2\sigma_1^2`, equalised with an
/// estimate :math:`\hat{H}` of variance :math:`2\sigma_2^2` and cross-correlation
/// :math:`\frac{1}{2}\mathbb{E}[\hat{H} H^*] = \mu_1 + j\mu_2`, with
/// :math:`\rho_i = \mu_i/(\sigma_1\sigma_2)`. At the mean SNR per bit
/// :math:`\bar{\gamma}_b`, the BER of BPSK (eq. 31) and of Gray QPSK (eq. 32) are
///
/// .. math::
///
///     \begin{aligned}
///     P_b^{\mathrm{BPSK}} &= \frac{1}{2}\left[1 - \frac{\rho_1}{\sqrt{1 + 1/\bar{\gamma}_b
///     - \rho_2^2}}\right], \\
///     P_b^{\mathrm{QPSK}} &= \frac{1}{2}\left[1 - \frac{1}{2}\frac{(\rho_1 + \rho_2)/\sqrt{2}}
///     {\sqrt{1 + \frac{1}{2\bar{\gamma}_b} - \frac{(\rho_1 - \rho_2)^2}{2}}} - \frac{1}{2}
///     \frac{(\rho_1 - \rho_2)/\sqrt{2}}{\sqrt{1 + \frac{1}{2\bar{\gamma}_b}
///     - \frac{(\rho_1 + \rho_2)^2}{2}}}\right],
///     \end{aligned}
///
/// and that of Gray 16-QAM the sum of 24 terms of eq. 53 with the coefficients of their
/// Table I.
///
/// Args:
///     modulation (str): ``"bpsk"``, ``"qpsk"`` (Gray) or ``"16qam"`` (Gray), any name of
///         a constellation (``"16-QAM"``, ``"QPSK"``).
///     ebn0_db (float | array_like): the mean :math:`E_b/N_0` (dB).
///     rho1 (float): the real correlation coefficient.
///     rho2 (float): the imaginary correlation coefficient.
///     sigma_ratio (float): :math:`\sigma_2/\sigma_1` (16-QAM only: PSK does not depend on
///         :math:`|\hat{H}|`).
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> p = comnum.theory_ber_imperfect_csi("16qam", 50.0, 0.999)
///     >>> round(p, 4)          # the error floor of Chang & Su's Fig. 4
///     0.0034
#[pyfunction]
#[pyo3(signature = (modulation, ebn0_db, rho1, *, rho2 = 0.0, sigma_ratio = 1.0))]
fn theory_ber_imperfect_csi(
    modulation: &str,
    ebn0_db: &Bound<'_, PyAny>,
    rho1: f64,
    rho2: f64,
    sigma_ratio: f64,
) -> PyResult<Py<PyAny>> {
    let c = Constellation::from_name(modulation).py()?;
    map_real(ebn0_db, |e| csi::ber_imperfect_csi(&c, comnum::math::db2lin(e), rho1, rho2, sigma_ratio).py())
}

/// The bit error rate of DBPSK (`m = 2`) or Gray DQPSK (`m = 4`) over Jakes' Rayleigh
/// fading of normalised Doppler `fd_ts` (Chang & Su 2002, eqs. 40, 41).
///
/// At the mean SNR per bit :math:`\bar{\gamma}_b`:
///
/// .. math::
///
///     \begin{aligned}
///     P_b^{\mathrm{DBPSK}} &= \frac{1}{2}\left[1 - \frac{J_0(2\pi f_d T_s)}{1
///     + 1/\bar{\gamma}_b}\right], \\
///     P_b^{\mathrm{DQPSK}} &= \frac{1}{2}\left[1 - \frac{J_0(2\pi f_d T_s)}{\sqrt{(2
///     + 1/\bar{\gamma}_b)(1 + 1/(2\bar{\gamma}_b)) - J_0^2(2\pi f_d T_s)}}\right],
///     \end{aligned}
///
/// an error floor of :math:`(1 - J_0(2\pi f_d T_s))/2` for DBPSK.
///
/// Args:
///     m (int): 2 or 4.
///     ebn0_db (float | array_like): the mean :math:`E_b/N_0` (dB).
///     fd_ts (float): the maximum Doppler frequency times the symbol duration.
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every :math:`E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn theory_ber_dpsk_jakes(m: usize, ebn0_db: &Bound<'_, PyAny>, fd_ts: f64) -> PyResult<Py<PyAny>> {
    let g = |e: f64| comnum::math::db2lin(e);
    match m {
        2 => map_real(ebn0_db, |e| csi::ber_dbpsk_jakes(g(e), fd_ts).py()),
        4 => map_real(ebn0_db, |e| csi::ber_dqpsk_jakes(g(e), fd_ts).py()),
        _ => Err(PyValueError::new_err(format!("m is 2 or 4, got {m}"))),
    }
}

/// The coefficients :math:`c_n` of the output autocorrelation of a memoryless envelope
/// nonlinearity :math:`f` driven by a circular complex Gaussian signal of unit power
/// (P. Banelli and S. Cacopardi, *IEEE Trans. Commun.* 48(3), 2000, eq. 7): :math:`c_0` is
/// the useful power :math:`|\alpha|^2` (Bussgang), the term :math:`n` has the input spectrum
/// convolved :math:`2n + 1` times with itself, and the coefficients sum to the output power.
///
/// For an envelope nonlinearity :math:`y = f(r) e^{j\theta}` and an input of power
/// :math:`2\sigma^2` (here 1), of autocorrelation :math:`R_{xx}` (eqs. 6–8):
///
/// .. math::
///
///     R_{yy}(\tau) = \sum_{n=0}^{\infty} c_n \frac{R_{xx}(\tau)^{n+1}
///     R_{xx}^*(\tau)^n}{(2\sigma^2)^{2n+1}}, \qquad
///     c_n = \frac{1}{2\sigma^2} \frac{1}{n+1} \left| \int_0^{\infty} f(r)
///     \frac{r^2}{\sigma^2} e^{-r^2/2\sigma^2} L_n^{(1)}\!\left(\frac{r^2}{2\sigma^2}
///     \right) dr \right|^2,
///
/// with the generalised Laguerre polynomials :math:`L_n^{(1)}`.
///
/// Args:
///     f (callable): the complex output envelope :math:`A(r) \exp(j\Phi(r))` of an input
///         envelope :math:`r` (vectorised over a NumPy array).
///     n_terms (int): the number of coefficients.
///
/// Returns:
///     numpy.ndarray: :math:`c_0, \dots, c_{n_{\mathrm{terms}} - 1}`.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> c = comnum.envelope_series(lambda r: np.minimum(r, np.sqrt(2.0)), 60)  # 3 dB IBO
///     >>> round(float(c.sum()), 4), round(float(1 - np.exp(-2.0)), 4)
///     (0.8647, 0.8647)
#[pyfunction]
#[pyo3(signature = (f, n_terms = 60))]
fn envelope_series<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, n_terms: i64) -> PyResult<FArr<'py>> {
    let n_terms = count(n_terms, "n_terms")?;
    let r = farr(py, nonlinear::series_grid());
    let values: Vec<C64> = cvec(&f.call1((r,))?)?;
    Ok(farr(py, nonlinear::output_series_sampled(&values, n_terms)))
}

/// The power of the useful term and of the distortion on each of `bins` frequency bins,
/// for the coefficients `c` of `envelope_series` and an input flat on the `occupied`
/// bins centred on bin 0 (an OFDM signal of `occupied` subcarriers oversampled to
/// `bins`; Banelli & Cacopardi, eq. 9).
///
/// Args:
///     c (array_like): the coefficients :math:`c_n`.
///     occupied (int): the occupied bins.
///     bins (int): the bins of the circular grid (FFT order: bin 0 first).
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the useful and the distortion power per bin.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn distortion_spectrum<'py>(
    py: Python<'py>,
    c: &Bound<'py, PyAny>,
    occupied: usize,
    bins: usize,
) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let (u, d) = nonlinear::distortion_spectrum(&fvec(c)?, occupied, bins).py()?;
    Ok((farr(py, u), farr(py, d)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theory_ber_imperfect_csi, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_dpsk_jakes, m)?)?;
    m.add_function(wrap_pyfunction!(envelope_series, m)?)?;
    m.add_function(wrap_pyfunction!(distortion_spectrum, m)?)?;
    Ok(())
}
