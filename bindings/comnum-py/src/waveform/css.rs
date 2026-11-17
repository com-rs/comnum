//! Chirp spread spectrum (the modulation of LoRa) and its closed forms.

use super::*;

/// Chirp spread spectrum, the frequency-shift chirp modulation of LoRa:
/// :math:`M = 2^{\mathrm{SF}}` chirps of :math:`M` chips, a noncoherent DFT receiver. LoRa
/// is a trademark of Semtech; this models the published descriptions for research.
///
/// The symbol :math:`a \in \{0, \dots, M - 1\}` is a chirp over the bandwidth :math:`B`
/// whose initial frequency is shifted by :math:`aB/M` and folded at :math:`B/2`; its
/// complex envelope over :math:`0 \le t < T_s = M/B` is ([1]_, eq. (5))
///
/// .. math::
///
///     x(t; a) = \exp\left\{j 2\pi B t \left(\frac{a}{M} - \frac{1}{2} + \frac{Bt}{2M} -
///     u\left(t - \frac{M - a}{B}\right)\right)\right\},
///
/// sampled `oversampling` times per chip. The receiver multiplies every symbol by the
/// conjugate of the chirp of symbol 0, takes a DFT and decides the bin of largest energy
/// [2]_.
///
/// Args:
///     sf (int): spreading factor.
///     oversampling (int): samples per chip.
///     waveform (str): `"lora"`, or for comparison `"vangelista"` (the discrete chirp of
///         [3]_) and `"fsk"` (one sample per chip).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform
///        properties and spectral analysis," *IEEE Internet Things J.*, 6(5), 8463–8470,
///        2019.
///     .. [2] O. Afisiadis, M. Cotting, A. Burg and A. Balatsoukas-Stimming, "On the error
///        rate of the LoRa modulation with interference," *IEEE Trans. Wireless Commun.*,
///        19(2), 1292–1304, 2020.
///     .. [3] L. Vangelista, "Frequency shift chirp modulation: the LoRa modulation," *IEEE
///        Signal Process. Lett.*, 24(12), 1818–1821, 2017.
///
/// Examples:
///     >>> import comnum
///     >>> css = comnum.Css(7)
///     >>> s = css.modulate([0, 5, 127])
///     >>> s.size, css.demodulate(s).tolist()
///     (384, [0, 5, 127])
#[pyclass(name = "Css", module = "comnum")]
pub struct PyCss {
    inner: Css,
}

#[pymethods]
impl PyCss {
    #[new]
    #[pyo3(signature = (sf, *, oversampling=1, waveform="lora"))]
    fn new(sf: u32, oversampling: usize, waveform: &str) -> PyResult<Self> {
        let inner = Css {
            sf,
            oversampling,
            waveform: waveform.parse::<Waveform>().py()?,
            sample_rate: comnum::core::units::Hertz(1.0),
        };
        inner.symbol(0).py()?;
        Ok(PyCss { inner })
    }

    /// int: The spreading factor.
    #[getter]
    fn sf(&self) -> u32 {
        self.inner.sf
    }

    /// int: The number of symbols and of chips per symbol, :math:`M = 2^{\mathrm{SF}}`.
    #[getter]
    fn m(&self) -> usize {
        self.inner.m()
    }

    /// int: Samples per symbol, :math:`J M`.
    #[getter]
    fn symbol_len(&self) -> usize {
        self.inner.symbol_len()
    }

    /// The samples of one symbol.
    ///
    /// Args:
    ///     a (int): the symbol, :math:`0 \le a < M`.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`J M` unit-envelope samples.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn symbol<'py>(&self, py: Python<'py>, a: usize) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.symbol(a).py()?))
    }

    /// The samples of a sequence of symbols.
    ///
    /// Args:
    ///     symbols (array_like): integers in :math:`[0, M)`.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`J M` samples per symbol.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&uvec(symbols)?).py()?))
    }

    /// The energy of the :math:`M` candidates of every received symbol (squared magnitudes of
    /// the DFT of the dechirped samples).
    ///
    /// Args:
    ///     samples (array_like): a multiple of :math:`J M` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`M` energies per symbol.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn energies<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.energies(&cvec(samples)?).py()?))
    }

    /// The noncoherent decisions.
    ///
    /// Args:
    ///     samples (array_like): a multiple of :math:`J M` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: one symbol per :math:`J M` samples.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn demodulate<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, self.inner.demodulate(&cvec(samples)?).py()?))
    }

    /// One symbol of a same-SF interferer offset by `tau` chips: the end of its symbol
    /// `s1`, then its symbol `s2` (Afisiadis et al. 2020, eq. (21)).
    ///
    /// Args:
    ///     s1 (int): the symbol that ends in the window.
    ///     s2 (int): the symbol that starts `tau` chips in.
    ///     tau (float): offset in chips, :math:`0 \le \tau < M`.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`M` samples.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn interferer<'py>(&self, py: Python<'py>, s1: usize, s2: usize, tau: f64) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.interferer(s1, s2, tau).py()?))
    }

    fn __repr__(&self) -> String {
        let c = &self.inner;
        format!("Css(sf={}, oversampling={}, waveform={:?})", c.sf, c.oversampling, c.waveform)
    }
}

/// The SER of the noncoherent CSS receiver over AWGN at the SNR per sample (dB).
///
/// The exact SER, that of :math:`M` orthogonal signals detected noncoherently, is
/// ([1]_, eq. (16))
///
/// .. math::
///
///     P_s = \int_{0}^{\infty} f_{\mathrm{Ri}}(y; v, 1)\,\left[1 - \left(1 -
///     e^{-y^{2}/2}\right)^{M-1}\right] dy,
///
/// with :math:`f_{\mathrm{Ri}}` the Rice density of location :math:`v = \sqrt{2M\rho}`,
/// :math:`\rho` the SNR per sample; `approximation="rice"` or `"gumbel"` gives their
/// eqs. (17) and (18).
///
/// Args:
///     sf (int): spreading factor.
///     snr_db (float | array_like): SNR per sample (dB).
///     approximation (str | None): `None`, `"rice"` or `"gumbel"`.
///
/// Returns:
///     float | numpy.ndarray: the SER.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] O. Afisiadis, M. Cotting, A. Burg and A. Balatsoukas-Stimming, "On the error
///        rate of the LoRa modulation with interference," *IEEE Trans. Wireless Commun.*,
///        19(2), 1292–1304, 2020.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_css_ser(7, -8.0), 5)
///     0.00161
#[pyfunction]
#[pyo3(signature = (sf, snr_db, *, approximation=None))]
fn theory_css_ser(sf: u32, snr_db: &Bound<'_, PyAny>, approximation: Option<&str>) -> PyResult<Py<PyAny>> {
    let f: fn(u32, f64) -> comnum::Result<f64> = match approximation {
        None => css_theory::ser,
        Some("rice") => css_theory::ser_rice,
        Some("gumbel") => css_theory::ser_gumbel,
        Some(other) => {
            return Err(PyValueError::new_err(format!("approximation must be rice or gumbel, got {other:?}")));
        }
    };
    map_real(snr_db, |s| f(sf, s).py())
}

/// The BER of the noncoherent CSS receiver over AWGN.
///
/// Every wrong symbol being equally likely,
///
/// .. math::
///
///     P_b = P_s \frac{M}{2(M - 1)},
///
/// with :math:`P_s` the SER of :func:`theory_css_ser`.
///
/// Args:
///     sf (int): spreading factor.
///     snr_db (float | array_like): SNR per sample (dB).
///
/// Returns:
///     float | numpy.ndarray: the BER.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn theory_css_ber(sf: u32, snr_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(snr_db, |s| css_theory::ber(sf, s).py())
}

/// The cross-correlation of two continuous-time LoRa waveforms.
///
/// The correlation of the waveforms of the symbols :math:`\ell` and :math:`m` ([1]_,
/// eq. (7)) is
///
/// .. math::
///
///     C_{\ell, m} = \frac{1}{T_{s}}\int_{0}^{T_{s}} x(t; \ell)\, x^{*}(t; m)\, dt
///     = M \frac{e^{j2\pi\ell(m - \ell)/M} - e^{j2\pi m(m - \ell)/M}}
///     {j2\pi(M - |m - \ell|)\,|m - \ell|}
///
/// for :math:`\ell \ne m`, and :math:`1` for :math:`\ell = m`.
///
/// Args:
///     sf (int): spreading factor.
///     l (int): first symbol.
///     m (int): second symbol.
///
/// Returns:
///     complex: :math:`C(l, m)`.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform
///        properties and spectral analysis," *IEEE Internet Things J.*, 6(5), 8463–8470,
///        2019.
#[pyfunction]
fn css_cross_correlation(sf: u32, l: usize, m: usize) -> PyResult<C64> {
    css_theory::cross_correlation(sf, l, m).py()
}

/// The largest real cross-correlation of the continuous-time waveforms.
///
/// See Table I of [1]_.
///
/// Args:
///     sf (int): spreading factor.
///
/// Returns:
///     float: :math:`\max \operatorname{Re} C(l, m)` over :math:`l \ne m`.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform
///        properties and spectral analysis," *IEEE Internet Things J.*, 6(5), 8463–8470,
///        2019.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.css_max_cross_correlation(7), 3)
///     0.045
#[pyfunction]
fn css_max_cross_correlation(sf: u32) -> PyResult<f64> {
    css_theory::max_cross_correlation(sf).py()
}

/// The continuous part of the power spectrum of LoRa, :math:`G^c(f) B`, at :math:`f` in units
/// of :math:`B`.
///
/// With :math:`X(f; \ell)` the Fourier transform of the waveform of symbol :math:`\ell`
/// ([1]_, eqs. (20)–(25)),
///
/// .. math::
///
///     G^{c}(f) B = \frac{1}{T_{s} M} \left(\sum_{\ell} |X(f; \ell)|^{2}
///     - \frac{1}{M} \Big|\sum_{\ell} X(f; \ell)\Big|^{2}\right),
///
/// for independent uniform symbols; its integral is :math:`1 - 1/M`.
///
/// Args:
///     sf (int): spreading factor.
///     f (array_like): frequencies in units of :math:`B`.
///
/// Returns:
///     numpy.ndarray: one value per frequency.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform
///        properties and spectral analysis," *IEEE Internet Things J.*, 6(5), 8463–8470,
///        2019.
#[pyfunction]
fn css_continuous_spectrum<'py>(py: Python<'py>, sf: u32, f: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
    Ok(farr(py, css_theory::continuous_spectrum(sf, &fvec(f)?).py()?))
}

/// The lines of the spectrum of LoRa at :math:`n B/M`, :math:`|n| \le n_{\max}`.
///
/// The line :math:`n` has the power ([1]_, eq. (21))
///
/// .. math::
///
///     P_n = \frac{1}{T_{s}^{2} M^{2}} \Big|\sum_{\ell} X(nB/M; \ell)\Big|^{2},
///
/// :math:`X(f; \ell)` the Fourier transform of the waveform of symbol :math:`\ell`; the
/// lines hold :math:`1/M` of the power.
///
/// Args:
///     sf (int): spreading factor.
///     n_max (int): largest index :math:`|n|` of a line.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the frequencies (units of :math:`B`) and powers.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform
///        properties and spectral analysis," *IEEE Internet Things J.*, 6(5), 8463–8470,
///        2019.
#[pyfunction]
fn css_spectral_lines<'py>(py: Python<'py>, sf: u32, n_max: i64) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let lines = css_theory::spectral_lines(sf, count(n_max, "n_max")?).py()?;
    let (f, p): (Vec<f64>, Vec<f64>) = lines.into_iter().unzip();
    Ok((farr(py, f), farr(py, p)))
}

/// The band (units of :math:`B`) that holds `fraction` of the power of LoRa.
///
/// See Table I of [1]_ (:math:`B_{99}`).
///
/// Args:
///     sf (int): spreading factor.
///     fraction (float): fraction of the power.
///     span (float): half-width (units of :math:`B`) of the computed spectrum.
///
/// Returns:
///     float: the band in units of :math:`B`.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform
///        properties and spectral analysis," *IEEE Internet Things J.*, 6(5), 8463–8470,
///        2019.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.css_occupied_bandwidth(5), 2)
///     1.18
#[pyfunction]
#[pyo3(signature = (sf, fraction=0.99, span=3.0))]
fn css_occupied_bandwidth(sf: u32, fraction: f64, span: f64) -> PyResult<f64> {
    css_theory::occupied_bandwidth(sf, fraction, span).py()
}

/// The pattern :math:`R_k` that a same-SF interferer leaves after the dechirping.
///
/// See eq. (25) of [1]_.
///
/// Args:
///     sf (int): spreading factor.
///     s1 (int): the interfering symbol that ends in the window.
///     s2 (int): the interfering symbol that starts `tau` chips in.
///     tau (float): offset in chips, :math:`0 \le \tau < M`.
///
/// Returns:
///     numpy.ndarray: :math:`M` complex values.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] O. Afisiadis, M. Cotting, A. Burg and A. Balatsoukas-Stimming, "On the error
///        rate of the LoRa modulation with interference," *IEEE Trans. Wireless Commun.*,
///        19(2), 1292–1304, 2020.
#[pyfunction]
fn css_interference_pattern<'py>(py: Python<'py>, sf: u32, s1: usize, s2: usize, tau: f64) -> PyResult<CArr<'py>> {
    Ok(carr(py, css_theory::interference_pattern(sf, s1, s2, tau).py()?))
}

/// The approximate SER (or, with `frame`, the frame error rate) under AWGN and a same-SF
/// interferer of power `pi_db` relative to the signal.
///
/// The approximation is that of [1]_ (eqs. (61)–(67)), with the strongest bin at
/// :math:`-\operatorname{round}(\tau)`.
///
/// Args:
///     sf (int): spreading factor.
///     snr_db (float | array_like): SNR per sample (dB).
///     pi_db (float): power of the interferer (dB).
///     frame (int | None): symbols per frame for the frame error rate.
///     eps (float): step of the offset (chips).
///
/// Returns:
///     float | numpy.ndarray: the SER or the FER at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] O. Afisiadis, M. Cotting, A. Burg and A. Balatsoukas-Stimming, "On the error
///        rate of the LoRa modulation with interference," *IEEE Trans. Wireless Commun.*,
///        19(2), 1292–1304, 2020.
#[pyfunction]
#[pyo3(signature = (sf, snr_db, pi_db, *, frame=None, eps=0.2))]
fn theory_css_ser_interference(
    sf: u32,
    snr_db: &Bound<'_, PyAny>,
    pi_db: f64,
    frame: Option<usize>,
    eps: f64,
) -> PyResult<Py<PyAny>> {
    map_real(snr_db, |s| {
        match frame {
            None => css_theory::ser_interference(sf, s, pi_db, eps),
            Some(f) => css_theory::fer_interference(sf, s, pi_db, f, eps),
        }
        .py()
    })
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCss>()?;
    m.add_function(wrap_pyfunction!(theory_css_ser, m)?)?;
    m.add_function(wrap_pyfunction!(theory_css_ber, m)?)?;
    m.add_function(wrap_pyfunction!(css_cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(css_max_cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(css_continuous_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(css_spectral_lines, m)?)?;
    m.add_function(wrap_pyfunction!(css_occupied_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(css_interference_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(theory_css_ser_interference, m)?)?;
    Ok(())
}
