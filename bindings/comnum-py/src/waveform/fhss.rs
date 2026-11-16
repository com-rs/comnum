//! Frequency-hopping spread spectrum with noncoherent M-FSK, the partial-band noise
//! jammer and the closed forms.

use super::*;
use crate::rng::PyRng;
use comnum::waveform::fhss::{Fhss, theory as fhss_theory};

/// Frequency-hopping spread spectrum with noncoherent :math:`M`-FSK: a band of :math:`Q`
/// channels of :math:`M` orthogonal tones, :math:`N = Q M` samples per symbol, the channel
/// changed every `symbols_per_hop` symbols along a pattern known to both ends; the receiver
/// keeps the :math:`M` bins of the current channel of a DFT.
///
/// The symbol :math:`a` sent while the link sits on the channel :math:`q` is the tone
///
/// .. math::
///
///     x[n] = e^{j 2\pi (qM + a) n / N}, \qquad 0 \le n < N.
///
/// Args:
///     n_channels (int): the channels :math:`Q` of the band.
///     m (int): the tones :math:`M` of a channel, the size of the alphabet.
///     pattern (array_like): the hopping pattern, channels in :math:`[0, Q)`, repeated.
///     symbols_per_hop (int): symbols sent on a channel before the next hop.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> link = comnum.Fhss(16, 2, comnum.Fhss.random_pattern(16, 100, comnum.Rng(1)))
///     >>> x = link.modulate([0, 1, 1, 0])
///     >>> x.size, link.demodulate(x).tolist()
///     (128, [0, 1, 1, 0])
#[pyclass(name = "Fhss", module = "comnum")]
pub struct PyFhss {
    inner: Fhss,
}

#[pymethods]
impl PyFhss {
    #[new]
    #[pyo3(signature = (n_channels, m, pattern, *, symbols_per_hop=1))]
    fn new(n_channels: usize, m: usize, pattern: &Bound<'_, PyAny>, symbols_per_hop: usize) -> PyResult<Self> {
        Ok(PyFhss { inner: Fhss::new(n_channels, m, symbols_per_hop, uvec(pattern)?).py()? })
    }

    /// A pattern of `n` hops drawn uniformly among `n_channels` channels.
    ///
    /// Args:
    ///     n_channels (int): the channels of the band.
    ///     n (int): the number of hops.
    ///     rng (Rng): the generator.
    ///
    /// Returns:
    ///     numpy.ndarray: the `n` channels (int64).
    ///
    /// Raises:
    ///     ValueError: without a channel.
    #[staticmethod]
    fn random_pattern<'py>(
        py: Python<'py>,
        n_channels: usize,
        n: usize,
        mut rng: PyRefMut<'py, PyRng>,
    ) -> PyResult<IArr<'py>> {
        if n_channels == 0 {
            return Err(PyValueError::new_err("at least one channel"));
        }
        Ok(iarr(py, Fhss::random_pattern(n_channels, n, &mut rng.inner)))
    }

    /// int: The channels :math:`Q` of the band.
    #[getter]
    fn n_channels(&self) -> usize {
        self.inner.n_channels
    }

    /// int: The tones :math:`M` of a channel.
    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    /// int: Symbols per hop.
    #[getter]
    fn symbols_per_hop(&self) -> usize {
        self.inner.symbols_per_hop
    }

    /// int: Samples per symbol, :math:`N = Q M`.
    #[getter]
    fn symbol_len(&self) -> usize {
        self.inner.symbol_len()
    }

    /// numpy.ndarray: The hopping pattern.
    #[getter]
    fn pattern<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.pattern.clone())
    }

    /// The unit-envelope samples of a sequence of symbols, the first at the start of
    /// the pattern.
    ///
    /// Args:
    ///     symbols (array_like): integers in :math:`[0, M)`.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`N` samples per symbol.
    ///
    /// Raises:
    ///     ValueError: for a symbol outside the alphabet.
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&uvec(symbols)?).py()?))
    }

    /// The energy of the :math:`M` candidates of every received symbol, in the channel of
    /// the pattern.
    ///
    /// Args:
    ///     samples (array_like): a multiple of :math:`N` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`M` energies per symbol.
    ///
    /// Raises:
    ///     ValueError: unless `samples` holds whole symbols.
    fn energies<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.energies(&cvec(samples)?).py()?))
    }

    /// The noncoherent decisions: the tone of largest energy.
    ///
    /// Args:
    ///     samples (array_like): a multiple of :math:`N` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: one symbol per :math:`N` samples.
    ///
    /// Raises:
    ///     ValueError: unless `samples` holds whole symbols.
    fn demodulate<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, self.inner.demodulate(&cvec(samples)?).py()?))
    }

    fn __repr__(&self) -> String {
        let l = &self.inner;
        format!("Fhss(n_channels={}, m={}, symbols_per_hop={})", l.n_channels, l.m, l.symbols_per_hop)
    }
}

/// A partial-band noise jammer: Gaussian noise of mean power `power` per sample in
/// `n_bins` consecutive bins of DFT blocks of `n_fft` samples, nothing elsewhere.
///
/// In its bins it is white noise of variance
///
/// .. math::
///
///     \sigma_J^2 = \mathrm{power} \cdot n_{\mathrm{fft}} / n_{\mathrm{bins}}
///
/// per sample.
///
/// Args:
///     n (int): the number of samples.
///     n_fft (int): the size of the DFT grid (the `symbol_len` of an `Fhss` link).
///     first_bin (int): the first jammed bin (modulo `n_fft`).
///     n_bins (int): the number of jammed bins, 1 to `n_fft`.
///     power (float): mean power per sample.
///     rng (Rng): the generator.
///
/// Returns:
///     numpy.ndarray: the `n` samples of the jammer.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> j = comnum.partial_band_noise(64 * 500, 64, 16, 16, 2.0, comnum.Rng(1))
///     >>> bool(abs(np.mean(np.abs(j) ** 2) - 2.0) < 0.1)
///     True
#[pyfunction]
fn partial_band_noise<'py>(
    py: Python<'py>,
    n: i64,
    n_fft: usize,
    first_bin: usize,
    n_bins: usize,
    power: f64,
    mut rng: PyRefMut<'py, PyRng>,
) -> PyResult<CArr<'py>> {
    let n = count(n, "n")?;
    Ok(carr(py, comnum::channel::jamming::partial_band_noise(n, n_fft, first_bin, n_bins, power, &mut rng.inner).py()?))
}

/// The bit error rate of slow frequency hopping with noncoherent :math:`M`-FSK under a noise
/// jammer on the fraction `rho` of the band.
///
/// With :math:`P_b(\gamma)` the bit error rate of noncoherent :math:`M`-FSK at
/// :math:`E_b/N_0 = \gamma`,
///
/// .. math::
///
///     \bar{P}_b = (1 - \rho)\, P_b(E_b/N_0) + \rho\, P_b\bigl(E_b/(N_0 + N_J/\rho)\bigr),
///
/// :math:`\tfrac{\rho}{2} \exp(-\rho E_b/(2 N_J))` for binary FSK without thermal noise.
///
/// Args:
///     ebnj_db (float | array_like): :math:`E_b/N_J` (dB), :math:`N_J` the density of the jammer
///         spread over the whole band.
///     rho (float): the jammed fraction of the band, in :math:`(0, 1]`.
///     m (int): the size of the alphabet.
///     ebn0_db (float | None): the thermal :math:`E_b/N_0` (dB); none without thermal noise.
///
/// Returns:
///     float | numpy.ndarray: the bit error rate.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum, math
///     >>> round(comnum.theory_ber_fh_partial_band(10.0, 1.0), 9) == round(0.5 * math.exp(-5), 9)
///     True
#[pyfunction]
#[pyo3(signature = (ebnj_db, rho, *, m=2, ebn0_db=None))]
fn theory_ber_fh_partial_band(
    ebnj_db: &Bound<'_, PyAny>,
    rho: f64,
    m: usize,
    ebn0_db: Option<f64>,
) -> PyResult<Py<PyAny>> {
    map_real(ebnj_db, |e| fhss_theory::ber_partial_band(m, e, rho, ebn0_db).py())
}

/// The worst partial-band noise jammer of slow frequency hopping with noncoherent
/// :math:`M`-FSK: the fraction of the band that maximises the bit error rate, and that
/// error rate (:math:`2 N_J/E_b` and :math:`e^{-1} N_J/E_b` for binary FSK without thermal
/// noise, Viterbi and Jacobs 1975).
///
/// Args:
///     ebnj_db (float): :math:`E_b/N_J` (dB).
///     m (int): the size of the alphabet.
///     ebn0_db (float | None): the thermal :math:`E_b/N_0` (dB); none without thermal noise.
///
/// Returns:
///     tuple[float, float]: the worst fraction and the bit error rate there.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> rho, ber = comnum.fh_worst_case_partial_band(20.0)
///     >>> round(rho, 4), round(ber * 100, 4)
///     (0.02, 0.3679)
#[pyfunction]
#[pyo3(signature = (ebnj_db, *, m=2, ebn0_db=None))]
fn fh_worst_case_partial_band(ebnj_db: f64, m: usize, ebn0_db: Option<f64>) -> PyResult<(f64, f64)> {
    fhss_theory::worst_case_partial_band(m, ebnj_db, ebn0_db).py()
}

/// Register the bindings of this module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFhss>()?;
    m.add_function(wrap_pyfunction!(partial_band_noise, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_fh_partial_band, m)?)?;
    m.add_function(wrap_pyfunction!(fh_worst_case_partial_band, m)?)?;
    Ok(())
}
