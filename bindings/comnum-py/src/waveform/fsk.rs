//! Orthogonal M-FSK with a noncoherent receiver.

use super::*;
use comnum::waveform::fsk::Fsk;

/// Orthogonal :math:`M`-FSK with a continuous phase and a noncoherent receiver.
///
/// The symbol :math:`a` is the tone of :math:`a \Delta` cycles per symbol,
/// :math:`\Delta` the `spacing`, held for :math:`N` = `sps` samples:
///
/// .. math::
///
///     x[n] = e^{j 2\pi a \Delta n / N}, \qquad 0 \le n < N.
///
/// The receiver correlates every symbol with the :math:`M` tones and needs neither the
/// phase nor the exact frequency of the carrier; its error rate over white noise is
/// `theory_ber_fsk_noncoherent`.
///
/// Args:
///     m (int): the number of tones.
///     sps (int): samples per symbol.
///     spacing (int): cycles per symbol between two adjacent tones
///         (:math:`m \cdot \mathrm{spacing} \le \mathrm{sps}`).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> fsk = comnum.Fsk(4, 16, spacing=2)
///     >>> x = fsk.modulate([0, 3, 1, 2])
///     >>> x.size, fsk.demodulate(x).tolist()
///     (64, [0, 3, 1, 2])
#[pyclass(name = "Fsk", module = "comnum")]
pub struct PyFsk {
    inner: Fsk,
}

#[pymethods]
impl PyFsk {
    #[new]
    #[pyo3(signature = (m, sps, *, spacing=1))]
    fn new(m: usize, sps: usize, spacing: usize) -> PyResult<Self> {
        Ok(PyFsk { inner: Fsk::new(m, sps, spacing).py()? })
    }

    /// int: The number of tones :math:`M`.
    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    /// int: Samples per symbol.
    #[getter]
    fn sps(&self) -> usize {
        self.inner.sps
    }

    /// int: Cycles per symbol between two adjacent tones.
    #[getter]
    fn spacing(&self) -> usize {
        self.inner.spacing
    }

    /// The unit-envelope samples of a sequence of symbols.
    ///
    /// Args:
    ///     symbols (array_like): integers in :math:`[0, M)`.
    ///
    /// Returns:
    ///     numpy.ndarray: `sps` samples per symbol.
    ///
    /// Raises:
    ///     ValueError: for a symbol outside the alphabet.
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&uvec(symbols)?).py()?))
    }

    /// The energy of the :math:`M` tones in every whole symbol, the first symbol at the
    /// first sample.
    ///
    /// Args:
    ///     samples (array_like): the received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`M` energies per symbol.
    ///
    /// Raises:
    ///     ValueError: for an input that is not a 1-D complex array.
    fn energies<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.energies(&cvec(samples)?).py()?))
    }

    /// The noncoherent decisions: the tone of largest energy in every whole symbol.
    ///
    /// Args:
    ///     samples (array_like): the received samples, the first symbol at the first sample.
    ///
    /// Returns:
    ///     numpy.ndarray: one symbol per `sps` samples.
    ///
    /// Raises:
    ///     ValueError: for an input that is not a 1-D complex array.
    fn demodulate<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, self.inner.demodulate(&cvec(samples)?).py()?))
    }

    /// The offset in :math:`[0, \mathrm{sps})` at which the symbols start: the one that
    /// maximises the energy of the decisions, to within one sample. Give it the part of a
    /// recording that holds the signal.
    ///
    /// Args:
    ///     samples (array_like): the received samples.
    ///
    /// Returns:
    ///     int: the offset of the first whole symbol.
    ///
    /// Raises:
    ///     ValueError: for fewer samples than two symbols.
    fn timing(&self, samples: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.inner.timing(&cvec(samples)?).py()
    }

    fn __repr__(&self) -> String {
        format!("Fsk(m={}, sps={}, spacing={})", self.inner.m, self.inner.sps, self.inner.spacing)
    }
}

/// The bit error rate of :math:`M` orthogonal signals detected noncoherently over white
/// noise (noncoherent :math:`M`-FSK; Proakis & Salehi, Sec. 4.5).
///
/// The magnitude of the matched filter of the sent signal is Rice of location
/// :math:`v = \sqrt{2 E_s/N_0}`, :math:`E_s = E_b \log_2 M`, the :math:`M - 1` others
/// Rayleigh, and the largest wins:
///
/// .. math::
///
///     \begin{aligned}
///     P_{s} &= \int_{0}^{\infty} y\, e^{-(y^{2} + v^{2})/2} I_{0}(v y)
///     \left[1 - \left(1 - e^{-y^{2}/2}\right)^{M - 1}\right] dy, \\
///     P_b &= P_s \frac{M}{2(M - 1)},
///     \end{aligned}
///
/// every wrong symbol being equally likely; for :math:`M = 2`,
/// :math:`P_b = \tfrac{1}{2} \exp(-E_b/(2 N_0))`.
///
/// Args:
///     m (int): the number of tones.
///     ebn0_db (float | array_like): :math:`E_b/N_0` (dB).
///
/// Returns:
///     float | numpy.ndarray: the bit error rate.
///
/// Raises:
///     ValueError: for fewer than two tones.
///
/// Examples:
///     >>> import comnum, math
///     >>> round(comnum.theory_ber_fsk_noncoherent(2, 10.0), 9) == round(0.5 * math.exp(-5), 9)
///     True
#[pyfunction]
fn theory_ber_fsk_noncoherent(m: usize, ebn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let bits = comnum::math::lin2db((m.max(2) as f64).log2());
    map_real(ebn0_db, |e| comnum::mapping::theory::ber_orthogonal_noncoherent(m, e + bits).py())
}

/// Register the bindings of this module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFsk>()?;
    m.add_function(wrap_pyfunction!(theory_ber_fsk_noncoherent, m)?)
}
