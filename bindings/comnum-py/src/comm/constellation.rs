//! Constellations, their mapping and demapping, and the Gray code.

use super::*;
use comnum::math::Jacobian;

/// A constellation with Gray labels: point :math:`i` carries the bits of :math:`i`, MSB first.
///
/// The points and their labels are those of OptiCommPy (`GrayMapping`): square QAM
/// as the product of two PAM listed row by row from the top in a "snake" order, PSK
/// on :math:`e^{j2\pi k/M}`, PAM on :math:`-(M-1), \dots, -1, 1, \dots, M-1`, OOK on
/// :math:`\{0, 1\}`; the :math:`i`-th point of the construction order gets the Gray label
/// `i ^ (i >> 1)`. The grids keep their integer spacing; call `normalized()` for unit
/// mean energy.
///
/// Args:
///     kind (str): `"qam"`, `"psk"`, `"pam"` or `"ook"`.
///     m (int): the order, a power of two (`2` for OOK).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> c = comnum.Constellation("qam", 16).normalized()
///     >>> round(c.energy, 12)
///     1.0
///     >>> bits = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
///     >>> c.demap_bits(c.map_bits(bits)).tolist()
///     [0, 0, 0, 0, 1, 1, 1, 1]
#[pyclass(name = "Constellation", module = "comnum")]
#[derive(Clone)]
pub struct PyConstellation {
    /// The Rust constellation.
    pub inner: Constellation,
}

#[pymethods]
impl PyConstellation {
    #[new]
    fn new(kind: &str, m: usize) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::new(self::kind(kind)?, m).py()? })
    }

    /// A custom constellation from its points, labelled by their index.
    ///
    /// Args:
    ///     points (array_like): the :math:`M` points (complex), :math:`M` a power of two.
    ///
    /// Returns:
    ///     Constellation: the constellation (kind `"custom"`).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comnum
    ///     >>> comnum.Constellation.from_points([1, -1]).order
    ///     2
    #[staticmethod]
    fn from_points(points: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::from_points(cvec(points)?).py()? })
    }

    /// The points of `Constellation(kind, m)` with a set-partitioning labelling
    /// (Ungerboeck): natural labels for PSK, Fig. 2(a) of Caire, Taricco & Biglieri
    /// (1998) for 16-QAM.
    ///
    /// Args:
    ///     kind (str): `"psk"` or `"qam"` (16-QAM only).
    ///     m (int): the order.
    ///
    /// Returns:
    ///     Constellation: the constellation (kind `"custom"`).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    ///
    /// Examples:
    ///     >>> import comnum
    ///     >>> comnum.Constellation.set_partitioning("qam", 16).points[0b1001]
    ///     np.complex128(-3+3j)
    #[staticmethod]
    fn set_partitioning(kind: &str, m: usize) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::set_partitioning(self::kind(kind)?, m).py()? })
    }

    /// A constellation from the name the specifications use (``"16-QAM"``, ``"QPSK"``,
    /// ``"8PSK"``, ``"4-PAM"``, ``"OOK"``; case, dashes and spaces are ignored), with
    /// its integer grid (call `normalized()` for unit energy).
    ///
    /// Args:
    ///     name (str): the name.
    ///
    /// Returns:
    ///     Constellation: the constellation.
    ///
    /// Raises:
    ///     ValueError: for an unknown family or an invalid order.
    ///
    /// Examples:
    ///     >>> import comnum
    ///     >>> c = comnum.Constellation.from_name("16-QAM")
    ///     >>> c.kind, c.order
    ///     ('qam', 16)
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::from_name(name).py()? })
    }

    /// str: The family, `"qam"`, `"psk"`, `"pam"`, `"ook"` or `"custom"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().name()
    }

    /// int: The number of points :math:`M`.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// int: Bits per symbol, :math:`\log_2 M`.
    #[getter]
    fn bits_per_symbol(&self) -> usize {
        self.inner.bits_per_symbol()
    }

    /// numpy.ndarray: The points, in label order (a copy, complex128).
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.points().to_vec())
    }

    /// float: Mean energy of the points under uniform probabilities.
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy()
    }

    /// float: The minimum distance between two points, :math:`\min |s_i - s_j|`: at high
    /// SNR the error rate falls as :math:`Q(d_{\min} / \sqrt{2 N_0})`. :math:`d_{\min}^2 / E_b`
    /// compares the constellations (4 for BPSK and QPSK, 1.6 for 16-QAM).
    #[getter]
    fn min_distance(&self) -> f64 {
        self.inner.min_distance()
    }

    /// Mean energy of the points under the probabilities `px`.
    ///
    /// Args:
    ///     px (array_like): the probabilities of the points.
    ///
    /// Returns:
    ///     float: :math:`\sum_i p_i |s_i|^2`.
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn energy_with(&self, px: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.inner.energy_with(&crate::fvec(px)?).py()
    }

    /// The same constellation scaled to unit mean energy.
    ///
    /// Returns:
    ///     Constellation: the normalised constellation.
    fn normalized(&self) -> Self {
        PyConstellation { inner: self.inner.normalized() }
    }

    /// The same constellation with every point multiplied by `s`.
    ///
    /// Args:
    ///     s (float): the scale.
    ///
    /// Returns:
    ///     Constellation: the scaled constellation.
    fn scaled(&self, s: f64) -> Self {
        PyConstellation { inner: self.inner.scaled(s) }
    }

    /// Symbol indices of a bit sequence (:math:`\log_2 M` bits per symbol, MSB first).
    ///
    /// Args:
    ///     bits (array_like): the bits, a multiple of :math:`\log_2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the indices (int64).
    ///
    /// Raises:
    ///     ValueError: if the length is not a multiple of :math:`\log_2 M`.
    fn indices<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let idx = self.inner.indices(&bvec(bits)?).py()?;
        Ok(iarr(py, idx))
    }

    /// Symbols of a bit sequence (OptiCommPy `modulateGray`).
    ///
    /// Args:
    ///     bits (array_like): the bits, a multiple of :math:`\log_2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: one symbol per :math:`\log_2 M` bits (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn map_bits<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.map_bits(&bvec(bits)?).py()?))
    }

    /// Symbols of an index sequence.
    ///
    /// Args:
    ///     indices (array_like): the indices, below :math:`M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the points (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn symbols<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let idx = uvec(indices)?;
        if let Some(bad) = idx.iter().find(|&&i| i >= self.inner.order()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("index {bad} out of range")));
        }
        Ok(carr(py, self.inner.symbols(&idx)))
    }

    /// Index of the nearest point of every sample (minimum-distance decisions).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the indices (int64).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn decide<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let z = cvec(z)?;
        Ok(iarr(py, self.inner.decide(&z)))
    }

    /// The nearest point of every sample.
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the points (complex128).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn project<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.project(&cvec(z)?)))
    }

    /// Hard-decision bits of every sample (OptiCommPy `demodulateGray`).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`\log_2 M` bits per sample (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn demap_bits<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.demap_bits(&cvec(z)?).into_pyarray(py))
    }

    /// Bit log-likelihood ratios of every sample (OptiCommPy `calcLLR`).
    ///
    /// For every bit :math:`b` of the label, with the priors :math:`p(s)` of the points
    /// and :math:`\sigma^2` the variance of the complex noise:
    ///
    /// .. math::
    ///
    ///     L(b) = \ln \sum_{s:\, b = 0} p(s)\, e^{-|y - s|^2/\sigma^2}
    ///     - \ln \sum_{s:\, b = 1} p(s)\, e^{-|y - s|^2/\sigma^2},
    ///
    /// that is :math:`\ln p(b=0|y) - \ln p(b=1|y)`, positive for a 0. The max-log
    /// approximation keeps the largest term of each sum.
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///     noise_var (float): complex noise variance :math:`\mathbb{E}|w|^2` (for a real
    ///         constellation over a real channel, twice the variance of the real noise).
    ///     px (array_like, optional): prior probabilities of the points.
    ///     max_log (bool): max-log approximation instead of log-sum-exp.
    ///
    /// Returns:
    ///     numpy.ndarray: :math:`\log_2 M` LLRs per sample, MSB first (float64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comnum
    ///     >>> c = comnum.Constellation("qam", 4).normalized()
    ///     >>> llr = c.llr(c.map_bits([0, 1]), 0.1)
    ///     >>> bool(llr[0] > 0 and llr[1] < 0)
    ///     True
    #[pyo3(signature = (z, noise_var, *, px = None, max_log = false))]
    fn llr<'py>(
        &self,
        py: Python<'py>,
        z: &Bound<'py, PyAny>,
        noise_var: f64,
        px: Option<&Bound<'py, PyAny>>,
        max_log: bool,
    ) -> PyResult<FArr<'py>> {
        let z = cvec(z)?;
        let px = px.map(crate::fvec).transpose()?;
        let c = &self.inner;
        let v = py.detach(|| c.llr(&z, noise_var, px.as_deref(), Jacobian::from_max_log(max_log))).py()?;
        Ok(farr(py, v))
    }

    fn __len__(&self) -> usize {
        self.inner.order()
    }

    fn __repr__(&self) -> String {
        format!("comnum.Constellation('{}', {}, energy={:.6})", self.kind(), self.inner.order(), self.inner.energy())
    }
}

/// The reflected binary Gray code of `n_bits` bits: `g[i] = i ^ (i >> 1)`.
///
/// Args:
///     n_bits (int): the number of bits.
///
/// Returns:
///     numpy.ndarray: the :math:`2^{\text{n\_bits}}` code words (int64).
///
/// Raises:
///     ValueError: for more than 24 bits (more than 16 777 216 code words).
///
/// Examples:
///     >>> import comnum
///     >>> comnum.gray_code(2).tolist()
///     [0, 1, 3, 2]
#[pyfunction]
fn gray_code(py: Python<'_>, n_bits: u32) -> PyResult<IArr<'_>> {
    Ok(iarr(py, modulation::gray_code(n_bits).py()?))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConstellation>()?;
    m.add_function(wrap_pyfunction!(gray_code, m)?)?;
    Ok(())
}
