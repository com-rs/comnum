//! LDPC codes: progressive edge growth, a parity-check matrix, belief propagation.

use super::*;

/// An LDPC code: `LdpcCode.peg(n, k, var_degree=…, seed=…)` (progressive edge growth)
/// or `LdpcCode(n, rows)` from the rows of a parity-check matrix.
///
/// The codes of [1]_, built by the progressive edge growth of [2]_ or from a matrix.
/// Decoded by belief propagation (see `set_decoder`): `rule` is `"sum_product"` (the
/// default, as in the specifications), `"min_sum"`, `"normalized_min_sum"` (scaled by
/// `alpha`, 0.8 by default, [3]_) or `"offset_min_sum"` (offset `alpha`); `layered`
/// selects the layered schedule ([4]_, the default); 50 iterations at most by default.
/// The PEG construction uses the variable degree 3 and the seed 1 by default.
///
/// Args:
///     n (int): coded bits per block.
///     rows (list[list[int]]): the positions of the ones of every row of the
///         parity-check matrix.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// References:
///     .. [1] R. G. Gallager, "Low-density parity-check codes," *IRE Trans. Inf. Theory*,
///        8(1), 21–28, 1962.
///     .. [2] X.-Y. Hu, E. Eleftheriou and D. M. Arnold, "Regular and irregular
///        progressive edge-growth Tanner graphs," *IEEE Trans. Inf. Theory*, 51(1),
///        386–398, 2005.
///     .. [3] J. Chen and M. P. C. Fossorier, "Near optimum universal belief propagation
///        based decoding of low-density parity check codes," *IEEE Trans. Commun.*,
///        50(3), 406–414, 2002.
///     .. [4] D. E. Hocevar, "A reduced complexity decoder architecture via layered
///        decoding of LDPC codes," *Proc. IEEE SiPS*, 107–112, 2004.
#[pyclass(name = "LdpcCode", module = "comnum")]
pub struct PyLdpc {
    pub(crate) inner: LdpcCode,
}

#[pymethods]
impl PyLdpc {
    fn __repr__(&self) -> String {
        format!("LdpcCode(n={}, k={})", self.inner.n(), self.inner.k())
    }

    #[new]
    fn new(n: usize, rows: Vec<Vec<usize>>) -> PyResult<Self> {
        Ok(PyLdpc { inner: LdpcCode::from_rows(n, rows).py()? })
    }

    /// A code built by progressive edge growth.
    ///
    /// Args:
    ///     n (int): coded bits per block.
    ///     k (int): information bits per block.
    ///     var_degree (int): the degree of every variable node.
    ///     seed (int): the seed of the construction.
    ///
    /// Returns:
    ///     LdpcCode: the code.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    ///
    /// Examples:
    ///     >>> import comnum
    ///     >>> code = comnum.LdpcCode.peg(96, 48)
    ///     >>> code.n, code.k, code.girth() >= 6
    ///     (96, 48, True)
    #[staticmethod]
    #[pyo3(signature = (n, k, *, var_degree=ldpc::DEFAULT_VAR_DEGREE, seed=ldpc::DEFAULT_PEG_SEED))]
    fn peg(n: usize, k: usize, var_degree: usize, seed: u64) -> PyResult<Self> {
        Ok(PyLdpc { inner: LdpcCode::peg(n, k, var_degree, seed).py()? })
    }

    /// Choose the decoder.
    ///
    /// A variable node sends the sum of its channel LLR and of the other incoming
    /// messages; a check node sends
    ///
    /// .. math::
    ///
    ///     L_{c \to v} = 2 \operatorname{atanh} \prod_{v' \ne v} \tanh\frac{L_{v' \to c}}{2}
    ///     \quad \text{(sum-product)}, \qquad
    ///     L_{c \to v} \approx \alpha \prod_{v' \ne v} \operatorname{sgn} L_{v' \to c}
    ///     \min_{v' \ne v} |L_{v' \to c}| \quad \text{(min-sum)},
    ///
    /// with the normalisation :math:`\alpha` (Chen & Fossorier 2002) or an offset.
    ///
    /// Args:
    ///     rule (str): the check-node rule, ``"sum_product"``, ``"min_sum"``,
    ///         ``"normalized_min_sum"`` (scaled by ``alpha``) or ``"offset_min_sum"``
    ///         (with the offset ``alpha``), as the specifications.
    ///     alpha (float): the scale or the offset of the min-sum rules.
    ///     layered (bool): the layered schedule (else flooding).
    ///     max_iterations (int): iterations at most.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    #[pyo3(signature = (
        rule=ldpc::DEFAULT_RULE.name(),
        *,
        alpha=ldpc::DEFAULT_ALPHA,
        layered=ldpc::DEFAULT_LAYERED,
        max_iterations=ldpc::DEFAULT_MAX_ITERATIONS,
    ))]
    fn set_decoder(&mut self, rule: &str, alpha: f64, layered: bool, max_iterations: usize) -> PyResult<()> {
        self.inner.decoder.rule = CheckRule::from_name(rule, alpha).py()?;
        self.inner.decoder.layered = layered;
        self.inner.decoder.max_iterations = max_iterations;
        Ok(())
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Coded bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// float: Code rate.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// list[list[int]]: The rows of the parity-check matrix (the positions of their
    /// ones).
    #[getter]
    fn rows(&self) -> Vec<Vec<usize>> {
        self.inner.rows().to_vec()
    }

    /// The girth of the Tanner graph.
    ///
    /// Returns:
    ///     int: the length of its shortest cycle.
    fn girth(&self) -> usize {
        self.inner.girth()
    }

    /// Whether ``H c = 0``.
    ///
    /// Args:
    ///     c (array_like): ``n`` bits.
    ///
    /// Returns:
    ///     bool: every parity check holds.
    ///
    /// Raises:
    ///     ValueError: unless ``c`` has ``n`` bits.
    fn is_codeword(&self, c: &Bound<'_, PyAny>) -> PyResult<bool> {
        let b = bvec(c)?;
        if b.len() != self.inner.n() {
            return Err(PyValueError::new_err(format!("{} bits for a code of length {}", b.len(), self.inner.n())));
        }
        Ok(self.inner.is_codeword(&b))
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` coded bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    ///
    /// Examples:
    ///     >>> import comnum
    ///     >>> comnum.LdpcCode.peg(96, 48).encode(comnum.bits(96, comnum.Rng(1))).size
    ///     192
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (belief propagation).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    ///
    /// Examples:
    ///     >>> import comnum, numpy as np
    ///     >>> code = comnum.LdpcCode.peg(96, 48)
    ///     >>> b = comnum.bits(48, comnum.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }

    /// Decode one block into the a posteriori LLRs of the coded bits.
    ///
    /// Args:
    ///     llr (array_like): `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int]: the `n` a posteriori LLRs and the iterations run.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn decode_llr<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<(FArr<'py>, usize)> {
        let l = fvec(llr)?;
        let (post, it) = py.detach(|| self.inner.decode_llr(&l)).py()?;
        Ok((farr(py, post), it))
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLdpc>()?;
    Ok(())
}
