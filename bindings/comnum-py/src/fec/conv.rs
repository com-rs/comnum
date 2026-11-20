//! The convolutional code (Viterbi) and the turbo code (iterative BCJR), punctured or not.

use super::*;

/// A terminated rate-1/n convolutional code on blocks of `k` bits, decoded by Viterbi:
/// the (133, 171) code of constraint length 7 by default, punctured by `puncture`.
///
/// The decoder is the Viterbi algorithm of [1]_, the puncturing that of [2]_: `puncture`
/// is the name of a pattern of a standard (`"ieee80211_2_3"`, `"ieee80211_3_4"`,
/// `"ieee80211_5_6"` of [3]_, Sec. 17.3.5.6 and 19.3.11.6; `"dvbs_2_3"`, `"dvbs_3_4"`,
/// `"dvbs_5_6"`, `"dvbs_7_8"` of [4]_, Table 1) or the rows of a pattern, row `i` the
/// output `i` (`[[1, 1, 0], [1, 0, 1]]`). The receiver puts a zero LLR at every
/// punctured position.
///
/// The code follows the contract of every code of the package: `encode` maps whole
/// blocks of `k` bits to `n` coded bits, `decode` whole blocks of `n` LLRs (positive
/// for a 0) to `k` bits.
///
/// Args:
///     k (int): information bits per block.
///     generators (list[int]): octal generators (write them as `0o133`).
///     constraint (int): constraint length `K`.
///     hard_decision (bool): decode the signs of the LLRs only.
///     puncture (str | list[list[int]] | None): the puncturing pattern.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically
///        optimum decoding algorithm," *IEEE Trans. Inf. Theory*, 13(2), 260–269, 1967.
///     .. [2] J. B. Cain, G. C. Clark and J. M. Geist, "Punctured convolutional codes of
///        rate :math:`(n-1)/n` and simplified maximum likelihood decoding," *IEEE Trans.
///        Inf. Theory*, 25(1), 97–100, 1979.
///     .. [3] IEEE Std 802.11-2020.
///     .. [4] ETSI EN 300 421 V1.1.2, *Digital Video Broadcasting (DVB); Framing
///        structure, channel coding and modulation for 11/12 GHz satellite services*,
///        1997.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> code = comnum.ConvolutionalCode(100)
///     >>> b = comnum.bits(100, comnum.Rng(1))
///     >>> c = code.encode(b)
///     >>> llr = 4.0 * (1.0 - 2.0 * c)
///     >>> bool(np.array_equal(code.decode(llr), b))
///     True
///     >>> punctured = comnum.ConvolutionalCode(100, puncture="ieee80211_3_4")
///     >>> (punctured.n, round(punctured.rate, 3))
///     (142, 0.704)
///     >>> bool(np.array_equal(punctured.decode(4.0 * (1.0 - 2.0 * punctured.encode(b))), b))
///     True
#[pyclass(name = "ConvolutionalCode", module = "comnum")]
pub struct PyConv {
    inner: Punctured<ConvolutionalCode>,
    punctured: bool,
}

#[pymethods]
impl PyConv {
    fn __repr__(&self) -> String {
        format!(
            "ConvolutionalCode(k={}, n={}, constraint={}, punctured={})",
            self.inner.k(),
            self.inner.n(),
            self.inner.code().trellis().constraint(),
            py_bool(self.punctured)
        )
    }

    #[new]
    #[pyo3(signature = (k, *, generators=vec![0o133, 0o171], constraint=7, hard_decision=false, puncture=None))]
    fn new(
        k: usize,
        generators: Vec<u32>,
        constraint: u32,
        hard_decision: bool,
        puncture: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut code = ConvolutionalCode::new(Trellis::feedforward(constraint, &generators).py()?, k).py()?;
        code.hard_decision = hard_decision;
        let p = match puncture {
            Some(p) => puncturer(p)?,
            None => Puncturer::identity(code.trellis().n_out()).py()?,
        };
        Ok(PyConv { inner: Punctured::new(code, p).py()?, punctured: puncture.is_some() })
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Coded bits per block (tail included, after puncturing).
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// float: Code rate `k / n` (after puncturing).
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// list[list[int]] | None: The rows of the puncturing pattern (row `i` the output
    /// `i`), `None` without puncturing.
    #[getter]
    fn puncture(&self) -> Option<Vec<Vec<u32>>> {
        self.punctured.then(|| pattern_rows(self.inner.puncturer()))
    }

    /// The free distance of the (punctured) code (see :func:`free_distance`).
    ///
    /// Returns:
    ///     int: the free distance.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn free_distance(&self) -> PyResult<u32> {
        comnum::fec::puncture::free_distance(self.inner.code().trellis(), self.inner.puncturer()).py()
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
    ///     >>> comnum.ConvolutionalCode(10).encode(comnum.bits(20, comnum.Rng(1))).size
    ///     64
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (Viterbi).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0; 0 at the
    ///         punctured positions is added by the code).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    ///
    /// Examples:
    ///     >>> import comnum, numpy as np
    ///     >>> code = comnum.ConvolutionalCode(10)
    ///     >>> b = comnum.bits(20, comnum.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(1.0 - 2.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }

    /// Decode whole blocks of LLRs with the soft-output Viterbi algorithm (Hagenauer &
    /// Hoeher 1989): the decisions of Viterbi with, as LLR, the metric difference with
    /// the best competing survivor that decides differently (positive for a 0; at least
    /// the magnitude of the max-log-MAP LLR).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0; 0 at the
    ///         punctured positions is added by the code).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` LLRs of the information bits per block.
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    ///
    /// Examples:
    ///     >>> import comnum, numpy as np
    ///     >>> code = comnum.ConvolutionalCode(10)
    ///     >>> b = comnum.bits(20, comnum.Rng(1))
    ///     >>> soft = code.soft_decode(1.0 - 2.0 * code.encode(b))
    ///     >>> bool(np.array_equal((soft < 0).astype(np.uint8), b))
    ///     True
    fn soft_decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        let l = fvec(llr)?;
        let (code, punct) = (self.inner.code(), self.inner.puncturer());
        let n_tx = self.inner.n();
        if n_tx == 0 || l.len() % n_tx != 0 {
            return Err(PyValueError::new_err(format!("{} LLRs is not a multiple of the {n_tx} of a block", l.len())));
        }
        let n_coded = code.n();
        let (trellis, k) = (code.trellis(), code.k());
        let out = py.detach(|| -> comnum::Result<Vec<f64>> {
            let mut out = Vec::with_capacity(l.len() / n_tx * k);
            for block in l.chunks_exact(n_tx) {
                let full = punct.depuncture(block, n_coded)?;
                let soft = comnum::fec::conv::sova(trellis, &full, true)?;
                out.extend_from_slice(&soft[..k]);
            }
            Ok(out)
        });
        Ok(farr(py, out.py()?))
    }
}

/// The turbo code of LTE: two (13, 15) recursive systematic codes in parallel, rate
/// about 1/3, decoded by iterative log-MAP.
///
/// The code of [1]_ with the constituent codes of [2]_. The interleaver is the QPP
/// `(f1, f2)` [3]_ if given (`"lte"`: the coefficients of LTE for `k`, [2]_
/// Table 5.1.3-3), else a random permutation.
/// `puncture` punctures the systematic and the two parity streams by a pattern of
/// three rows applied at every information index (the tail bits are kept): the name
/// `"turbo_1_2"` (systematic 1 1, parity 1: 1 0, parity 2: 0 1, the rate-1/2 code of
/// [1]_, Sec. II) or the rows of a pattern.
///
/// Args:
///     k (int): information bits per block.
///     iterations (int): decoding iterations.
///     max_log (bool): max-log-MAP (else log-MAP, the exact Jacobian logarithm).
///     qpp (str | tuple[int, int] | None): the QPP interleaver `(f1, f2)`, or `"lte"`.
///     seed (int): seed of the random interleaver.
///     puncture (str | list[list[int]] | None): the puncturing pattern (three rows).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] C. Berrou, A. Glavieux and P. Thitimajshima, "Near Shannon limit
///        error-correcting coding and decoding: turbo-codes," *Proc. IEEE ICC*,
///        1064–1070, 1993.
///     .. [2] 3GPP TS 36.212, *E-UTRA; Multiplexing and channel coding*.
///     .. [3] J. Sun and O. Y. Takeshita, "Interleavers for turbo codes using permutation
///        polynomials over integer rings," *IEEE Trans. Inf. Theory*, 51(1), 101–119,
///        2005.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> code = comnum.TurboCode(40, qpp=(3, 10), puncture="turbo_1_2")
///     >>> (code.n, round(code.rate, 3))
///     (92, 0.435)
#[pyclass(name = "TurboCode", module = "comnum")]
pub struct PyTurbo {
    inner: Turbo,
}

/// A turbo code, punctured or not (the unpunctured codeword keeps its stream order).
enum Turbo {
    Plain(TurboCode),
    Punctured(PuncturedTurbo),
}

impl Turbo {
    fn code(&self) -> &TurboCode {
        match self {
            Turbo::Plain(c) => c,
            Turbo::Punctured(p) => p.code(),
        }
    }

    fn decoder_mut(&mut self) -> &mut TurboDecoder {
        match self {
            Turbo::Plain(c) => &mut c.decoder,
            Turbo::Punctured(p) => p.decoder_mut(),
        }
    }

    fn block(&self) -> &(dyn BlockCode + Sync) {
        match self {
            Turbo::Plain(c) => c,
            Turbo::Punctured(p) => p,
        }
    }
}

#[pymethods]
impl PyTurbo {
    fn __repr__(&self) -> String {
        format!(
            "TurboCode(k={}, n={}, iterations={}, punctured={})",
            self.inner.block().k(),
            self.inner.block().n(),
            self.inner.code().decoder.iterations,
            py_bool(matches!(self.inner, Turbo::Punctured(_)))
        )
    }

    #[new]
    #[pyo3(signature = (k, *, iterations=8, max_log=false, qpp=None, seed=7, puncture=None))]
    fn new(
        k: usize,
        iterations: usize,
        max_log: bool,
        qpp: Option<&Bound<'_, PyAny>>,
        seed: u64,
        puncture: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let il = match qpp {
            Some(q) if q.is_instance_of::<PyString>() => Interleaver::from_name(&q.extract::<String>()?, k).py()?,
            Some(q) => {
                let (f1, f2): (usize, usize) =
                    q.extract().map_err(|e| extract_error(q.py(), &e, "qpp is \"lte\" or a pair (f1, f2)"))?;
                Interleaver::qpp(k, f1, f2).py()?
            }
            None => Interleaver::random(k, seed),
        };
        let mut code = TurboCode::with_interleaver(il).py()?;
        code.decoder.iterations = iterations;
        code.decoder.jacobian = Jacobian::from_max_log(max_log);
        let inner = match puncture {
            Some(p) => Turbo::Punctured(PuncturedTurbo::new(code, puncturer(p)?).py()?),
            None => Turbo::Plain(code),
        };
        Ok(PyTurbo { inner })
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.block().k()
    }

    /// int: Coded bits per block (`3k + 12` without puncturing).
    #[getter]
    fn n(&self) -> usize {
        self.inner.block().n()
    }

    /// float: Code rate (after puncturing).
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.block().code_rate()
    }

    /// list[list[int]] | None: The rows of the puncturing pattern (systematic, parity
    /// 1, parity 2), `None` without puncturing.
    #[getter]
    fn puncture(&self) -> Option<Vec<Vec<u32>>> {
        match &self.inner {
            Turbo::Plain(_) => None,
            Turbo::Punctured(p) => Some(pattern_rows(p.puncturer())),
        }
    }

    /// int: Decoding iterations (read and write).
    #[getter]
    fn get_iterations(&self) -> usize {
        self.inner.code().decoder.iterations
    }

    #[setter]
    fn set_iterations(&mut self, n: usize) {
        self.inner.decoder_mut().iterations = n;
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
    ///     >>> comnum.TurboCode(40).encode(comnum.bits(80, comnum.Rng(1))).size
    ///     264
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, self.inner.block(), bits)
    }

    /// Decode whole blocks of LLRs (iterative log-MAP).
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
    ///     >>> code = comnum.TurboCode(40)
    ///     >>> b = comnum.bits(40, comnum.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, self.inner.block(), llr)
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConv>()?;
    m.add_class::<PyTurbo>()?;
    Ok(())
}
