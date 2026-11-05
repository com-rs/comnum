//! The channel codes of CCSDS 131.0-B-5 (TM Synchronization and Channel Coding, 2023):
//! the interleaved Reed–Solomon codeblocks, the turbo codes and the LDPC codes.

use super::*;
use comnum::fec::block::SoftDecode;
use comnum::fec::ccsds::{self, Ar4jaRate, TurboRate};

/// The interleaved Reed–Solomon codeblock of CCSDS 131.0-B-5, Sec. 4: the RS(255, 255 − 2E)
/// code over the field of ``x^8 + x^7 + x^2 + x + 1`` with the roots ``α^(11 j)``,
/// ``j = 128 − E, …, 127 + E``, its symbols in the dual basis of Annex F, ``depth``
/// codewords interleaved symbol by symbol, shortened by a virtual fill of ``fill``
/// symbols per codeword.
///
/// The code works on octets (``encode_bytes``, ``decode_bytes``, dual-basis octets as
/// they are transmitted) and on bits, eight per octet, the most significant first
/// (``encode``, ``decode`` with hard decisions of the LLRs): ``k`` and ``n`` count bits.
///
/// Args:
///     e (int): symbol errors corrected per codeword, 8 or 16.
///     depth (int): the interleaving depth ``I``, 1, 2, 3, 4, 5 or 8.
///     fill (int): the virtual fill ``q`` of every codeword (0: full length).
///
/// Raises:
///     ValueError: for another ``E`` or ``I``, or a fill that leaves no information.
///
/// Examples:
///     The codeblock of ``E = 16`` and ``I = 5`` (Sec. 4.3.8): 1115 octets of frame,
///     160 check octets, 10 200 bits; 16 octets in error in each codeword are corrected.
///
///     >>> import comnum, numpy as np
///     >>> rs = comnum.CcsdsRs(16, 5)
///     >>> (rs.frame_len, rs.codeblock_len, rs.n, rs.k)
///     (1115, 1275, 10200, 8920)
///     >>> frame = np.arange(1115) % 256
///     >>> block = rs.encode_bytes(frame)
///     >>> block[:80] ^= 0xFF  # 80 consecutive octets: 16 in each codeword
///     >>> decoded, corrected = rs.decode_bytes(block)
///     >>> bool(np.array_equal(decoded, frame)), corrected
///     (True, 80)
#[pyclass(name = "CcsdsRs", module = "comnum")]
pub struct PyCcsdsRs {
    inner: ccsds::CcsdsRs,
}

#[pymethods]
impl PyCcsdsRs {
    fn __repr__(&self) -> String {
        format!("CcsdsRs(e={}, depth={}, fill={})", self.inner.e(), self.inner.depth(), self.inner.fill())
    }

    #[new]
    #[pyo3(signature = (e = 16, depth = 1, fill = 0))]
    fn new(e: usize, depth: usize, fill: usize) -> PyResult<Self> {
        Ok(PyCcsdsRs { inner: ccsds::CcsdsRs::new(e, depth, fill).py()? })
    }

    /// int: Symbol errors corrected per codeword ``E``.
    #[getter]
    fn e(&self) -> usize {
        self.inner.e()
    }

    /// int: The interleaving depth ``I``.
    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// int: The virtual fill ``q`` of a codeword.
    #[getter]
    fn fill(&self) -> usize {
        self.inner.fill()
    }

    /// int: Octets of a frame, ``(255 − 2E − q) I``.
    #[getter]
    fn frame_len(&self) -> usize {
        self.inner.frame_len()
    }

    /// int: Octets of a codeblock, ``(255 − q) I``.
    #[getter]
    fn codeblock_len(&self) -> usize {
        self.inner.codeblock_len()
    }

    /// int: Information bits per codeblock, ``8 frame_len``.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Coded bits per codeblock, ``8 codeblock_len``.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// The codeblock of a frame: the frame, then the ``2 E I`` check octets.
    ///
    /// Args:
    ///     frame (array_like): ``frame_len`` octets (dual basis).
    ///
    /// Returns:
    ///     numpy.ndarray: ``codeblock_len`` octets (uint8).
    ///
    /// Raises:
    ///     ValueError: for a frame of another length or a value above 255.
    fn encode_bytes<'py>(&self, py: Python<'py>, frame: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        let f = crate::bytevec(frame)?;
        Ok(self.inner.encode_bytes(&f).py()?.into_pyarray(py))
    }

    /// Decode a received codeblock.
    ///
    /// Args:
    ///     block (array_like): ``codeblock_len`` octets.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int | None]: the frame and the number of corrected
    ///     octets (``None`` if a codeword could not be decoded: its octets are then those
    ///     received).
    ///
    /// Raises:
    ///     ValueError: for a codeblock of another length or a value above 255.
    fn decode_bytes<'py>(&self, py: Python<'py>, block: &Bound<'py, PyAny>) -> PyResult<(BArr<'py>, Option<usize>)> {
        let b = crate::bytevec(block)?;
        let (frame, fixed) = self.inner.decode_bytes(&b).py()?;
        Ok((frame.into_pyarray(py), fixed))
    }

    /// Encode whole codeblocks of bits.
    ///
    /// Args:
    ///     bits (array_like): a multiple of ``k`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n`` bits per codeblock (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of bits of the expected shape.
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole codeblocks of LLRs (hard decisions, then the algebraic decoder).
    ///
    /// Args:
    ///     llr (array_like): a multiple of ``n`` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: ``k`` bits per codeblock (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// An LDPC code of CCSDS 131.0-B-5, Sec. 7: the (8160, 7136) code of rate 223/255 (Sec.
/// 7.3, ``CcsdsLdpc("7/8")``), or an AR4JA code of ``k`` = 1024, 4096 or 16384 bits and
/// rate 1/2, 2/3 or 4/5 (Sec. 7.4, ``CcsdsLdpc("1/2", 1024)``), with the encoder of the
/// standard (the circulants of Table C-1, the matrix ``C`` of Sec. 7.4.2), the virtual
/// zeros and the punctured bits, decoded by belief propagation on the full parity-check
/// matrix.
///
/// Building the AR4JA code of ``k = 16384`` and rate 1/2 inverts an 8192 × 8192 binary
/// matrix: about a second.
///
/// Args:
///     rate (str): ``"7/8"`` (the (8160, 7136) code), ``"1/2"``, ``"2/3"`` or ``"4/5"``.
///     k (int | None): the information bits of an AR4JA code (1024, 4096 or 16384).
///     rule (str): the check-node rule (as ``LdpcCode.set_decoder``).
///     alpha (float): the scale or the offset of the min-sum rules.
///     max_iterations (int): the maximum number of iterations.
///     layered (bool): the layered schedule (default), or flooding (``False``).
///
/// Raises:
///     ValueError: for another rate or ``k``.
///
/// Examples:
///     The lengths of Table 7-5 and a codeword decoded through a noisy channel:
///
///     >>> import comnum, numpy as np
///     >>> code = comnum.CcsdsLdpc("2/3", 1024)
///     >>> (code.k, code.n)
///     (1024, 1536)
///     >>> b = comnum.bits(1024, comnum.Rng(1))
///     >>> bool(np.array_equal(code.decode(4.0 - 8.0 * code.encode(b)), b))
///     True
///     >>> comnum.CcsdsLdpc("7/8").n
///     8160
#[pyclass(name = "CcsdsLdpc", module = "comnum")]
pub struct PyCcsdsLdpc {
    inner: ccsds::CcsdsLdpc,
    rate: String,
}

#[pymethods]
impl PyCcsdsLdpc {
    fn __repr__(&self) -> String {
        format!("CcsdsLdpc(rate={:?}, k={}, n={})", self.rate, self.inner.k(), self.inner.n())
    }

    #[new]
    #[pyo3(signature = (
        rate,
        k = None,
        *,
        rule = ldpc::DEFAULT_RULE.name(),
        alpha = ldpc::DEFAULT_ALPHA,
        max_iterations = ldpc::DEFAULT_MAX_ITERATIONS,
        layered = ldpc::DEFAULT_LAYERED,
    ))]
    fn new(
        rate: &str,
        k: Option<usize>,
        rule: &str,
        alpha: f64,
        max_iterations: usize,
        layered: bool,
    ) -> PyResult<Self> {
        let mut inner = match (rate, k) {
            ("7/8", None | Some(7136)) => ccsds::CcsdsLdpc::c2().py()?,
            ("7/8", Some(k)) => {
                return Err(PyValueError::new_err(format!("the rate-7/8 code has k = 7136, got {k}")));
            }
            (r, Some(k)) => ccsds::CcsdsLdpc::ar4ja(k, r.parse::<Ar4jaRate>().py()?).py()?,
            (_, None) => return Err(PyValueError::new_err("an AR4JA code needs k (1024, 4096 or 16384)")),
        };
        let dec = &mut inner.ldpc_mut().decoder;
        dec.rule = CheckRule::from_name(rule, alpha).py()?;
        dec.max_iterations = max_iterations;
        dec.layered = layered;
        Ok(PyCcsdsLdpc { inner, rate: rate.to_owned() })
    }

    /// str: The rate, ``"7/8"``, ``"1/2"``, ``"2/3"`` or ``"4/5"``.
    #[getter]
    fn rate(&self) -> &str {
        &self.rate
    }

    /// int: Information bits per codeword.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Transmitted bits per codeword (Table 7-5; 8160 for the rate-7/8 code).
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// LdpcCode: The code of the full parity-check matrix (the virtual and punctured bits
    /// included), a copy.
    #[getter]
    fn ldpc(&self) -> super::ldpc_code::PyLdpc {
        super::ldpc_code::PyLdpc { inner: self.inner.ldpc().clone() }
    }

    /// The codeword of the full parity-check matrix: the 18 virtual zeros of the rate-7/8
    /// code, the information and every parity bit, the punctured ones included.
    ///
    /// Args:
    ///     info (array_like): ``k`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: the bits (uint8), a codeword of :attr:`ldpc`.
    ///
    /// Raises:
    ///     ValueError: unless ``info`` has ``k`` bits.
    fn full_codeword<'py>(&self, py: Python<'py>, info: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        let b = bvec(info)?;
        Ok(self.inner.full_codeword(&b).py()?.into_pyarray(py))
    }

    /// Encode whole codewords.
    ///
    /// Args:
    ///     bits (array_like): a multiple of ``k`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n`` bits per codeword (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of bits of the expected shape.
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole codewords of LLRs.
    ///
    /// Args:
    ///     llr (array_like): a multiple of ``n`` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: ``k`` bits per codeword (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }

    /// The a posteriori LLRs of the information bits of one codeword.
    ///
    /// Args:
    ///     llr (array_like): ``n`` LLRs.
    ///
    /// Returns:
    ///     numpy.ndarray: ``k`` LLRs (float64).
    ///
    /// Raises:
    ///     ValueError: unless ``llr`` has ``n`` values.
    fn app<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        let l = fvec(llr)?;
        Ok(farr(py, py.detach(|| self.inner.app(&l, None)).py()?))
    }
}

/// A turbo code of CCSDS 131.0-B-5, Sec. 6: two 16-state recursive encoders
/// (``G0 = 23``, ``G1 = 33``, ``G2 = 25``, ``G3 = 37`` in octal), the permutation of
/// Sec. 6.3 g) for ``k`` = 1784, 3568, 7136 or 8920 bits, 4 termination steps, the
/// outputs of Sec. 6.4 at rates 1/2, 1/3, 1/4 and 1/6, ``n = (k + 4)/r`` (Table 6-2);
/// decoded by iterative log-MAP decoding.
///
/// Args:
///     k (int): information bits per codeword.
///     rate (str): ``"1/2"``, ``"1/3"``, ``"1/4"`` or ``"1/6"``.
///     iterations (int): decoding iterations.
///     max_log (bool): max-log-MAP instead of log-MAP.
///
/// Raises:
///     ValueError: for another ``k`` or rate.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> code = comnum.CcsdsTurbo(1784, "1/3")
///     >>> (code.k, code.n)
///     (1784, 5364)
///     >>> b = comnum.bits(1784, comnum.Rng(2))
///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
///     True
#[pyclass(name = "CcsdsTurbo", module = "comnum")]
pub struct PyCcsdsTurbo {
    inner: ccsds::CcsdsTurbo,
}

#[pymethods]
impl PyCcsdsTurbo {
    fn __repr__(&self) -> String {
        format!(
            "CcsdsTurbo(k={}, rate={:?}, iterations={})",
            self.inner.k(),
            self.inner.rate().name(),
            self.inner.decoder.iterations
        )
    }

    #[new]
    #[pyo3(signature = (k, rate = "1/3", *, iterations = 8, max_log = false))]
    fn new(k: usize, rate: &str, iterations: usize, max_log: bool) -> PyResult<Self> {
        let mut inner = ccsds::CcsdsTurbo::new(k, rate.parse::<TurboRate>().py()?).py()?;
        inner.decoder.iterations = iterations;
        inner.decoder.jacobian = Jacobian::from_max_log(max_log);
        Ok(PyCcsdsTurbo { inner })
    }

    /// str: The rate.
    #[getter]
    fn rate(&self) -> &'static str {
        self.inner.rate().name()
    }

    /// int: Information bits per codeword.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Transmitted bits per codeword, ``(k + 4)/r``.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// numpy.ndarray: The permutation of the second encoder, ``π(s)`` for
    /// ``s = 0, …, k − 1`` (int64): the encoder b reads the bit ``π(s)`` at step ``s``.
    #[getter]
    fn permutation<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.interleaver().perm().to_vec())
    }

    /// Encode whole codewords.
    ///
    /// Args:
    ///     bits (array_like): a multiple of ``k`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n`` bits per codeword (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of bits of the expected shape.
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole codewords of LLRs.
    ///
    /// Args:
    ///     llr (array_like): a multiple of ``n`` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: ``k`` bits per codeword (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCcsdsRs>()?;
    m.add_class::<PyCcsdsLdpc>()?;
    m.add_class::<PyCcsdsTurbo>()?;
    Ok(())
}
