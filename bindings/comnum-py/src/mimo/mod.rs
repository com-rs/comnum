//! MIMO: channels, detection, space-time codes, precoding and capacity.
//!
//! Channel matrices are `(n_rx, n_tx)` arrays, or `(n, n_rx, n_tx)` stacks with one
//! matrix per channel use; the received streams are stream-major `(n_rx, n)` arrays
//! (one column per channel use), and the decisions `(n_tx, n)`, as every signal of the
//! package.

use crate::comm::PyConstellation;
use crate::rng::PyRng;
use crate::{CArr, FArr, IntoPy, carray, count, farr};
use comnum::channel::fading::Mimo;
use comnum::channel::spatial::{Kronecker, exponential_correlation as exp_corr};
use comnum::mimo::capacity;
use comnum::mimo::detection::{Linear, MimoDetector};
use comnum::mimo::estimation::{self as est, ChannelEstimator, TracePrior, Training};
use comnum::mimo::message_passing::{self as mp, AmpVariance, Domain, MessagePassing, OampLinear, Prior};
use comnum::mimo::precoding::{Eigenmode, LinearPrecoder};
use comnum::mimo::receiver::DetectorKind;
use comnum::mimo::stbc::Ostbc;
use ndarray::{Array2, Array3, ArrayView2, Axis};
use num_complex::Complex64 as C64;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod amp;
mod estimation;

use estimation::PyMimoTraining;

type Detector = Box<dyn MimoDetector + Send + Sync>;

/// One channel matrix, or one per received vector.
enum Channels {
    One(Array2<C64>),
    Many(Array3<C64>),
}

impl Channels {
    fn get(&self, i: usize) -> ArrayView2<'_, C64> {
        match self {
            Channels::One(h) => h.view(),
            Channels::Many(h) => h.index_axis(Axis(0), i),
        }
    }
}

fn channels(h: &Bound<'_, PyAny>) -> PyResult<Channels> {
    let a = carray(h)?;
    match a.ndim() {
        2 => Ok(Channels::One(a.into_dimensionality().map_err(|e| PyValueError::new_err(e.to_string()))?)),
        3 => Ok(Channels::Many(a.into_dimensionality().map_err(|e| PyValueError::new_err(e.to_string()))?)),
        d => Err(PyValueError::new_err(format!(
            "the channel must be (n_rx, n_tx) or (n, n_rx, n_tx), got {d} dimensions"
        ))),
    }
}

/// The received vectors, one row per channel use, from the stream-major `(n_rx, n)`
/// array (a 1-D array is one vector `(n_rx,)`).
fn received(y: &Bound<'_, PyAny>) -> PyResult<(Array2<C64>, bool)> {
    let a = carray(y)?;
    match a.ndim() {
        1 => {
            let n = a.len();
            Ok((a.into_shape_with_order((1, n)).map_err(|e| PyValueError::new_err(e.to_string()))?, true))
        }
        2 => {
            let a: Array2<C64> = a.into_dimensionality().map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok((a.reversed_axes().as_standard_layout().into_owned(), false))
        }
        d => Err(PyValueError::new_err(format!("the received streams must be (n_rx, n), got {d} dimensions"))),
    }
}

pub(crate) fn detector(
    method: &str,
    c: &comnum::mapping::constellation::Constellation,
    k: usize,
    iterations: usize,
) -> PyResult<Detector> {
    DetectorKind::from_name(method, k).py()?.with_iterations(iterations).py()?.detector(c).py()
}

/// The rows of a per-vector result, stored flat (`width` values per vector).
struct Rows<T> {
    data: Vec<T>,
    width: usize,
}

impl<T> Rows<T> {
    fn map<U>(self, f: impl FnMut(T) -> U) -> Rows<U> {
        Rows { data: self.data.into_iter().map(f).collect(), width: self.width }
    }
}

/// Run `f` on every received vector with its channel, the rows of the result in one
/// buffer (a vector is passed without a copy when its row is contiguous).
fn per_vector<T: Send>(
    h: &Channels,
    y: &Array2<C64>,
    f: impl Fn(&ArrayView2<'_, C64>, &[C64]) -> comnum::Result<Vec<T>> + Sync,
) -> comnum::Result<Rows<T>> {
    if let Channels::Many(hs) = h {
        if hs.shape()[0] != y.nrows() {
            return Err(comnum::Error::Shape(format!(
                "{} channel matrices for {} received vectors",
                hs.shape()[0],
                y.nrows()
            )));
        }
    }
    let mut out = Rows { data: Vec::new(), width: 0 };
    for i in 0..y.nrows() {
        let row = y.row(i);
        let r = match row.as_slice() {
            Some(v) => f(&h.get(i), v)?,
            None => f(&h.get(i), &row.to_vec())?,
        };
        if i == 0 {
            out.width = r.len();
            out.data.reserve(r.len() * y.nrows());
        } else {
            comnum::core::same_len(r.len(), out.width, "values per vector")?;
        }
        out.data.extend(r);
    }
    Ok(out)
}

/// The rows of a per-vector result as a stream-major array: `(width, n)`, one column per
/// vector (`(width,)` for one vector given as a 1-D array).
fn rows_to_array<T: numpy::Element + Clone>(py: Python<'_>, rows: Rows<T>, single: bool) -> PyResult<Py<PyAny>> {
    if single {
        return Ok(rows.data.into_pyarray(py).into_any().unbind());
    }
    let n = rows.data.len().checked_div(rows.width).unwrap_or(0);
    let a = Array2::from_shape_vec((n, rows.width), rows.data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.reversed_axes().as_standard_layout().into_owned().into_pyarray(py).into_any().unbind())
}

/// The bit LLRs of every vector (stream after stream, `bits` per symbol) as the LLR
/// sequence of every stream: `(n_streams, n · bits)` (the `n_streams · bits` LLRs for
/// one vector given as a 1-D array).
fn llr_streams(py: Python<'_>, rows: Rows<f64>, bits: usize, single: bool) -> PyResult<Py<PyAny>> {
    if single || bits == 0 {
        return Ok(rows.data.into_pyarray(py).into_any().unbind());
    }
    let (w, n) = (rows.width, rows.data.len().checked_div(rows.width).unwrap_or(0));
    let streams = w / bits;
    let a =
        ndarray::Array2::from_shape_fn((streams, n * bits), |(s, j)| rows.data[(j / bits) * w + s * bits + j % bits]);
    Ok(a.into_pyarray(py).into_any().unbind())
}

/// Flat MIMO channel matrices: Rayleigh or Rician fading, with exponential antenna
/// correlation (Kronecker model).
///
/// The channel of the Kronecker model [1]_ is
///
/// .. math::
///
///     \mathbf{H} = \sqrt{\tfrac{K}{K+1}}\, \mathbf{H}_{\mathrm{LOS}}
///     + \sqrt{\tfrac{1}{K+1}}\, \mathbf{L}_r \mathbf{G} \mathbf{L}_t^{\mathsf{T}},
///
/// with :math:`\mathbf{G}` of independent :math:`\mathcal{CN}(0, 1)` entries,
/// :math:`\mathbf{L}\mathbf{L}^{\mathsf{H}}` the exponential correlation
/// :math:`[\mathbf{R}]_{ij} = \rho^{j-i}` [2]_, and :math:`\mathbf{H}_{\mathrm{LOS}}` all
/// ones; every entry has unit mean power.
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas.
///     rng (Rng): the generator.
///     k_factor (float): Rician factor :math:`K` (linear, 0 for Rayleigh).
///     rx_correlation (complex): correlation :math:`\rho` of adjacent receive antennas.
///     tx_correlation (complex): correlation :math:`\rho` of adjacent transmit antennas.
///
/// Returns:
///     numpy.ndarray: one matrix `(n_rx, n_tx)` (complex128); :func:`mimo_channels`
///     draws one per channel use.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] J. P. Kermoal et al., "A stochastic MIMO radio channel model with
///        experimental validation," *IEEE J. Sel. Areas Commun.*, 20(6), 2002.
///     .. [2] S. L. Loyka, "Channel capacity of MIMO architecture using the exponential
///        correlation matrix," *IEEE Commun. Lett.*, 5(9), 2001.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.mimo_channel(4, 2, comnum.Rng(1)).shape
///     (4, 2)
#[pyfunction]
#[pyo3(signature = (n_rx, n_tx, rng, *, k_factor=0.0, rx_correlation=C64::new(0.0, 0.0), tx_correlation=C64::new(0.0, 0.0)))]
fn mimo_channel<'py>(
    py: Python<'py>,
    n_rx: usize,
    n_tx: usize,
    mut rng: PyRefMut<'py, PyRng>,
    k_factor: f64,
    rx_correlation: C64,
    tx_correlation: C64,
) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let ch = flat_mimo(n_rx, n_tx, k_factor, rx_correlation, tx_correlation)?;
    Ok(ch.draw(&mut rng.inner).py()?.into_pyarray(py))
}

/// Independent flat MIMO channel matrices, one per channel use, as :func:`mimo_channel`.
///
/// Args:
///     n (int): number of matrices.
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas.
///     rng (Rng): the generator.
///     k_factor (float): Rician factor :math:`K` (linear, 0 for Rayleigh).
///     rx_correlation (complex): correlation :math:`\rho` of adjacent receive antennas.
///     tx_correlation (complex): correlation :math:`\rho` of adjacent transmit antennas.
///
/// Returns:
///     numpy.ndarray: `(n, n_rx, n_tx)` (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> comnum.mimo_channels(10, 4, 2, comnum.Rng(1)).shape
///     (10, 4, 2)
#[pyfunction]
#[pyo3(signature = (n, n_rx, n_tx, rng, *, k_factor=0.0, rx_correlation=C64::new(0.0, 0.0), tx_correlation=C64::new(0.0, 0.0)))]
#[allow(clippy::too_many_arguments, reason = "the antennas and the options of the channel, by keyword")]
fn mimo_channels<'py>(
    py: Python<'py>,
    n: i64,
    n_rx: usize,
    n_tx: usize,
    mut rng: PyRefMut<'py, PyRng>,
    k_factor: f64,
    rx_correlation: C64,
    tx_correlation: C64,
) -> PyResult<Bound<'py, numpy::PyArray3<C64>>> {
    let ch = flat_mimo(n_rx, n_tx, k_factor, rx_correlation, tx_correlation)?;
    let n = count(n, "n")?;
    let mut out = Array3::<C64>::zeros((n, n_rx, n_tx));
    for i in 0..n {
        out.index_axis_mut(Axis(0), i).assign(&ch.draw(&mut rng.inner).py()?);
    }
    Ok(out.into_pyarray(py))
}

/// The flat channel of `mimo_channel`: Rician factor and exponential correlation.
fn flat_mimo(n_rx: usize, n_tx: usize, k_factor: f64, rx_correlation: C64, tx_correlation: C64) -> PyResult<Mimo> {
    let correlation = if rx_correlation.norm() > 0.0 || tx_correlation.norm() > 0.0 {
        Some(Kronecker::new(exp_corr(n_rx, rx_correlation), exp_corr(n_tx, tx_correlation)).py()?)
    } else {
        None
    };
    Ok(Mimo { k_factor, correlation, ..Mimo::rayleigh(n_rx, n_tx) })
}

/// The exponential correlation matrix (Hermitian) of `n` antennas:
///
/// .. math::
///
///     [\mathbf{R}]_{ij} = \rho^{j-i}.
///
/// It is the one-parameter model of a uniform linear array [1]_.
///
/// Args:
///     n (int): number of antennas.
///     rho (complex): correlation :math:`\rho` of adjacent antennas, :math:`|\rho| \le 1`.
///
/// Returns:
///     numpy.ndarray: the `(n, n)` matrix (complex128).
///
/// Raises:
///     ValueError: for a negative count.
///
/// References:
///     .. [1] S. L. Loyka, "Channel capacity of MIMO architecture using the exponential
///        correlation matrix," *IEEE Commun. Lett.*, 5(9), 2001.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.exponential_correlation(3, 0.5).real.tolist()
///     [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
#[pyfunction]
fn exponential_correlation<'py>(py: Python<'py>, n: i64, rho: C64) -> PyResult<Bound<'py, PyArray2<C64>>> {
    Ok(exp_corr(count(n, "n")?, rho).into_pyarray(py))
}

/// Detect the transmitted symbols of a MIMO link.
///
/// The link is
///
/// .. math::
///
///     \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{w},
///
/// with :math:`\mathbf{H}` the :math:`n_{\mathrm{rx}} \times n_{\mathrm{tx}}` channel,
/// :math:`\mathbf{x}` the transmitted symbols and :math:`\mathbf{w}` the noise of
/// variance :math:`\mathbb{E}|w_i|^2` per receive antenna.
///
/// Methods: `"mr"` (matched filter), `"zf"`, `"mmse"` (linear, unbiased), `"zf_sic"`, `"mmse_sic"` (ordered
/// successive interference cancellation, V-BLAST [1]_), `"ml"` (exhaustive search),
/// `"sphere"` (exact ML by the Schnorr–Euchner sphere decoder [2]_), `"kbest"` (K-best
/// tree search keeping `k` paths [3]_), `"amp"` (LAMA, the complex Bayesian AMP of [4]_)
/// and `"oamp"` (OAMP-LMMSE of [5]_), both for large systems.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received streams, `(n_rx, n)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna :math:`\mathbb{E}|w_i|^2`.
///     method (str): the detector.
///     k (int): paths kept by `"kbest"` (8, as the ``mimo_rx`` block).
///     iterations (int): iterations of `"amp"` and `"oamp"`.
///
/// Returns:
///     numpy.ndarray: the labels of the decided points, `(n_tx, n)` (or `(n_tx,)`).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] P. W. Wolniansky, G. J. Foschini, G. D. Golden and R. A. Valenzuela,
///        "V-BLAST: an architecture for realizing very high data rates over the
///        rich-scattering wireless channel," *Proc. ISSSE*, 1998.
///     .. [2] E. Agrell, T. Eriksson, A. Vardy and K. Zeger, "Closest point search in
///        lattices," *IEEE Trans. Inf. Theory*, 48(8), 2002.
///     .. [3] Z. Guo and P. Nilsson, "Algorithm and implementation of the K-best sphere
///        decoding for MIMO detection," *IEEE J. Sel. Areas Commun.*, 24(3), 2006.
///     .. [4] C. Jeon, R. Ghods, A. Maleki and C. Studer, "Optimal data detection in large
///        MIMO," arXiv:1811.01917, 2018.
///     .. [5] J. Ma and L. Ping, "Orthogonal AMP," *IEEE Access*, 5, 2020–2033, 2017
///        (arXiv:1602.06509).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> c = comnum.Constellation("qam", 4).normalized()
///     >>> h = comnum.mimo_channel(4, 2, comnum.Rng(3))
///     >>> x = c.points[[1, 2]]
///     >>> comnum.mimo_detect(h, h @ x, c, 1e-3, "sphere").tolist()
///     [1, 2]
#[pyfunction]
#[pyo3(signature = (
    h, y, constellation, noise_var, method="mmse", *, k=DetectorKind::DEFAULT_PATHS,
    iterations=DetectorKind::DEFAULT_ITERATIONS
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn mimo_detect(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    k: usize,
    iterations: usize,
) -> PyResult<Py<PyAny>> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let d = detector(method, &constellation.inner, k, iterations)?;
    let rows = py.detach(|| per_vector(&hs, &ys, |h, y| d.detect(h, y, noise_var))).py()?;
    rows_to_array(py, rows.map(|v| v as i64), single)
}

/// Bit log-likelihood ratios of a MIMO link (positive for a 0, most significant bit
/// first), the LLR sequence of every transmitted stream.
///
/// The link is :math:`\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{w}`, as for
/// :func:`mimo_detect`. The LLRs are exact per stream after the unbiased linear filter
/// for `"mr"`, `"zf"` and `"mmse"`, max-log over all hypotheses for `"ml"`, over the final
/// list for `"kbest"`, exact LLRs of the decoupled channel for `"amp"` and `"oamp"`.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received streams, `(n_rx, n)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna :math:`\mathbb{E}|w_i|^2`.
///     method (str): `"mr"`, `"zf"`, `"mmse"`, `"ml"`, `"kbest"`, `"amp"` or `"oamp"` (the SIC
///         and sphere detectors have no soft output).
///     k (int): paths kept by `"kbest"`.
///     iterations (int): iterations of `"amp"` and `"oamp"`.
///
/// Returns:
///     numpy.ndarray: :math:`(n_{\mathrm{tx}}, n \log_2 M)`, the LLRs of the symbols of every
///     stream in order (for one vector `(n_rx,)`, the :math:`n_{\mathrm{tx}} \log_2 M` LLRs
///     stream after stream), float64.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (
    h, y, constellation, noise_var, method="mmse", *, k=DetectorKind::DEFAULT_PATHS,
    iterations=DetectorKind::DEFAULT_ITERATIONS
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn mimo_llr(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    k: usize,
    iterations: usize,
) -> PyResult<Py<PyAny>> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let d = detector(method, &constellation.inner, k, iterations)?;
    let rows = py.detach(|| per_vector(&hs, &ys, |h, y| d.llr(h, y, noise_var))).py()?;
    llr_streams(py, rows, constellation.inner.bits_per_symbol(), single)
}

/// An orthogonal space-time block code (Alamouti, or the rate-3/4 codes for 3 and 4
/// antennas), with linear maximum-likelihood decoding.
///
/// A block sends :math:`K` = `n_symbols` symbols over :math:`T` = `n_slots` channel uses
/// from `n_tx` antennas, with a unit average transmitted energy per channel use: the
/// :math:`T \times n_{\mathrm{tx}}` codeword is
///
/// .. math::
///
///     X(s) = c \sum_{k=1}^{K} \left(\Re(s_{k})\, A_{k} + \Im(s_{k})\, B_{k}\right),
///
/// with the dispersion matrices :math:`A_{k} = X(e_{k})` and :math:`B_{k} = X(j e_{k})`
/// and the scale :math:`c` of the unit energy. Decoding turns the block
/// :math:`Y = H X^{T} + W` into one scalar channel per symbol [1]_,
///
/// .. math::
///
///     \hat s_{k} = s_{k} + n_{k}, \qquad
///     \mathbb{E}|n_k|^2 = \sigma^2 / (c^2 \lVert \mathbf{H} \rVert_F^2),
///
/// a diversity of :math:`n_{\mathrm{tx}} \cdot n_{\mathrm{rx}}`. Build it with
/// `Ostbc.alamouti()` [2]_ or `Ostbc.rate_3_4(n_tx)`.
///
/// References:
///     .. [1] V. Tarokh, H. Jafarkhani and A. R. Calderbank, "Space-time block codes from
///        orthogonal designs," *IEEE Trans. Inf. Theory*, 45(5), 1999.
///     .. [2] S. M. Alamouti, "A simple transmit diversity technique for wireless
///        communications," *IEEE J. Sel. Areas Commun.*, 16(8), 1998.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> code = comnum.Ostbc.alamouti()
///     >>> s = np.array([1 + 1j, -1 + 1j]) / np.sqrt(2)
///     >>> h = comnum.mimo_channel(1, 2, comnum.Rng(2))
///     >>> est, var = code.decode(h, h @ code.encode(s), 0.01)
///     >>> np.allclose(est, s)
///     True
#[pyclass(name = "Ostbc", module = "comnum")]
pub struct PyOstbc {
    inner: Ostbc,
}

#[pymethods]
impl PyOstbc {
    /// Alamouti's code: two antennas, rate one.
    ///
    /// Returns:
    ///     Ostbc: the code.
    #[staticmethod]
    fn alamouti() -> Self {
        PyOstbc { inner: Ostbc::alamouti() }
    }

    /// The rate-3/4 code for 3 or 4 antennas.
    ///
    /// Args:
    ///     n_tx (int): 3 or 4.
    ///
    /// Returns:
    ///     Ostbc: the code.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    #[staticmethod]
    fn rate_3_4(n_tx: usize) -> PyResult<Self> {
        Ok(PyOstbc { inner: Ostbc::rate_3_4(n_tx).py()? })
    }

    /// str: Name of the code.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// int: Transmit antennas.
    #[getter]
    fn n_tx(&self) -> usize {
        self.inner.n_tx()
    }

    /// int: Channel uses per block.
    #[getter]
    fn n_slots(&self) -> usize {
        self.inner.n_slots()
    }

    /// int: Symbols per block.
    #[getter]
    fn n_symbols(&self) -> usize {
        self.inner.n_symbols()
    }

    /// float: Symbols per channel use.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    /// The signal of the antennas.
    ///
    /// Args:
    ///     symbols (array_like): a multiple of `n_symbols` symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n_tx, n_slots · blocks)` (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn encode<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(self.inner.encode(&crate::cvec(symbols)?).py()?.into_pyarray(py))
    }

    /// Decode the received blocks (linear maximum likelihood).
    ///
    /// Args:
    ///     h (array_like): the channel, `(n_rx, n_tx)`, or `(blocks, n_rx, n_tx)`.
    ///     y (array_like): the received blocks, `(n_rx, n_slots · blocks)`.
    ///     noise_var (float): noise variance per receive antenna.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: the symbol estimates and their noise
    ///     variances.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn decode<'py>(
        &self,
        py: Python<'py>,
        h: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        noise_var: f64,
    ) -> PyResult<(CArr<'py>, FArr<'py>)> {
        let ya: Array2<C64> = carray(y)?
            .into_dimensionality()
            .map_err(|_| PyValueError::new_err("y must be (n_rx, n_slots · blocks)"))?;
        let e = match channels(h)? {
            Channels::One(h) => py.detach(|| self.inner.decode(&h.view(), &ya.view(), noise_var)),
            Channels::Many(hs) => {
                let list: Vec<Array2<C64>> = hs.outer_iter().map(|m| m.to_owned()).collect();
                py.detach(|| self.inner.decode_blocks(&list, &ya.view(), noise_var))
            }
        }
        .py()?;
        Ok((e.symbols.into_pyarray(py), farr(py, e.noise_var)))
    }

    fn __repr__(&self) -> String {
        format!("Ostbc({:?}, n_tx={}, rate={})", self.inner.name(), self.inner.n_tx(), self.inner.rate())
    }
}

/// Water-filling powers of parallel channels.
///
/// The powers that maximise :math:`\sum_i \log_2(1 + p_i g_i)` are [1]_
///
/// .. math::
///
///     p_i = (\mu - 1/g_i)^+,
///
/// with the level :math:`\mu` such that :math:`\sum_i p_i` is the total power.
///
/// Args:
///     gains (array_like): the gains :math:`g_i` (SNR per unit power).
///     total_power (float): the total power.
///
/// Returns:
///     numpy.ndarray: the power of every channel (float64).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] E. Telatar, "Capacity of multi-antenna Gaussian channels,"
///        *Eur. Trans. Telecommun.*, 10(6), 1999.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.water_filling_powers([1.0, 1.0], 2.0).tolist()
///     [1.0, 1.0]
#[pyfunction]
fn water_filling_powers<'py>(
    py: Python<'py>,
    gains: &Bound<'py, PyAny>,
    total_power: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    Ok(farr(py, comnum::mimo::precoding::water_filling_powers(&crate::fvec(gains)?, total_power).py()?))
}

/// Capacity of MIMO channel realisations (bits per channel use) at the SNR
/// :math:`\rho = P/\sigma^2`:
///
/// .. math::
///
///     C = \log_2 \det\left(\mathbf{I} + \frac{\rho}{n_{\mathrm{tx}}}
///     \mathbf{H}\mathbf{H}^{\mathsf{H}}\right)
///
/// with equal power, or :math:`C = \sum_{i} \log_{2}(1 + p_{i} \sigma_{i}^{2})` with
/// water-filling on the eigenmodes [1]_.
///
/// Args:
///     h (array_like): `(n, n_rx, n_tx)`, one matrix per realisation (one matrix
///         `(n_rx, n_tx)` is one realisation).
///     snr_db (float): total transmit power over the noise variance per receive antenna
///         (dB).
///     water_filling (bool): channel knowledge at the transmitter.
///
/// Returns:
///     numpy.ndarray: the capacity of every realisation, `(n,)`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] E. Telatar, "Capacity of multi-antenna Gaussian channels,"
///        *Eur. Trans. Telecommun.*, 10(6), 1999.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> comnum.mimo_capacity(np.eye(2), 10 * np.log10(10.0)).round(6).tolist()
///     [5.169925]
#[pyfunction]
#[pyo3(signature = (h, snr_db, *, water_filling=false))]
fn mimo_capacity<'py>(py: Python<'py>, h: &Bound<'py, PyAny>, snr_db: f64, water_filling: bool) -> PyResult<FArr<'py>> {
    let snr = comnum::math::db2lin(snr_db);
    let f = |m: &ArrayView2<'_, C64>| {
        if water_filling { capacity::water_filling(m, snr) } else { capacity::equal_power(m, snr) }
    };
    let c = match channels(h)? {
        Channels::One(m) => vec![f(&m.view()).py()?],
        Channels::Many(hs) => {
            py.detach(|| hs.outer_iter().map(|m| f(&m)).collect::<comnum::Result<Vec<f64>>>()).py()?
        }
    };
    Ok(farr(py, c))
}

/// The ergodic capacity (bits per channel use) of the i.i.d. Rayleigh MIMO channel with
/// equal power on the transmit antennas, in closed form: the mean of `mimo_capacity`
/// over the realisations.
///
/// With :math:`m = \min(n_{r}, n_{t})`, :math:`n = \max(n_{r}, n_{t})` and the
/// associated Laguerre polynomials :math:`L_{k}^{n-m}` ([1]_, Theorem 2, eq. (8)),
///
/// .. math::
///
///     C = \int_{0}^{\infty} \log_{2}\left(1 + \frac{\rho}{n_{t}} \lambda\right)
///     \sum_{k=0}^{m-1} \frac{k!}{(k+n-m)!} \left[L_{k}^{n-m}(\lambda)\right]^{2}
///     \lambda^{n-m} e^{-\lambda} \, d\lambda.
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas.
///     snr_db (float | array_like): total transmit power over the noise variance per
///         receive antenna (dB).
///
/// Returns:
///     float | numpy.ndarray: the capacity at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] E. Telatar, "Capacity of multi-antenna Gaussian channels,"
///        *Eur. Trans. Telecommun.*, 10(6), 1999.
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.theory_ergodic_capacity(1, 1, 10.0), 4)   # log2(e) e^0.1 E1(0.1)
///     2.9065
#[pyfunction]
fn theory_ergodic_capacity(n_rx: usize, n_tx: usize, snr_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    crate::map_real(snr_db, |s| capacity::ergodic_rayleigh(n_rx, n_tx, comnum::math::db2lin(s)).py())
}

/// The probability that a Rayleigh channel drawn once cannot carry `rate` bits per
/// channel use with equal power, one antenna on one side.
///
/// :math:`\lVert h \rVert^{2}` is a gamma variable of shape
/// :math:`k = \max(n_{r}, n_{t})`, so that
///
/// .. math::
///
///     P_{\mathrm{out}} = P\left(\log_{2} \det\left(I + \frac{\rho}{n_{t}} H H^{H}\right) < R\right)
///     = P\left(k, \frac{n_{t} (2^{R} - 1)}{\rho}\right),
///
/// with the regularized incomplete gamma function :math:`P(k, x)` ([1]_, eq. (16) and
/// Example 7).
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas (`n_rx` or `n_tx` is 1).
///     snr_db (float | array_like): the SNR (dB).
///     rate (float): the rate (bits per channel use).
///
/// Returns:
///     float | numpy.ndarray: the outage probability at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] E. Telatar, "Capacity of multi-antenna Gaussian channels,"
///        *Eur. Trans. Telecommun.*, 10(6), 1999.
#[pyfunction]
fn theory_outage_probability(n_rx: usize, n_tx: usize, snr_db: &Bound<'_, PyAny>, rate: f64) -> PyResult<Py<PyAny>> {
    crate::map_real(snr_db, |s| capacity::outage_rayleigh(n_rx, n_tx, comnum::math::db2lin(s), rate).py())
}

/// The :math:`\epsilon`-capacity (bits per channel use): the rate of outage probability `eps`
/// over a Rayleigh channel drawn once, one antenna on one side.
///
/// The rate is ([1]_, eqs. (17), (18))
///
/// .. math::
///
///     R = \log_{2}(1 + \rho x / n_{t}), \qquad P(k, x) = \epsilon,
///
/// with :math:`k = \max(n_{r}, n_{t})` and the regularized incomplete gamma function
/// :math:`P(k, x)` of :func:`theory_outage_probability`.
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas (`n_rx` or `n_tx` is 1).
///     snr_db (float | array_like): the SNR (dB).
///     eps (float): the outage probability, in (0, 1).
///
/// Returns:
///     float | numpy.ndarray: the rate at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] E. Telatar, "Capacity of multi-antenna Gaussian channels,"
///        *Eur. Trans. Telecommun.*, 10(6), 1999.
#[pyfunction]
fn theory_outage_capacity(n_rx: usize, n_tx: usize, snr_db: &Bound<'_, PyAny>, eps: f64) -> PyResult<Py<PyAny>> {
    crate::map_real(snr_db, |s| capacity::outage_capacity_rayleigh(n_rx, n_tx, comnum::math::db2lin(s), eps).py())
}

/// Eigenmode (SVD) transmission over `h`: precoder, receive filter, gains and powers of
/// the active modes (water-filling of `total_power` against `noise_var`, or equal power).
///
/// With the SVD :math:`\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{\mathsf{H}}`,
/// sending :math:`\mathbf{x} = \mathbf{V}\mathbf{P}^{1/2} \mathbf{s}` and filtering with
/// :math:`\mathbf{U}^{\mathsf{H}}` gives parallel scalar channels [1]_:
///
/// .. math::
///
///     \tilde y_{i} = \sigma_{i} \sqrt{p_{i}}\, s_{i} + \tilde w_{i}.
///
/// Args:
///     h (array_like): one channel `(n_rx, n_tx)`.
///     total_power (float): the total transmit power.
///     noise_var (float): noise variance per receive antenna.
///     water_filling (bool): water-filling (else equal power on every mode).
///
/// Returns:
///     dict: ``precoder`` `(n_tx, streams)`, ``combiner`` `(streams, n_rx)`, ``gains``,
///     ``powers``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] E. Telatar, "Capacity of multi-antenna Gaussian channels,"
///        *Eur. Trans. Telecommun.*, 10(6), 1999.
#[pyfunction]
#[pyo3(signature = (h, total_power, noise_var, *, water_filling=true))]
fn eigenmode<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    total_power: f64,
    noise_var: f64,
    water_filling: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let Channels::One(m) = channels(h)? else {
        return Err(PyValueError::new_err("eigenmode takes one (n_rx, n_tx) channel"));
    };
    let e = Eigenmode::new(&m.view(), total_power, noise_var, water_filling).py()?;
    let d = PyDict::new(py);
    d.set_item("precoder", e.precoder.into_pyarray(py))?;
    d.set_item("combiner", e.combiner.into_pyarray(py))?;
    d.set_item("gains", farr(py, e.gains))?;
    d.set_item("powers", farr(py, e.powers))?;
    Ok(d)
}

/// The (regularised) zero-forcing precoder of the users of the rows of `h`, scaled to
/// `total_power`:
///
/// .. math::
///
///     \mathbf{W} \propto \mathbf{H}^{\mathsf{H}} (\mathbf{H}\mathbf{H}^{\mathsf{H}} + \alpha \mathbf{I})^{-1},
///     \qquad \lVert \mathbf{W} \rVert_{F}^{2} = P,
///
/// with :math:`\alpha = 0` for zero forcing, :math:`K \sigma^2 / P` for the regularised
/// version [1]_.
///
/// Args:
///     h (array_like): the channel `(n_users, n_tx)`.
///     total_power (float): the total transmit power.
///     alpha (float): the regularisation :math:`\alpha`.
///
/// Returns:
///     numpy.ndarray: the precoder `(n_tx, n_users)` (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// References:
///     .. [1] C. B. Peel, B. M. Hochwald and A. L. Swindlehurst, "A vector-perturbation
///        technique for near-capacity multiantenna multiuser communication — Part I:
///        channel inversion and regularization," *IEEE Trans. Commun.*, 53(1), 2005.
#[pyfunction]
#[pyo3(signature = (h, *, total_power=1.0, alpha=0.0))]
fn zf_precoder<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    total_power: f64,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let Channels::One(m) = channels(h)? else {
        return Err(PyValueError::new_err("zf_precoder takes one (n_users, n_tx) channel"));
    };
    let p = if alpha == 0.0 {
        LinearPrecoder::zero_forcing(&m.view(), total_power)
    } else {
        LinearPrecoder::regularized(&m.view(), alpha, total_power)
    }
    .py()?;
    Ok(p.w.into_pyarray(py))
}

/// Register the MIMO bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOstbc>()?;
    m.add_class::<PyMimoTraining>()?;
    m.add_function(wrap_pyfunction!(estimation::blue_weights, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::blue_combine, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::blue_mse, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_channel, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_channels, m)?)?;
    m.add_function(wrap_pyfunction!(exponential_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_detect, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_llr, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::mimo_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(amp::message_passing, m)?)?;
    m.add_function(wrap_pyfunction!(amp::amp_state_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(amp::amp_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(amp::oamp_state_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(water_filling_powers, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ergodic_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(theory_outage_probability, m)?)?;
    m.add_function(wrap_pyfunction!(theory_outage_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(eigenmode, m)?)?;
    m.add_function(wrap_pyfunction!(zf_precoder, m)?)?;
    Ok(())
}
