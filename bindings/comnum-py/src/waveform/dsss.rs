//! Direct-sequence spread spectrum: the spreading codes, the DS-CDMA link with its
//! multiuser receivers and rake, and the closed forms.

use super::*;
use comnum::math::sequences;
use comnum::mimo::detection::{Criterion, Linear};
use comnum::mimo::receiver::DetectorKind;
use comnum::waveform::dsss::{Dsss, theory as dsss_theory};
use ndarray::{Array2, Ix1, Ix2};

/// Codes of one length as a `(n_codes, N)` array of bits.
fn codes_array(py: Python<'_>, codes: Vec<Vec<u8>>) -> PyResult<Bound<'_, numpy::PyArray2<u8>>> {
    let n = codes.first().map_or(0, Vec::len);
    let a =
        Array2::from_shape_vec((codes.len(), n), codes.concat()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py))
}

/// One period of the m-sequence of order `order`: `N = 2^order - 1` bits of the
/// register of a primitive polynomial, balanced, with the periodic autocorrelation
/// `-1` at every nonzero lag.
///
/// Args:
///     order (int): the order of the register, 2 to 16.
///
/// Returns:
///     numpy.ndarray: the `N` bits (uint8).
///
/// Raises:
///     ValueError: for an order outside 2 to 16.
///
/// Examples:
///     >>> import comnum
///     >>> u = comnum.m_sequence(5)
///     >>> u.size, int(u.sum()), comnum.periodic_correlation(u, u)[:3].tolist()
///     (31, 16, [31, -1, -1])
#[pyfunction]
fn m_sequence(py: Python<'_>, order: usize) -> PyResult<BArr<'_>> {
    Ok(sequences::m_sequence(order).py()?.into_pyarray(py))
}

/// The `2^order + 1` Gold codes of length `N = 2^order - 1` (Gold 1967): the
/// m-sequence `u`, its decimation `v` and the sums `u ⊕ T^k v`. The periodic
/// correlations between two codes, and of each off its peak, take the three values
/// `-1`, `-t` and `t - 2`, `t = 1 + 2^⌊(order + 2)/2⌋`.
///
/// Args:
///     order (int): 3 to 16, not a multiple of 4.
///
/// Returns:
///     numpy.ndarray: the codes, `(2^order + 1, N)` (uint8).
///
/// Raises:
///     ValueError: for an order without preferred pair or outside the range.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> g = comnum.gold_codes(5)
///     >>> g.shape, sorted(set(comnum.periodic_correlation(g[2], g[7]).tolist()))
///     ((33, 31), [-9, -1, 7])
#[pyfunction]
fn gold_codes(py: Python<'_>, order: usize) -> PyResult<Bound<'_, numpy::PyArray2<u8>>> {
    codes_array(py, sequences::gold_codes(order).py()?)
}

/// The small set of `2^(order/2)` Kasami codes of length `N = 2^order - 1`:
/// correlations in `{-1, -s, s - 2}`, `s = 1 + 2^(order/2)`.
///
/// Args:
///     order (int): even, 4 to 16.
///
/// Returns:
///     numpy.ndarray: the codes, `(2^(order/2), N)` (uint8).
///
/// Raises:
///     ValueError: for an odd order or one outside the range.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.kasami_codes(6).shape
///     (8, 63)
#[pyfunction]
fn kasami_codes(py: Python<'_>, order: usize) -> PyResult<Bound<'_, numpy::PyArray2<u8>>> {
    codes_array(py, sequences::kasami_codes(order).py()?)
}

/// The `n` Walsh–Hadamard codes of length `n` (the rows of the Sylvester–Hadamard
/// matrix): orthogonal when aligned.
///
/// Args:
///     n (int): the length, a power of two.
///
/// Returns:
///     numpy.ndarray: the codes, `(n, n)` (uint8).
///
/// Raises:
///     ValueError: for a length that is not a power of two.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.walsh_codes(4).tolist()
///     [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0]]
#[pyfunction]
fn walsh_codes(py: Python<'_>, n: usize) -> PyResult<Bound<'_, numpy::PyArray2<u8>>> {
    codes_array(py, sequences::walsh_codes(n).py()?)
}

/// The Barker code of length `n`: aperiodic autocorrelation sidelobes of magnitude at
/// most 1.
///
/// Args:
///     n (int): 2, 3, 4, 5, 7, 11 or 13.
///
/// Returns:
///     numpy.ndarray: the `n` bits (uint8).
///
/// Raises:
///     ValueError: for another length.
///
/// Examples:
///     >>> import comnum
///     >>> c = comnum.barker_code(11)
///     >>> comnum.aperiodic_correlation(c, c)[10:14].tolist()
///     [11, 0, -1, 0]
#[pyfunction]
fn barker_code(py: Python<'_>, n: usize) -> PyResult<BArr<'_>> {
    Ok(sequences::barker_code(n).py()?.into_pyarray(py))
}

/// A vector of correlations as an array of `int64`.
fn correlations(py: Python<'_>, v: Vec<i32>) -> IArr<'_> {
    v.into_iter().map(i64::from).collect::<Vec<_>>().into_pyarray(py)
}

/// The periodic correlation `θ(l) = Σ_i (-1)^(a_i + b_((i + l) mod N))` of two binary
/// codes at the lags `0 … N - 1`.
///
/// Args:
///     a (array_like): `N` bits.
///     b (array_like): `N` bits.
///
/// Returns:
///     numpy.ndarray: the `N` values (int64).
///
/// Raises:
///     ValueError: for empty codes, codes of different lengths or values other than 0 and 1.
#[pyfunction]
fn periodic_correlation<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
    Ok(correlations(py, sequences::periodic_correlation(&bvec(a)?, &bvec(b)?).py()?))
}

/// The aperiodic correlation of two binary codes at the lags `-(N - 1) … N - 1` (the
/// lag 0 at the index `N - 1`): the output of a matched filter on one isolated period.
///
/// Args:
///     a (array_like): `N` bits.
///     b (array_like): `N` bits.
///
/// Returns:
///     numpy.ndarray: the `2N - 1` values (int64).
///
/// Raises:
///     ValueError: for empty codes, codes of different lengths or values other than 0 and 1.
#[pyfunction]
fn aperiodic_correlation<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
    Ok(correlations(py, sequences::aperiodic_correlation(&bvec(a)?, &bvec(b)?).py()?))
}

/// The linear receiver of a name: `"mr"` (conventional), `"zf"` (decorrelator) or `"mmse"`.
fn criterion(receiver: &str) -> PyResult<Criterion> {
    match receiver.parse::<DetectorKind>().py()? {
        DetectorKind::Mr => Ok(Criterion::Mr),
        DetectorKind::Zf => Ok(Criterion::Zf),
        DetectorKind::Mmse => Ok(Criterion::Mmse),
        _ => Err(PyValueError::new_err(format!("a linear receiver is \"mr\", \"zf\" or \"mmse\", got {receiver:?}"))),
    }
}

/// A `(n_streams, n)` complex array-like as its rows (a 1-D array is one row).
fn rows(x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<C64>>> {
    let a = crate::carray(x)?;
    let bad = |e: ndarray::ShapeError| PyValueError::new_err(e.to_string());
    match a.ndim() {
        1 => Ok(vec![a.into_dimensionality::<Ix1>().map_err(bad)?.to_vec()]),
        2 => Ok(a.into_dimensionality::<Ix2>().map_err(bad)?.rows().into_iter().map(|r| r.to_vec()).collect()),
        d => Err(PyValueError::new_err(format!("expected (n_users, n) symbols, got {d} dimensions"))),
    }
}

/// Rows of one length as a `(n_rows, n)` array.
fn matrix<T: numpy::Element + Clone>(py: Python<'_>, rows: Vec<Vec<T>>) -> PyResult<Bound<'_, numpy::PyArray2<T>>> {
    let n = rows.first().map_or(0, Vec::len);
    let a = Array2::from_shape_vec((rows.len(), n), rows.concat()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py))
}

/// A direct-sequence spread-spectrum link of `K` synchronous users (DS-CDMA): every
/// user sends its symbols on a signature of `N` chips, `y = S A b + w` per symbol
/// period. The receivers are the bank of matched filters (`demodulate`), the multiuser
/// detectors of `mimo_detect` on the signature matrix (`detect`) and the rake
/// receiver over a multipath channel (`rake`).
///
/// Args:
///     codes (array_like): the binary codes, `(K, N)` (or `(N,)` for one user), sent as
///         the chips `(-1)^c / √N`.
///     amplitudes (array_like | None): the received amplitude of every user (1 by default).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> cdma = comnum.Dsss(comnum.gold_codes(5)[2:5], amplitudes=[1.0, 10.0, 1.0])
///     >>> bpsk = comnum.Constellation("psk", 2)
///     >>> labels = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]])
///     >>> chips = cdma.modulate(bpsk.points[labels])
///     >>> chips.size, bool((cdma.detect(chips, bpsk, 1e-2, "zf") == labels).all())
///     (93, True)
#[pyclass(name = "Dsss", module = "comnum")]
pub struct PyDsss {
    inner: Dsss,
}

#[pymethods]
impl PyDsss {
    #[new]
    #[pyo3(signature = (codes, *, amplitudes=None))]
    fn new(codes: &Bound<'_, PyAny>, amplitudes: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let np = codes.py().import("numpy")?;
        let codes: Vec<Vec<u8>> =
            np.call_method1("atleast_2d", (codes,))?.try_iter()?.map(|row| bvec(&row?)).collect::<PyResult<_>>()?;
        let inner = Dsss::new(&codes).py()?;
        let inner = match amplitudes {
            Some(a) => inner.with_amplitudes(&fvec(a)?).py()?,
            None => inner,
        };
        Ok(PyDsss { inner })
    }

    /// int: The spreading factor `N`, chips per symbol.
    #[getter]
    fn spreading_factor(&self) -> usize {
        self.inner.spreading_factor()
    }

    /// int: The number of users `K`.
    #[getter]
    fn n_users(&self) -> usize {
        self.inner.n_users()
    }

    /// numpy.ndarray: The received amplitude of every user.
    #[getter]
    fn amplitudes<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.amplitudes())
    }

    /// numpy.ndarray: The matrix `H = S A`, `(N, K)`: the unit-norm signatures in
    /// columns, scaled by the amplitudes (the channel matrix of `mimo_detect`).
    #[getter]
    fn signatures<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<C64>> {
        self.inner.signatures().to_owned().into_pyarray(py)
    }

    /// The chips of the symbols of every user.
    ///
    /// Args:
    ///     symbols (array_like): `(K, n)` complex symbols (`(n,)` for one user).
    ///
    /// Returns:
    ///     numpy.ndarray: the `n N` chips of the sum of the users.
    ///
    /// Raises:
    ///     ValueError: unless there is one row of symbols per user.
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&rows(symbols)?).py()?))
    }

    /// The bank of matched filters: the estimates `s_k^H y / A_k` of the symbols of
    /// every user (despreading; with several users, the interference of the others
    /// remains).
    ///
    /// Args:
    ///     chips (array_like): a multiple of `N` chips.
    ///
    /// Returns:
    ///     numpy.ndarray: the estimates, `(K, n)`.
    ///
    /// Raises:
    ///     ValueError: unless `chips` holds whole symbols.
    fn demodulate<'py>(
        &self,
        py: Python<'py>,
        chips: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<C64>>> {
        matrix(py, self.inner.demodulate(&cvec(chips)?).py()?)
    }

    /// The decisions of a multiuser detector, one of `mimo_detect` on the signature
    /// matrix: `"mr"` (conventional receiver), `"zf"` (decorrelator), `"mmse"`,
    /// `"zf_sic"`, `"mmse_sic"` (successive cancellation), `"ml"`, `"sphere"` (optimum),
    /// `"kbest"`, `"amp"`, `"oamp"`.
    ///
    /// Args:
    ///     chips (array_like): a multiple of `N` chips.
    ///     constellation (Constellation): the alphabet of every user.
    ///     noise_var (float): noise variance per chip `E|w_i|²`.
    ///     method (str): the detector.
    ///     k (int): paths kept by `"kbest"`.
    ///     iterations (int): iterations of `"amp"` and `"oamp"`.
    ///
    /// Returns:
    ///     numpy.ndarray: the labels of the decided points, `(K, n)`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (
        chips, constellation, noise_var, method="mmse", *, k=DetectorKind::DEFAULT_PATHS,
        iterations=DetectorKind::DEFAULT_ITERATIONS
    ))]
    #[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python method")]
    fn detect<'py>(
        &self,
        py: Python<'py>,
        chips: &Bound<'py, PyAny>,
        constellation: &PyConstellation,
        noise_var: f64,
        method: &str,
        k: usize,
        iterations: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
        let d = crate::mimo::detector(method, &constellation.inner, k, iterations)?;
        let chips = cvec(chips)?;
        let labels = py.detach(|| self.inner.detect(d.as_ref(), &chips, noise_var)).py()?;
        matrix(py, labels.into_iter().map(|row| row.into_iter().map(|l| l as i64).collect()).collect())
    }

    /// The signal-to-interference-plus-noise ratio of every user at the output of a
    /// linear receiver, for these signatures and unit-energy symbols.
    ///
    /// Args:
    ///     noise_var (float): noise variance per chip `E|w_i|²`.
    ///     receiver (str): `"mr"` (conventional), `"zf"` (decorrelator) or `"mmse"`.
    ///
    /// Returns:
    ///     numpy.ndarray: the `K` ratios (linear).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    #[pyo3(signature = (noise_var, receiver="mmse"))]
    fn sinr<'py>(&self, py: Python<'py>, noise_var: f64, receiver: &str) -> PyResult<FArr<'py>> {
        let constellation = comnum::mapping::constellation::Constellation::psk(2).py()?;
        let linear = Linear { constellation, criterion: criterion(receiver)? };
        Ok(farr(py, self.inner.sinr(&linear, noise_var).py()?))
    }

    /// The rake receiver of one user over a multipath channel `r[n] = Σ_l g_l x[n -
    /// d_l] + w[n]`: one finger per path, combined with the conjugate gains
    /// (maximal-ratio combining of the paths).
    ///
    /// Args:
    ///     chips (array_like): the received chips.
    ///     delays (array_like): the delay of every finger (chips).
    ///     gains (array_like): the complex gain of every finger.
    ///     user (int): the user to receive.
    ///
    /// Returns:
    ///     numpy.ndarray: one estimate per whole symbol of `chips`.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (chips, delays, gains, *, user=0))]
    fn rake<'py>(
        &self,
        py: Python<'py>,
        chips: &Bound<'py, PyAny>,
        delays: &Bound<'py, PyAny>,
        gains: &Bound<'py, PyAny>,
        user: usize,
    ) -> PyResult<CArr<'py>> {
        let (delays, gains) = (uvec(delays)?, cvec(gains)?);
        if delays.len() != gains.len() {
            return Err(PyValueError::new_err("one gain per delay"));
        }
        let fingers: Vec<(usize, C64)> = delays.into_iter().zip(gains).collect();
        Ok(carr(py, self.inner.rake(&cvec(chips)?, user, &fingers).py()?))
    }

    fn __repr__(&self) -> String {
        format!("Dsss(n_users={}, spreading_factor={})", self.inner.n_users(), self.inner.spreading_factor())
    }
}

/// A `(N, K)` complex array-like as a matrix.
fn signature_matrix(h: &Bound<'_, PyAny>) -> PyResult<Array2<C64>> {
    crate::carray(h)?
        .into_dimensionality::<Ix2>()
        .map_err(|_| PyValueError::new_err("the signatures must be a (N, K) matrix"))
}

/// The exact bit error rate of one user of a synchronous DS-CDMA link with BPSK, for
/// given signatures: the conventional receiver (the mean over the patterns of the
/// interfering bits, 21 users at most) or the decorrelator, `Q(√(2/(σ² (G⁻¹)_kk)))`
/// (Lupas and Verdú 1989).
///
/// Args:
///     signatures (array_like): `H = S A`, `(N, K)` (`Dsss.signatures`).
///     noise_var (float | array_like): noise variance per chip `E|w_i|²`.
///     receiver (str): `"mr"` (conventional) or `"zf"` (decorrelator).
///     user (int): the user.
///
/// Returns:
///     float | numpy.ndarray: the bit error rate.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> h = comnum.Dsss(comnum.walsh_codes(8)).signatures
///     >>> round(comnum.theory_ber_cdma(h, 0.25), 6) == round(comnum.qfunc(8 ** 0.5), 6)
///     True
#[pyfunction]
#[pyo3(signature = (signatures, noise_var, receiver="mr", *, user=0))]
fn theory_ber_cdma(
    signatures: &Bound<'_, PyAny>,
    noise_var: &Bound<'_, PyAny>,
    receiver: &str,
    user: usize,
) -> PyResult<Py<PyAny>> {
    let h = signature_matrix(signatures)?;
    match criterion(receiver)? {
        Criterion::Mr => map_real(noise_var, |v| dsss_theory::ber_matched_filter(&h.view(), user, v).py()),
        Criterion::Zf => map_real(noise_var, |v| dsss_theory::ber_decorrelator(&h.view(), user, v).py()),
        Criterion::Mmse => Err(PyValueError::new_err("no exact error rate for the MMSE receiver: use Dsss.sinr")),
    }
}

/// The near–far resistance `1 / (G_kk (G⁻¹)_kk)` of one user: the asymptotic
/// efficiency of the decorrelator, and of the optimum detector against interferers of
/// the worst amplitudes (Lupas and Verdú 1989).
///
/// Args:
///     signatures (array_like): `H = S A`, `(N, K)`.
///     user (int): the user.
///
/// Returns:
///     float: in `(0, 1]`, 1 for a signature orthogonal to the others.
///
/// Raises:
///     ValueError: for an invalid parameter or linearly dependent signatures.
#[pyfunction]
#[pyo3(signature = (signatures, *, user=0))]
fn cdma_near_far_resistance(signatures: &Bound<'_, PyAny>, user: usize) -> PyResult<f64> {
    dsss_theory::near_far_resistance(&signature_matrix(signatures)?.view(), user).py()
}

/// The large-system SINR (dB) of a linear receiver for random signatures and users of
/// equal power, at the load `K/N` (Verdú and Shamai 1999, eqs. (10)–(12); Tse and Hanly
/// 1999, eq. (9)).
///
/// Args:
///     load (float): users per chip, `K/N`.
///     snr_db (float | array_like): `A²/σ²` (dB), the SNR after despreading without interference.
///     receiver (str): `"mr"` (conventional), `"zf"` (decorrelator) or `"mmse"`.
///
/// Returns:
///     float | numpy.ndarray: the SINR (dB; `-inf` for the decorrelator at a load of 1 or more).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.cdma_sinr_large_system(80 / 128, 20.0), 2)
///     15.92
#[pyfunction]
#[pyo3(signature = (load, snr_db, receiver="mmse"))]
fn cdma_sinr_large_system(load: f64, snr_db: &Bound<'_, PyAny>, receiver: &str) -> PyResult<Py<PyAny>> {
    let c = criterion(receiver)?;
    map_real(snr_db, |s| {
        Ok(comnum::math::lin2db(dsss_theory::sinr_large_system(c, load, comnum::math::db2lin(s)).py()?))
    })
}

/// The large-system SINR (linear) of the MMSE receiver of a user among interferers of
/// any powers, random signatures of `n` chips: the fixed point of eq. (6) of Tse and
/// Hanly 1999, every interferer counting for the effective interference `P P_i / (P +
/// P_i γ)`.
///
/// Args:
///     power (float): the received power of the user.
///     interferers (array_like): the received powers of the others.
///     n (int): the spreading factor.
///     noise_var (float): noise variance per chip.
///
/// Returns:
///     float: the SINR (linear).
///
/// Raises:
///     ValueError: unless the powers, `n` and the noise variance are positive.
#[pyfunction]
fn cdma_sinr_mmse_large_system(power: f64, interferers: &Bound<'_, PyAny>, n: usize, noise_var: f64) -> PyResult<f64> {
    dsss_theory::sinr_mmse_large_system(power, &fvec(interferers)?, n, noise_var).py()
}

/// The large-system spectral efficiency (bits per chip, real-valued chips) of DS-CDMA
/// with random signatures (Verdú and Shamai 1999, eqs. (9)–(12)): a linear receiver
/// followed by single-user decoders, or the optimum joint decoder. Give the SNR, or
/// the `Eb/N0`, in which case the SNR follows from the efficiency itself (their Fig. 1).
///
/// Args:
///     load (float | array_like): users per chip, `K/N`.
///     receiver (str): `"mr"`, `"zf"`, `"mmse"` or `"optimal"`.
///     snr_db (float | None): `A²/σ²` (dB).
///     ebn0_db (float | None): `Eb/N0` (dB).
///
/// Returns:
///     float | numpy.ndarray: bits per chip.
///
/// Raises:
///     ValueError: unless exactly one of `snr_db` and `ebn0_db` is given, and for an
///         invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> round(comnum.cdma_spectral_efficiency(0.5, "zf", ebn0_db=10.0), 3)
///     1.145
#[pyfunction]
#[pyo3(signature = (load, receiver="mmse", *, snr_db=None, ebn0_db=None))]
fn cdma_spectral_efficiency(
    load: &Bound<'_, PyAny>,
    receiver: &str,
    snr_db: Option<f64>,
    ebn0_db: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let linear = if receiver.eq_ignore_ascii_case("optimal") { None } else { Some(criterion(receiver)?) };
    let at_snr = |load: f64, snr: f64| match linear {
        Some(c) => dsss_theory::spectral_efficiency(c, load, snr),
        None => dsss_theory::spectral_efficiency_optimal(load, snr),
    };
    match (snr_db, ebn0_db) {
        (Some(s), None) => map_real(load, |b| at_snr(b, comnum::math::db2lin(s)).py()),
        (None, Some(e)) => map_real(load, |b| {
            dsss_theory::spectral_efficiency_at_ebn0(|snr| at_snr(b, snr), b, comnum::math::db2lin(e)).py()
        }),
        _ => Err(PyValueError::new_err("give exactly one of snr_db and ebn0_db")),
    }
}

/// Register the bindings of this module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDsss>()?;
    m.add_function(wrap_pyfunction!(m_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(gold_codes, m)?)?;
    m.add_function(wrap_pyfunction!(kasami_codes, m)?)?;
    m.add_function(wrap_pyfunction!(walsh_codes, m)?)?;
    m.add_function(wrap_pyfunction!(barker_code, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(aperiodic_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_cdma, m)?)?;
    m.add_function(wrap_pyfunction!(cdma_near_far_resistance, m)?)?;
    m.add_function(wrap_pyfunction!(cdma_sinr_large_system, m)?)?;
    m.add_function(wrap_pyfunction!(cdma_sinr_mmse_large_system, m)?)?;
    m.add_function(wrap_pyfunction!(cdma_spectral_efficiency, m)?)?;
    Ok(())
}
