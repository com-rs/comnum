//! The sources: random bits, PRBS and symbols.

use super::*;

/// Independent fair bits.
///
/// Args:
///     n (int): number of bits.
///     rng (Rng): the generator.
///
/// Returns:
///     numpy.ndarray: `n` bits (uint8).
///
/// Raises:
///     ValueError: for a negative `n`.
///
/// Examples:
///     >>> import comnum
///     >>> b = comnum.bits(8, comnum.Rng(1))
///     >>> b.dtype, b.size, set(b.tolist()) <= {0, 1}
///     (dtype('uint8'), 8, True)
#[pyfunction]
fn bits<'py>(py: Python<'py>, n: i64, mut rng: PyRefMut<'py, PyRng>) -> PyResult<BArr<'py>> {
    Ok(sources::bits(count(n, "n")?, &mut rng.inner).into_pyarray(py))
}

/// A maximal-length pseudo-random binary sequence (ITU-T O.150 polynomials).
///
/// Args:
///     order (int): 7, 9, 11, 15, 20, 23, 29 or 31 (period `2^order - 1`); the
///         polynomials of ITU-T O.150 at another phase, never inverted: see
///         :func:`o150_sequence` for the sequences of the Recommendation.
///     length (int): number of bits.
///     seed (int): initial register state (non-zero after masking).
///
/// Returns:
///     numpy.ndarray: `length` bits (uint8).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> int(comnum.prbs(7, 127).sum())
///     64
#[pyfunction]
#[pyo3(signature = (order, length, *, seed = 1))]
fn prbs(py: Python<'_>, order: u32, length: i64, seed: u64) -> PyResult<BArr<'_>> {
    Ok(sources::prbs(order, count(length, "length")?, seed).py()?.into_pyarray(py))
}

/// A pseudo-random test sequence of ITU-T O.150 (05/96), Sec. 5.
///
/// The eight sequences of the error-rate test sets, exactly as the Recommendation
/// defines them: the shift register of `n` stages fed back from the stages `k` and `n`,
/// the output inverted for `prbs15`, `prbs23`, `prbs29` and `prbs31`, the zero
/// suppression of `prbs20z` (Sec. 5.5). Every register starts with its stages at one,
/// the phase of Sec. 5.1 ("the first ONE of 9 consecutive ONEs"); an inverted sequence
/// then begins with its `n` zeros.
///
/// Args:
///     name (str): `prbs9`, `prbs11`, `prbs15`, `prbs20`, `prbs20z`, `prbs23`,
///         `prbs29` or `prbs31` (period `2^n - 1`).
///     length (int): number of bits.
///
/// Returns:
///     numpy.ndarray: `length` bits (uint8).
///
/// Raises:
///     ValueError: for an unknown name or a negative length.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.o150_sequence("prbs9", 10).tolist()
///     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
///     >>> int(comnum.o150_sequence("prbs15", 32767).sum())
///     16383
#[pyfunction]
fn o150_sequence<'py>(py: Python<'py>, name: &str, length: i64) -> PyResult<BArr<'py>> {
    let seq = sources::O150Sequence::from_name(name).py()?;
    Ok(seq.bits(count(length, "length")?).into_pyarray(py))
}

/// Random symbols and their indices.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     n (int): number of symbols.
///     rng (Rng): the generator.
///     px (array_like, optional): probabilities of the points (uniform by default).
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the indices (int64) and the symbols
///     (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (constellation, n, rng, *, px = None))]
fn symbols<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    n: i64,
    mut rng: PyRefMut<'py, PyRng>,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<(IArr<'py>, CArr<'py>)> {
    let px = px.map(crate::fvec).transpose()?;
    let (idx, x) = sources::symbols(&constellation.inner, count(n, "n")?, px.as_deref(), &mut rng.inner).py()?;
    Ok((iarr(py, idx), carr(py, x)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bits, m)?)?;
    m.add_function(wrap_pyfunction!(prbs, m)?)?;
    m.add_function(wrap_pyfunction!(o150_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(symbols, m)?)?;
    Ok(())
}
