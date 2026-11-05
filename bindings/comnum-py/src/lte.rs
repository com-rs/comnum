//! LTE: the frames of a cell, the cell search and the master information block.

use crate::rng::PyRng;
use crate::{CArr, IntoPy, carr, cvec, mat};
use comnum::systems::lte::cell::Downlink;
use comnum::systems::lte::mib::{Mib, PhichDuration, PhichResource};
use comnum::systems::lte::search::{self, FoundCell, SearchConfig};
use num_complex::Complex64 as C64;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// The PHICH resource of a value of `N_g`.
fn phich_resource(ng: f64) -> PyResult<PhichResource> {
    [
        (1.0 / 6.0, PhichResource::OneSixth),
        (0.5, PhichResource::Half),
        (1.0, PhichResource::One),
        (2.0, PhichResource::Two),
    ]
    .iter()
    .find(|(v, _)| (v - ng).abs() < 1e-9)
    .map(|(_, r)| *r)
    .ok_or_else(|| PyValueError::new_err(format!("phich_ng is 1/6, 1/2, 1 or 2, got {ng}")))
}

/// The samples of every antenna port of one 10 ms frame of an LTE cell (FDD, normal
/// cyclic prefix): the PSS and the SSS of slots 0 and 10 (port 0), the CRS of every port,
/// the PBCH of slot 1 carrying the master information block of the frame (its 8 bits of
/// frame number `sfn // 4`, its quarter `sfn % 4` of the 40 ms period, transmit
/// diversity for 2 or 4 ports), and QPSK symbols at random on the other elements of port
/// 0 when `rng` is given (TS 36.211 Sec. 6.6, 6.10.1, 6.11).
///
/// Args:
///     n_cell_id (int): the cell identity, 0 to 503.
///     sfn (int): the system frame number.
///     n_rb (int): the bandwidth in resource blocks, 6, 15, 25, 50, 75 or 100.
///     ports (int): 1, 2 or 4 antenna ports.
///     n_fft (int | None): the FFT size (the sampling rate `n_fft · 15` kHz); the usual one
///         of the bandwidth (128 to 2048) if None.
///     phich_extended (bool): the PHICH duration of the MIB.
///     phich_ng (float): the PHICH resource `N_g` of the MIB, 1/6, 1/2, 1 or 2.
///     rng (Rng | None): the generator of the data symbols (none if None).
///
/// Returns:
///     numpy.ndarray: `(ports, 10 ms of samples)` (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum
///     >>> comnum.lte_frame(77, sfn=41, ports=2).shape                # 6 RB at 1.92 MS/s
///     (2, 19200)
#[pyfunction]
#[pyo3(signature = (n_cell_id, *, sfn = 0, n_rb = 6, ports = 1, n_fft = None, phich_extended = false, phich_ng = 1.0, rng = None))]
#[allow(clippy::too_many_arguments)]
fn lte_frame<'py>(
    py: Python<'py>,
    n_cell_id: u32,
    sfn: usize,
    n_rb: usize,
    ports: usize,
    n_fft: Option<usize>,
    phich_extended: bool,
    phich_ng: f64,
    rng: Option<PyRefMut<'py, PyRng>>,
) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let n_fft = match n_fft {
        Some(n) => n,
        None => comnum::waveform::ofdm::lte::fft_len(n_rb).py()?,
    };
    let mib = Mib {
        n_rb,
        phich_duration: if phich_extended { PhichDuration::Extended } else { PhichDuration::Normal },
        phich_resource: phich_resource(phich_ng)?,
        sfn_msb: 0,
        rest: [0; 10],
    };
    let cell = Downlink { n_cell_id, mib, ports };
    let mut rng = rng;
    let x = cell.frame_samples(sfn, n_fft, rng.as_mut().map(|r| &mut r.inner)).py()?;
    mat(py, x)
}

/// The cell search of LTE in samples at 1.92 MS/s (the six central resource blocks:
/// :func:`select_channel` brings a capture there): the PSS of each of the three
/// identities `N_ID^(2)` is searched over a grid of frequency offsets in the first 10 ms,
/// the offset refined on the cyclic prefixes of the whole capture, then the SSS of every
/// half frame, equalised by the channel of the PSS, gives the group `N_ID^(1)` and the
/// start of the frames.
///
/// Args:
///     x (array_like): at least 10.5 ms of complex samples at 1.92 MS/s.
///     max_cfo_hz (float): the largest frequency offset searched (Hz).
///     cfo_step_hz (float): the step of the grid of offsets (Hz).
///     min_metric (float): the smallest normalised PSS correlation of a cell, in (0, 1].
///
/// Returns:
///     list[dict]: the strongest cell of each `N_ID^(2)` found, the strongest first, with
///     the keys ``n_cell_id``, ``cfo_hz`` (the offset of the cell with respect to the
///     receiver), ``frame_start`` (the first sample of the first whole frame),
///     ``pss_metric`` and ``sss_metric`` (normalised correlations, at most 1).
///
/// Raises:
///     ValueError: for fewer than 10.5 ms of samples or invalid settings.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> x = np.concatenate([comnum.lte_frame(211, sfn=s)[0] for s in (8, 9)])
///     >>> cell = comnum.lte_cell_search(x, max_cfo_hz=1e3)[0]
///     >>> cell["n_cell_id"], cell["frame_start"]
///     (211, 0)
#[pyfunction]
#[pyo3(signature = (x, *, max_cfo_hz = 20e3, cfo_step_hz = 1e3, min_metric = 0.3))]
fn lte_cell_search<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    max_cfo_hz: f64,
    cfo_step_hz: f64,
    min_metric: f64,
) -> PyResult<Bound<'py, PyList>> {
    let x = cvec(x)?;
    let cells = py.detach(|| search::cell_search(&x, &SearchConfig { max_cfo_hz, cfo_step_hz, min_metric })).py()?;
    let out = PyList::empty(py);
    for c in cells {
        let d = PyDict::new(py);
        d.set_item("n_cell_id", c.n_cell_id)?;
        d.set_item("cfo_hz", c.cfo_hz)?;
        d.set_item("frame_start", c.frame_start)?;
        d.set_item("pss_metric", c.pss_metric)?;
        d.set_item("sss_metric", c.sss_metric)?;
        out.append(d)?;
    }
    Ok(out)
}

/// The master information block of every frame of a capture at 1.92 MS/s whose PBCH
/// decodes, for a cell of :func:`lte_cell_search`: the channel of every port estimated
/// on the CRS of subframe 0, the PBCH equalised for 1, 2 and 4 ports (transmit
/// diversity), descrambled for each quarter of the 40 ms period, decoded by the
/// wrap-around Viterbi algorithm (tail-biting code) and checked by its CRC, whose mask
/// tells the number of ports (TS 36.212 Sec. 5.3.1).
///
/// Args:
///     x (array_like): the complex samples at 1.92 MS/s.
///     cell (dict): a cell of :func:`lte_cell_search` (``n_cell_id``, ``cfo_hz``,
///         ``frame_start``).
///     passes (int): the passes of the wrap-around Viterbi algorithm.
///
/// Returns:
///     list[dict]: one per decoded frame, with the keys ``start`` (its first sample),
///     ``sfn`` (the system frame number), ``ports``, ``n_rb`` (the downlink bandwidth),
///     ``phich_extended`` (bool), ``phich_ng`` and ``spare`` (the last 10 bits of the MIB).
///
/// Raises:
///     ValueError: for a cell without its keys, an identity above 503 or no pass.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> x = np.concatenate([comnum.lte_frame(211, sfn=s, ports=2)[0] for s in (8, 9)])
///     >>> cell = comnum.lte_cell_search(x, max_cfo_hz=1e3)[0]
///     >>> [(f["sfn"], f["ports"], f["n_rb"]) for f in comnum.lte_decode_mib(x, cell)]
///     [(8, 2, 6), (9, 2, 6)]
#[pyfunction]
#[pyo3(signature = (x, cell, *, passes = 4))]
fn lte_decode_mib<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    cell: &Bound<'py, PyDict>,
    passes: usize,
) -> PyResult<Bound<'py, PyList>> {
    let x = cvec(x)?;
    let get = |k: &str| -> PyResult<Bound<'py, PyAny>> {
        cell.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("the cell has no key {k:?}")))
    };
    let c = FoundCell {
        n_cell_id: get("n_cell_id")?.extract()?,
        cfo_hz: get("cfo_hz")?.extract()?,
        frame_start: get("frame_start")?.extract()?,
        pss_metric: 0.0,
        sss_metric: 0.0,
    };
    let frames = py.detach(|| search::decode_mib(&x, &c, passes)).py()?;
    let out = PyList::empty(py);
    for f in frames {
        let d = PyDict::new(py);
        d.set_item("start", f.start)?;
        d.set_item("sfn", f.sfn)?;
        d.set_item("ports", f.ports)?;
        d.set_item("n_rb", f.mib.n_rb)?;
        d.set_item("phich_extended", f.mib.phich_duration == PhichDuration::Extended)?;
        d.set_item("phich_ng", f.mib.phich_resource.ng())?;
        d.set_item("spare", f.mib.rest.iter().map(|&b| u32::from(b)).collect::<Vec<_>>())?;
        out.append(d)?;
    }
    Ok(out)
}

/// The PSS of LTE of the identity `N_ID^(2)` (the 62 values of the punctured Zadoff–Chu
/// sequence of length 63, roots 25, 29, 34; TS 36.211 Sec. 6.11.1.1).
///
/// Args:
///     n_id2 (int): 0, 1 or 2.
///
/// Returns:
///     numpy.ndarray: 62 values (complex128).
///
/// Raises:
///     ValueError: for an identity above 2.
#[pyfunction]
fn lte_pss(py: Python<'_>, n_id2: u32) -> PyResult<CArr<'_>> {
    Ok(carr(py, comnum::waveform::ofdm::lte::pss(n_id2).py()?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lte_frame, m)?)?;
    m.add_function(wrap_pyfunction!(lte_cell_search, m)?)?;
    m.add_function(wrap_pyfunction!(lte_decode_mib, m)?)?;
    m.add_function(wrap_pyfunction!(lte_pss, m)?)?;
    Ok(())
}
