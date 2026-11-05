//! The PDSCH of 5G NR: MCS and TBS, the transport channel with HARQ, and the slot.

use crate::comm::PyConstellation;
use crate::{BArr, FArr, IntoPy, bvec, carray, farr, fvec, uvec};
use comnum::systems::nr::mcs::{self, McsTable};
use comnum::systems::nr::modulation;
use comnum::systems::nr::pdsch::{Pdsch, PdschReceiver, SYMBOLS_PER_SLOT};
use comnum::systems::nr::transport::{HarqProcess, TransportBlock};
use comnum::waveform::ofdm::nr::PdschDmrs;
use ndarray::Array3;
use num_complex::Complex64 as C64;
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn table(name: &str) -> PyResult<McsTable> {
    Ok(match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
        "qam64" | "64qam" => McsTable::Qam64,
        "qam256" | "256qam" => McsTable::Qam256,
        "qam64lowse" | "64qamlowse" => McsTable::Qam64LowSe,
        "qam1024" | "1024qam" => McsTable::Qam1024,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown MCS table {name:?}: qam64, qam256, qam64lowse, qam1024"
            )));
        }
    })
}

/// An MCS of the PDSCH (TS 38.214 Tables 5.1.3.1-1 to 5.1.3.1-4).
///
/// Args:
///     table (str): ``"qam64"``, ``"qam256"``, ``"qam64lowse"`` or ``"qam1024"``.
///     index (int): the MCS index.
///
/// Returns:
///     tuple[int, float]: the bits per symbol ``Q_m`` and the target code rate times 1024.
///
/// Raises:
///     ValueError: for an unknown table or a reserved index.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.nr_mcs("qam64", 4)
///     (2, 308.0)
#[pyfunction]
fn nr_mcs(table: &str, index: usize) -> PyResult<(usize, f64)> {
    let m = mcs::mcs(self::table(table)?, index).py()?;
    Ok((m.qm, m.rate_x1024))
}

/// The transport block size of a PDSCH (TS 38.214 Sec. 5.1.3.2).
///
/// Args:
///     n_prb (int): the allocated PRBs.
///     n_symb (int): the PDSCH symbols of the slot.
///     n_dmrs (int): the DM-RS REs per PRB, the CDM groups without data included.
///     table (str): the MCS table, as ``nr_mcs``.
///     index (int): the MCS index.
///     layers (int): the layers.
///     n_oh (int): the overhead per PRB (``xOverhead``).
///
/// Returns:
///     int: the TBS (bits).
///
/// Raises:
///     ValueError: for an invalid parameter.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.nr_tbs(6, 12, 12, "qam64", 4)    # R.PDSCH.1-1.2 of TS 38.101-4
///     480
#[pyfunction]
#[pyo3(signature = (n_prb, n_symb, n_dmrs, table, index, layers=1, n_oh=0))]
fn nr_tbs(
    n_prb: usize,
    n_symb: usize,
    n_dmrs: usize,
    table: &str,
    index: usize,
    layers: usize,
    n_oh: usize,
) -> PyResult<usize> {
    let m = mcs::mcs(self::table(table)?, index).py()?;
    mcs::transport_block_size(n_prb, n_symb, n_dmrs, n_oh, m, layers).py()
}

/// The constellation of the modulation mapper of NR (TS 38.211 Sec. 5.1).
///
/// Args:
///     qm (int): the bits per symbol, 2 (QPSK) to 10 (1024-QAM).
///
/// Returns:
///     Constellation: the points, ``points[i]`` labelled by the bits of ``i``.
///
/// Raises:
///     ValueError: for another ``qm``.
#[pyfunction]
fn nr_constellation(qm: usize) -> PyResult<PyConstellation> {
    Ok(PyConstellation { inner: modulation::constellation(qm).py()? })
}

/// A transport block of the PDSCH and its segmentation (TS 38.212 Sec. 7.2): TB CRC,
/// code blocks with their CRC24B, LDPC and rate matching.
///
/// Args:
///     tbs (int): the transport block size (bits).
///     rate (float): the target code rate ``R`` (it selects the base graph).
///
/// Raises:
///     ValueError: for an empty block or a rate outside (0, 1).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> tb = comnum.NrTransportBlock(480, 308 / 1024)
///     >>> tb.code_blocks, tb.base_graph, tb.encode(np.zeros(480, np.uint8), 1584, 2).shape
///     (1, 2, (1584,))
#[pyclass(name = "NrTransportBlock", module = "comnum")]
pub struct PyNrTransportBlock {
    inner: TransportBlock,
}

#[pymethods]
impl PyNrTransportBlock {
    #[new]
    fn new(tbs: usize, rate: f64) -> PyResult<Self> {
        Ok(PyNrTransportBlock { inner: TransportBlock::new(tbs, rate).py()? })
    }

    /// int: The transport block size.
    #[getter]
    fn tbs(&self) -> usize {
        self.inner.tbs()
    }

    /// int: The code blocks ``C``.
    #[getter]
    fn code_blocks(&self) -> usize {
        self.inner.code_blocks()
    }

    /// int: The LDPC base graph, 1 or 2.
    #[getter]
    fn base_graph(&self) -> u8 {
        self.inner.base_graph()
    }

    /// int: The bits per code block ``K'``, its CRC included.
    #[getter]
    fn code_block_bits(&self) -> usize {
        self.inner.code_block_bits()
    }

    /// The rate-matched length ``E_r`` of every code block (TS 38.212 Sec. 5.4.2.1).
    ///
    /// Args:
    ///     g (int): the channel bits of the transport block.
    ///     qm (int): the bits per symbol.
    ///     layers (int): the layers.
    ///
    /// Returns:
    ///     list[int]: the lengths.
    ///
    /// Raises:
    ///     ValueError: unless ``g`` is a positive multiple of ``qm * layers``.
    #[pyo3(signature = (g, qm, layers=1))]
    fn rate_matched_lengths(&self, g: usize, qm: usize, layers: usize) -> PyResult<Vec<usize>> {
        self.inner.rate_matched_lengths(g, qm, layers).py()
    }

    /// The ``g`` channel bits of the transport block (before scrambling).
    ///
    /// Args:
    ///     bits (array_like): the ``tbs`` bits.
    ///     g (int): the channel bits.
    ///     qm (int): the bits per symbol (the bit interleaver).
    ///     layers (int): the layers.
    ///     rv (int): the redundancy version, 0 to 3.
    ///
    /// Returns:
    ///     numpy.ndarray: the bits (uint8).
    ///
    /// Raises:
    ///     ValueError: for a block of another size or an invalid ``g``.
    #[pyo3(signature = (bits, g, qm, layers=1, rv=0))]
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        bits: &Bound<'py, PyAny>,
        g: usize,
        qm: usize,
        layers: usize,
        rv: u8,
    ) -> PyResult<BArr<'py>> {
        Ok(self.inner.encode(&bvec(bits)?, g, qm, layers, rv).py()?.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "NrTransportBlock(tbs={}, C={}, BG{})",
            self.inner.tbs(),
            self.inner.code_blocks(),
            self.inner.base_graph()
        )
    }
}

/// The soft buffers of a HARQ process of a transport block: the LLRs of every
/// transmission combined on the bits of the full codewords.
///
/// Args:
///     tb (NrTransportBlock): the transport block.
///
/// Raises:
///     ValueError: if the code cannot be built.
#[pyclass(name = "NrHarqProcess", module = "comnum")]
pub struct PyNrHarqProcess {
    inner: HarqProcess,
}

#[pymethods]
impl PyNrHarqProcess {
    #[new]
    fn new(mut tb: PyRefMut<'_, PyNrTransportBlock>) -> PyResult<Self> {
        Ok(PyNrHarqProcess { inner: HarqProcess::new(&mut tb.inner).py()? })
    }

    /// int: The transmissions combined so far.
    #[getter]
    fn transmissions(&self) -> usize {
        self.inner.transmissions()
    }

    /// Add the LLRs of one transmission (positive for a 0, descrambled).
    ///
    /// Args:
    ///     tb (NrTransportBlock): the transport block.
    ///     llr (array_like): the ``g`` LLRs.
    ///     qm (int): the bits per symbol.
    ///     layers (int): the layers.
    ///     rv (int): the redundancy version of the transmission.
    ///
    /// Raises:
    ///     ValueError: for an invalid length.
    #[pyo3(signature = (tb, llr, qm, layers=1, rv=0))]
    fn add(
        &mut self,
        mut tb: PyRefMut<'_, PyNrTransportBlock>,
        llr: &Bound<'_, PyAny>,
        qm: usize,
        layers: usize,
        rv: u8,
    ) -> PyResult<()> {
        self.inner.add(&mut tb.inner, &fvec(llr)?, qm, layers, rv).py()
    }

    /// Decode the buffers.
    ///
    /// Args:
    ///     tb (NrTransportBlock): the transport block.
    ///
    /// Returns:
    ///     numpy.ndarray | None: the transport block when its CRCs pass, None otherwise.
    ///
    /// Raises:
    ///     ValueError: if the decoder fails.
    fn decode<'py>(&self, py: Python<'py>, mut tb: PyRefMut<'py, PyNrTransportBlock>) -> PyResult<Option<BArr<'py>>> {
        Ok(self.inner.decode(&mut tb.inner).py()?.map(|b| b.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        format!("NrHarqProcess({} transmissions)", self.inner.transmissions())
    }
}

/// One slot of a single-layer PDSCH (TS 38.211 Sec. 7.3.1): scrambling, modulation,
/// resource mapping around the DM-RS of port 1000 (configuration type 1, one CDM group
/// without data), a precoder per PRG, and a receiver that estimates the precoded channel
/// on the DM-RS by robust 2-D Wiener filtering.
///
/// Args:
///     n_rb_grid (int): the resource blocks of the carrier.
///     prb_start (int): the first PRB of the allocation.
///     n_prb (int): the PRBs of the allocation.
///     qm (int): the bits per symbol.
///     first_symbol (int): the first PDSCH symbol.
///     n_symbols (int): the PDSCH symbols.
///     dmrs_type_a_position (int): 2 or 3.
///     dmrs_additional_position (int): 0 to 3.
///     n_ports (int): the transmit ports, 1 or 2.
///     prg_size (int): the PRBs per precoding group.
///     rnti (int): the RNTI of the scrambling.
///     n_id (int): the scrambling identity.
///     dmrs_n_id (int): the DM-RS scrambling identity.
///     slot (int): the slot in the frame (the DM-RS sequence).
///     subcarrier_spacing_hz (float): the subcarrier spacing.
///
/// Raises:
///     ValueError: for an invalid configuration (when used).
///
/// Examples:
///     >>> import comnum
///     >>> p = comnum.NrPdsch(52, 23, 6, 2, dmrs_additional_position=1)
///     >>> p.g(), p.n_prgs
///     (1584, 4)
#[pyclass(name = "NrPdsch", module = "comnum")]
pub struct PyNrPdsch {
    inner: Pdsch,
}

fn reserved_in(reserved: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<(usize, usize)>> {
    let Some(r) = reserved else { return Ok(Vec::new()) };
    let flat = uvec(&r.call_method1("reshape", (-1,))?)?;
    if flat.len() % 2 != 0 {
        return Err(PyValueError::new_err("reserved must be (n, 2) pairs (symbol, subcarrier)"));
    }
    Ok(flat.chunks(2).map(|c| (c[0], c[1])).collect())
}

#[pymethods]
impl PyNrPdsch {
    #[new]
    #[pyo3(signature = (n_rb_grid, prb_start, n_prb, qm, *, first_symbol=2, n_symbols=12, dmrs_type_a_position=2, dmrs_additional_position=0, n_ports=1, prg_size=2, rnti=1, n_id=0, dmrs_n_id=0, slot=0, subcarrier_spacing_hz=15e3))]
    #[allow(clippy::too_many_arguments, reason = "the parameters of a PDSCH, all but four optional")]
    fn new(
        n_rb_grid: usize,
        prb_start: usize,
        n_prb: usize,
        qm: usize,
        first_symbol: usize,
        n_symbols: usize,
        dmrs_type_a_position: usize,
        dmrs_additional_position: usize,
        n_ports: usize,
        prg_size: usize,
        rnti: u32,
        n_id: u32,
        dmrs_n_id: u32,
        slot: u32,
        subcarrier_spacing_hz: f64,
    ) -> Self {
        let dmrs = PdschDmrs {
            config_type: 1,
            type_a_position: dmrs_type_a_position,
            additional_position: dmrs_additional_position,
            n_id: dmrs_n_id,
            slot,
        };
        PyNrPdsch {
            inner: Pdsch {
                n_rb_grid,
                prb_start,
                n_prb,
                first_symbol,
                n_symbols,
                dmrs,
                n_ports,
                prg_size,
                qm,
                rnti,
                n_id,
                subcarrier_spacing_hz,
            },
        }
    }

    /// int: The precoding resource block groups of the allocation.
    #[getter]
    fn n_prgs(&self) -> usize {
        self.inner.n_prgs()
    }

    /// The channel bits ``G`` of the slot.
    ///
    /// Args:
    ///     reserved (array_like | None): REs ``(symbol, subcarrier)`` not used for data.
    ///
    /// Returns:
    ///     int: ``G``.
    ///
    /// Raises:
    ///     ValueError: for an invalid configuration.
    #[pyo3(signature = (reserved=None))]
    fn g(&self, reserved: Option<&Bound<'_, PyAny>>) -> PyResult<usize> {
        self.inner.g(&reserved_in(reserved)?).py()
    }

    /// The data REs of the slot in their mapping order (frequency first).
    ///
    /// Args:
    ///     reserved (array_like | None): REs ``(symbol, subcarrier)`` not used for data.
    ///
    /// Returns:
    ///     numpy.ndarray: ``(n, 2)`` pairs ``(symbol, subcarrier)``.
    ///
    /// Raises:
    ///     ValueError: for an invalid configuration.
    #[pyo3(signature = (reserved=None))]
    fn data_res<'py>(
        &self,
        py: Python<'py>,
        reserved: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let res = self.inner.data_res(&reserved_in(reserved)?).py()?;
        let rows: Vec<Vec<i64>> = res.iter().map(|&(l, k)| vec![l as i64, k as i64]).collect();
        crate::mat(py, rows)
    }

    /// The grids of the slot carrying the ``g`` channel bits.
    ///
    /// Args:
    ///     bits (array_like): the channel bits (``NrTransportBlock.encode``).
    ///     precoders (array_like | None): the precoder index of every PRG (0 to 3), zeros
    ///         by default.
    ///     reserved (array_like | None): REs ``(symbol, subcarrier)`` left empty.
    ///
    /// Returns:
    ///     numpy.ndarray: ``(n_ports, 14, 12 n_rb_grid)`` resource elements.
    ///
    /// Raises:
    ///     ValueError: for bits of another length or one precoder per PRG missing.
    #[pyo3(signature = (bits, precoders=None, reserved=None))]
    fn transmit<'py>(
        &self,
        py: Python<'py>,
        bits: &Bound<'py, PyAny>,
        precoders: Option<&Bound<'py, PyAny>>,
        reserved: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray3<C64>>> {
        let pre = match precoders {
            Some(p) => uvec(p)?,
            None => vec![0; self.inner.n_prgs()],
        };
        let grids = self.inner.transmit(&bvec(bits)?, &pre, &reserved_in(reserved)?).py()?;
        let k = grids[0].len() / SYMBOLS_PER_SLOT;
        let flat: Vec<C64> = grids.into_iter().flatten().collect();
        let a = Array3::from_shape_vec((flat.len() / (SYMBOLS_PER_SLOT * k), SYMBOLS_PER_SLOT, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(a.into_pyarray(py))
    }

    /// The descrambled LLRs of a received slot.
    ///
    /// Args:
    ///     rx (array_like): ``(n_rx, 14, 12 n_rb_grid)`` received resource elements.
    ///     noise_var (float): the noise variance per RE and antenna.
    ///     max_delay_s (float): the largest delay the estimator expects.
    ///     max_doppler_hz (float): the largest Doppler shift it expects.
    ///     reserved (array_like | None): REs ``(symbol, subcarrier)`` without data.
    ///
    /// Returns:
    ///     numpy.ndarray: the ``g`` LLRs (positive for a 0).
    ///
    /// Raises:
    ///     ValueError: for grids of another size or a non-positive noise variance.
    #[pyo3(signature = (rx, noise_var, *, max_delay_s=4.69e-6, max_doppler_hz=100.0, reserved=None))]
    fn receive<'py>(
        &self,
        py: Python<'py>,
        rx: &Bound<'py, PyAny>,
        noise_var: f64,
        max_delay_s: f64,
        max_doppler_hz: f64,
        reserved: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<FArr<'py>> {
        let a = carray(rx)?;
        if a.ndim() != 3 {
            return Err(PyValueError::new_err("rx must be (n_rx, 14, K)"));
        }
        let n_rx = a.shape()[0];
        let flat: Vec<C64> = a.iter().copied().collect();
        let per = flat.len() / n_rx.max(1);
        let grids: Vec<Vec<C64>> = flat.chunks(per.max(1)).map(<[C64]>::to_vec).collect();
        let mut receiver = PdschReceiver::new(max_delay_s, max_doppler_hz);
        Ok(farr(py, self.inner.receive(&grids, noise_var, &mut receiver, &reserved_in(reserved)?).py()?))
    }

    fn __repr__(&self) -> String {
        format!("NrPdsch({} PRBs from {}, Qm {})", self.inner.n_prb, self.inner.prb_start, self.inner.qm)
    }
}

/// Register the PDSCH bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nr_mcs, m)?)?;
    m.add_function(wrap_pyfunction!(nr_tbs, m)?)?;
    m.add_function(wrap_pyfunction!(nr_constellation, m)?)?;
    m.add_class::<PyNrTransportBlock>()?;
    m.add_class::<PyNrHarqProcess>()?;
    m.add_class::<PyNrPdsch>()?;
    Ok(())
}
