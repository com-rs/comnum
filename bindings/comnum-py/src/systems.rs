//! Standard-compliant chains: DVB-S, DVB-S2, DVB-T, and the convolutional code, the
//! randomizers and the markers of CCSDS 131.0-B-5.

use crate::comm::PyConstellation;
use crate::rng::PyRng;
use crate::{BArr, CArr, CMat, FArr, IntoPy, bvec, bytevec, carr, cvec, farr, fvec, grid_in, grid_out};
use comnum::C64;
use comnum::channel::multipath::Echoes;
use comnum::fec::ccsds::TurboRate;
use comnum::math::Jacobian;
use comnum::systems::ccsds::{self, Randomizer};
use comnum::systems::dvb::{self, InnerRate};
use comnum::systems::dvbs::Dvbs;
use comnum::systems::dvbs2::{self, Dvbs2, FrameSize, Modcod};
use comnum::systems::dvbt::{Dvbt, Modulation, Tps};
use comnum::waveform::ofdm::dvbt::{Guard, Mode};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A DVB-S2 transmitter and receiver (ETSI EN 302 307-1) of one MODCOD, constant coding
/// and modulation.
///
/// The transmitter takes a BBFRAME of ``k`` bits (see :func:`dvbs2_bbframe`), encodes it
/// (BCH, then the LDPC code of the standard), interleaves the bits (8PSK, 16APSK,
/// 32APSK), maps them on the constellation of the MODCOD and builds the PLFRAME: the
/// PLHEADER in π/2-BPSK, the slots of 90 symbols, pilot blocks of 36 symbols every 16
/// slots, and the Gold scrambling sequence number ``scrambling_code``. Every symbol has
/// unit energy, so that a noise of variance ``1/(Es/N0)`` gives the abscissa of the
/// performance table of the standard (:attr:`threshold_db`). The receiver assumes the
/// frame found and the carrier recovered.
///
/// Args:
///     modulation (str): ``"QPSK"``, ``"8PSK"``, ``"16APSK"`` or ``"32APSK"``.
///     rate (str): ``"1/4"`` … ``"9/10"`` (the 28 MODCODs of Table 12).
///     frame (str): ``"normal"`` or ``"64800"`` (64 800 bits), ``"short"`` or ``"16200"``
///         (16 200 bits).
///     pilots (bool): pilot blocks.
///     scrambling_code (int): the number ``n`` of the Gold sequence.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> link = comnum.Dvbs2("8PSK", "3/5", frame="short")
///     >>> link.k, link.plframe_len
///     (9552, 5598)
///     >>> bb = comnum.dvbs2_bbframe(np.zeros(link.k - 80, dtype=np.uint8), link.k)
///     >>> x = link.modulate(bb)
///     >>> comnum.dvbs2_decode_plheader(x[:90])
///     (12, True, True)
///     >>> bool(np.array_equal(link.decode(x, 1e-3), bb))
///     True
#[pyclass(name = "Dvbs2", module = "comnum")]
pub struct PyDvbs2 {
    inner: Dvbs2,
}

#[pymethods]
impl PyDvbs2 {
    fn __repr__(&self) -> String {
        format!(
            "Dvbs2(modcod={}, k={}, plframe_len={})",
            self.inner.modcod().number(),
            self.inner.k(),
            self.inner.plframe_len()
        )
    }

    #[new]
    #[pyo3(signature = (modulation, rate, *, frame = "normal", pilots = true, scrambling_code = 0))]
    fn new(modulation: &str, rate: &str, frame: &str, pilots: bool, scrambling_code: usize) -> PyResult<Self> {
        let frame: FrameSize = frame.parse().py()?;
        let modcod = Modcod::from_name(modulation, rate).py()?;
        Ok(PyDvbs2 { inner: Dvbs2::new(modcod, frame, pilots, scrambling_code).py()? })
    }

    /// int: Bits of a BBFRAME (``K_bch``, Tables 5a and 5b).
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Symbols of a PLFRAME.
    #[getter]
    fn plframe_len(&self) -> usize {
        self.inner.plframe_len()
    }

    /// int: The MODCOD field of the PLHEADER (1 to 28, Table 12).
    #[getter]
    fn modcod(&self) -> PyResult<u8> {
        Ok(self.inner.modcod().number())
    }

    /// float: The ideal Es/N0 (dB) at a PER of 1e-7 of the normal frame (Table 13).
    #[getter]
    fn threshold_db(&self) -> f64 {
        self.inner.modcod().threshold_db()
    }

    /// Constellation: The constellation of the MODCOD (``points[l]`` carries the label
    /// ``l``).
    #[getter]
    fn constellation(&self) -> PyConstellation {
        PyConstellation { inner: self.inner.constellation().clone() }
    }

    /// int: The cap of the LDPC iterations (sum-product, layered, stopping on a
    /// codeword; read and write).
    #[getter]
    fn get_max_iterations(&self) -> usize {
        self.inner.fec().ldpc().decoder.max_iterations
    }

    #[setter]
    fn set_max_iterations(&mut self, n: usize) {
        self.inner.fec_mut().ldpc_mut().decoder.max_iterations = n;
    }

    /// The PLFRAME of a BBFRAME.
    ///
    /// Args:
    ///     bbframe (array_like): ``k`` bits (see :func:`dvbs2_bbframe`).
    ///
    /// Returns:
    ///     numpy.ndarray: the ``plframe_len`` unit-energy symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn modulate<'py>(&self, py: Python<'py>, bbframe: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let b = bvec(bbframe)?;
        let x = py.detach(|| self.inner.modulate(&b)).py()?;
        Ok(carr(py, x))
    }

    /// The deinterleaved bit LLRs of the FECFRAME of a received PLFRAME.
    ///
    /// Args:
    ///     rx (array_like): the received PLFRAME (``plframe_len`` symbols).
    ///     noise_var (float): complex noise variance per symbol.
    ///     max_log (bool): max-log LLRs instead of the exact ones.
    ///
    /// Returns:
    ///     numpy.ndarray: one LLR per bit of the FECFRAME (positive for a 0).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (rx, noise_var, *, max_log = false))]
    fn llr<'py>(&self, py: Python<'py>, rx: &Bound<'py, PyAny>, noise_var: f64, max_log: bool) -> PyResult<FArr<'py>> {
        let y = cvec(rx)?;
        let l = py.detach(|| self.inner.llr(&y, noise_var, Jacobian::from_max_log(max_log))).py()?;
        Ok(farr(py, l))
    }

    /// The BBFRAME of a received PLFRAME (LDPC, then BCH decoding).
    ///
    /// Args:
    ///     rx (array_like): the received PLFRAME (``plframe_len`` symbols).
    ///     noise_var (float): complex noise variance per symbol.
    ///
    /// Returns:
    ///     numpy.ndarray: the ``k`` bits of the BBFRAME (uint8).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn decode<'py>(&self, py: Python<'py>, rx: &Bound<'py, PyAny>, noise_var: f64) -> PyResult<BArr<'py>> {
        let y = cvec(rx)?;
        let b = py.detach(|| self.inner.decode(&y, noise_var)).py()?;
        Ok(b.into_pyarray(py))
    }
}

/// A DVB-S transmitter and frame-aligned receiver (ETSI EN 300 421 V1.1.2) of one inner
/// code rate.
///
/// The transmitter takes a burst of whole MPEG-2 transport packets (188 bytes, each
/// starting with the sync byte ``0x47``; see :func:`dvb_packets`): the energy dispersal
/// (the PRBS 1 + X^14 + X^15 reset every 8 packets, the first sync byte of a group
/// inverted to ``0xB8``), the outer code RS(204, 188), the convolutional interleaver
/// I = 12, M = 17 (flushed with 11 packets of zeros, so that the whole burst decodes), the
/// inner code of constraint length 7 (X from 171, Y from 133, octal) punctured by Table 2
/// of the standard, and the Gray QPSK of its Figure 5 with unit energy, at one sample per
/// symbol. The receiver computes the exact LLRs, decodes the inner code by soft Viterbi,
/// deinterleaves, decodes RS(204, 188) and removes the energy dispersal; it assumes the
/// burst found and the carrier recovered.
///
/// Args:
///     rate (str): the inner code rate, ``"1/2"``, ``"2/3"``, ``"3/4"``, ``"5/6"`` or
///         ``"7/8"``.
///
/// Raises:
///     ValueError: for another rate.
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> link = comnum.Dvbs("3/4")
///     >>> packets = comnum.dvb_packets(8, comnum.Rng(1))
///     >>> x = link.modulate(packets)
///     >>> len(x) == link.symbols_len(8)
///     True
///     >>> bool(np.array_equal(link.decode(x, 0.01)[0], packets))
///     True
///     >>> round(link.useful_bit_rate(25.776e6) / 1e6, 2)   # Table D.1: 35.631 Mbit/s
///     35.63
#[pyclass(name = "Dvbs", module = "comnum")]
pub struct PyDvbs {
    inner: Dvbs,
}

#[pymethods]
impl PyDvbs {
    fn __repr__(&self) -> String {
        format!("Dvbs(rate={:?})", self.inner.rate().name())
    }

    #[new]
    #[pyo3(signature = (rate = "1/2"))]
    fn new(rate: &str) -> PyResult<Self> {
        let rate: InnerRate = rate.parse().py()?;
        Ok(PyDvbs { inner: Dvbs::new(rate).py()? })
    }

    /// str: The inner code rate.
    #[getter]
    fn rate(&self) -> &'static str {
        self.inner.rate().name()
    }

    /// Constellation: The Gray QPSK of Figure 5 (label ``2 b_I + b_Q``, unit energy).
    #[getter]
    fn constellation(&self) -> PyResult<PyConstellation> {
        Ok(PyConstellation { inner: Dvbs::constellation().py()? })
    }

    /// The number of QPSK symbols of a burst of ``n_packets`` packets.
    ///
    /// Args:
    ///     n_packets (int): packets of the burst.
    ///
    /// Returns:
    ///     int: the symbols of :meth:`modulate`.
    fn symbols_len(&self, n_packets: usize) -> usize {
        self.inner.symbols_len(n_packets)
    }

    /// The useful bit rate at a symbol rate, ``2 Rs r 188/204`` (Annexes C and D).
    ///
    /// Args:
    ///     symbol_rate (float): the symbol rate (baud).
    ///
    /// Returns:
    ///     float: the useful bit rate (bit/s).
    fn useful_bit_rate(&self, symbol_rate: f64) -> f64 {
        self.inner.useful_bit_rate(symbol_rate)
    }

    /// The Es/N0 (dB) of an Eb/N0 per useful bit, the abscissa of Table 3.
    ///
    /// Args:
    ///     ebn0_db (float): Eb/N0 per useful bit (dB).
    ///
    /// Returns:
    ///     float: Es/N0 (dB), ``ebn0_db + 10 log10(2 r 188/204)``.
    fn esn0_db(&self, ebn0_db: f64) -> f64 {
        self.inner.esn0_db(ebn0_db)
    }

    /// The bytes entering the inner code: energy dispersal, RS(204, 188), interleaver.
    ///
    /// Args:
    ///     packets (array_like): whole transport packets (uint8, 188 bytes each).
    ///
    /// Returns:
    ///     numpy.ndarray: the bytes (uint8), the flush of the interleaver included.
    ///
    /// Raises:
    ///     ValueError: for a packet that does not start with 0x47 or a partial packet.
    fn outer<'py>(&self, py: Python<'py>, packets: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        let p = bytevec(packets)?;
        Ok(py.detach(|| dvb::burst_encode(&p)).py()?.into_pyarray(py))
    }

    /// The QPSK symbols of the bytes of :meth:`outer` (punctured inner code, Figure 5).
    ///
    /// Args:
    ///     data (array_like): the bytes of :meth:`outer` (uint8).
    ///
    /// Returns:
    ///     numpy.ndarray: the unit-energy symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for a value outside 0 to 255.
    fn inner<'py>(&self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let b = bytevec(data)?;
        Ok(carr(py, py.detach(|| self.inner.inner(&b)).py()?))
    }

    /// The symbols of a burst of whole transport packets (:meth:`inner` of :meth:`outer`).
    ///
    /// Args:
    ///     packets (array_like): whole transport packets (uint8, 188 bytes each).
    ///
    /// Returns:
    ///     numpy.ndarray: ``symbols_len(n)`` unit-energy symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for a packet that does not start with 0x47 or a partial packet.
    fn modulate<'py>(&self, py: Python<'py>, packets: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let p = bytevec(packets)?;
        Ok(carr(py, py.detach(|| self.inner.modulate(&p)).py()?))
    }

    /// The bytes decided by the soft Viterbi decoder, the estimate of :meth:`outer`.
    ///
    /// Args:
    ///     rx (array_like): the received symbols of a burst.
    ///     noise_var (float): complex noise variance per symbol.
    ///
    /// Returns:
    ///     numpy.ndarray: the bytes (uint8).
    ///
    /// Raises:
    ///     ValueError: if ``rx`` is not the length of a burst.
    fn inner_decode<'py>(&self, py: Python<'py>, rx: &Bound<'py, PyAny>, noise_var: f64) -> PyResult<BArr<'py>> {
        let y = cvec(rx)?;
        Ok(py.detach(|| self.inner.inner_decode(&y, noise_var)).py()?.into_pyarray(py))
    }

    /// The packets of the received symbols of a burst.
    ///
    /// Args:
    ///     rx (array_like): the received symbols of a burst, aligned on its first symbol.
    ///     noise_var (float): complex noise variance per symbol.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int, int]: the packets (uint8), the bytes corrected by
    ///     RS(204, 188) and the packets it could not correct.
    ///
    /// Raises:
    ///     ValueError: if ``rx`` is not the length of a burst.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        rx: &Bound<'py, PyAny>,
        noise_var: f64,
    ) -> PyResult<(BArr<'py>, usize, usize)> {
        let y = cvec(rx)?;
        let d = py.detach(|| self.inner.decode(&y, noise_var)).py()?;
        Ok((d.packets.into_pyarray(py), d.corrected, d.failed))
    }
}

/// A DVB-T transmitter and receiver (ETSI EN 300 744 V1.6.2) of one non-hierarchical
/// configuration.
///
/// The transmitter takes a burst of whole MPEG-2 transport packets (see
/// :func:`dvb_packets`): the energy dispersal, RS(204, 188) and the outer interleaver of
/// DVB-S (flushed by 11 packets of zeros), the inner code (171, 133, octal) punctured by
/// Table 2, the inner interleaver (six bit interleavers of 126 bits and the symbol
/// interleaver of the mode), the constellation of Figure 9 normalised by Table 6, the OFDM
/// frame with its scattered and continual pilots (boosted to 4/3) and its TPS (signalling
/// the mode, the guard interval, the constellation and the rate, BCH-protected,
/// differentially modulated), and the OFDM modulation with the guard interval, one sample
/// per elementary period T (7/64 µs at 8 MHz). The burst starts with the first symbol of a
/// super-frame. The receiver works on the cells of the symbols, given the noise variance of
/// a carrier and, if any, the gain of every carrier (a static channel known to the
/// receiver, as in Annex A).
///
/// The cells are arrays of shape ``(K, n_symbols)``, K = 1705 (2K) or 6817 (8K) carriers.
///
/// Args:
///     mode (str): ``"2K"`` or ``"8K"``.
///     guard (str): the guard interval, ``"1/4"``, ``"1/8"``, ``"1/16"`` or ``"1/32"``.
///     constellation (str): ``"QPSK"``, ``"16-QAM"`` or ``"64-QAM"``.
///     rate (str): the code rate, ``"1/2"``, ``"2/3"``, ``"3/4"``, ``"5/6"`` or ``"7/8"``.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comnum, numpy as np
///     >>> link = comnum.Dvbt("2K", "1/4", "16-QAM", "2/3")
///     >>> link.packets_per_superframe          # Table 16
///     672
///     >>> round(link.useful_bit_rate(8e6) / 1e6, 2)   # Table 17
///     13.27
///     >>> packets = comnum.dvb_packets(30, comnum.Rng(1))
///     >>> cells = link.demodulate(link.modulate(packets))
///     >>> cells.shape == (1705, link.symbols_len(30))
///     True
///     >>> bool(np.array_equal(link.decode(cells, 30, 1e-3)[0], packets))
///     True
#[pyclass(name = "Dvbt", module = "comnum")]
pub struct PyDvbt {
    inner: Dvbt,
}

fn channel_in(channel: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<C64>>> {
    channel.map(cvec).transpose()
}

#[pymethods]
impl PyDvbt {
    fn __repr__(&self) -> String {
        format!(
            "Dvbt(mode={:?}, guard={:?}, constellation={:?}, rate={:?})",
            self.inner.mode().name(),
            self.inner.guard().name(),
            self.inner.modulation().name(),
            self.inner.rate().name()
        )
    }

    #[new]
    #[pyo3(signature = (mode = "8K", guard = "1/4", constellation = "QPSK", rate = "1/2"))]
    fn new(mode: &str, guard: &str, constellation: &str, rate: &str) -> PyResult<Self> {
        let mode: Mode = mode.parse().py()?;
        let guard: Guard = guard.parse().py()?;
        let modulation: Modulation = constellation.parse().py()?;
        let rate: InnerRate = rate.parse().py()?;
        Ok(PyDvbt { inner: Dvbt::new(mode, guard, modulation, rate).py()? })
    }

    /// int: The carriers of a symbol, K.
    #[getter]
    fn carriers(&self) -> usize {
        self.inner.mode().carriers()
    }

    /// int: The data cells of a symbol (1512 in the 2K mode, 6048 in the 8K mode).
    #[getter]
    fn data_cells(&self) -> usize {
        self.inner.mode().data_cells()
    }

    /// int: The RS packets of a super-frame of 4 frames of 68 symbols (Table 16).
    #[getter]
    fn packets_per_superframe(&self) -> usize {
        self.inner.packets_per_superframe()
    }

    /// Constellation: The points of the data cells (label ``y0 y1 ... y(v-1)``, Figure 9).
    #[getter]
    fn constellation(&self) -> PyConstellation {
        PyConstellation { inner: self.inner.constellation().clone() }
    }

    /// The OFDM symbols of a burst of ``n_packets`` packets.
    ///
    /// Args:
    ///     n_packets (int): packets of the burst.
    ///
    /// Returns:
    ///     int: the symbols of :meth:`modulate`.
    fn symbols_len(&self, n_packets: usize) -> usize {
        self.inner.symbols_len(n_packets)
    }

    /// The useful bit rate in a channel of a bandwidth (Table 17).
    ///
    /// Args:
    ///     bandwidth_hz (float): 8e6, 7e6, 6e6 or 5e6 (any value: T = 7 / (8 B)).
    ///
    /// Returns:
    ///     float: the useful bit rate (bit/s).
    #[pyo3(signature = (bandwidth_hz = 8e6))]
    fn useful_bit_rate(&self, bandwidth_hz: f64) -> f64 {
        self.inner.useful_bit_rate(bandwidth_hz)
    }

    /// The noise variance of a carrier at a C/N calculated over the K carriers (Annex A):
    /// the mean power of the carriers, pilots boosted, over the C/N.
    ///
    /// Args:
    ///     cn_db (float): C/N (dB).
    ///
    /// Returns:
    ///     float: the complex noise variance of a carrier.
    fn carrier_noise_var(&self, cn_db: f64) -> f64 {
        self.inner.carrier_noise_var(cn_db)
    }

    /// The noise variance of a sample of :meth:`modulate` that gives a noise variance on
    /// every carrier.
    ///
    /// Args:
    ///     carrier_noise_var (float): the noise variance of a carrier.
    ///
    /// Returns:
    ///     float: the complex noise variance of a sample, ``carrier_noise_var N / K``.
    fn sample_noise_var(&self, carrier_noise_var: f64) -> f64 {
        self.inner.sample_noise_var(carrier_noise_var)
    }

    /// The baseband frequency of every carrier, ``k' / TU`` with ``k' = k - (K - 1)/2``.
    ///
    /// Args:
    ///     bandwidth_hz (float): the channel bandwidth (Hz).
    ///
    /// Returns:
    ///     numpy.ndarray: K frequencies (Hz).
    #[pyo3(signature = (bandwidth_hz = 8e6))]
    fn carrier_frequencies<'py>(&self, py: Python<'py>, bandwidth_hz: f64) -> FArr<'py> {
        farr(py, self.inner.carrier_frequencies(bandwidth_hz))
    }

    /// The cells of the symbols of the bytes of a burst (see :meth:`Dvbs.outer`).
    ///
    /// Args:
    ///     data (array_like): the bytes of the outer coding of a burst (uint8).
    ///
    /// Returns:
    ///     numpy.ndarray: the cells, shape ``(K, n_symbols)`` (complex128).
    ///
    /// Raises:
    ///     ValueError: for a value outside 0 to 255.
    fn cells<'py>(&self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<CMat<'py>> {
        let b = bytevec(data)?;
        let c = py.detach(|| self.inner.cells(&b)).py()?;
        grid_out(py, &c.concat(), self.inner.mode().carriers())
    }

    /// The samples of a burst of whole transport packets (one sample per T, the guard
    /// intervals included).
    ///
    /// Args:
    ///     packets (array_like): whole transport packets (uint8, 188 bytes each).
    ///
    /// Returns:
    ///     numpy.ndarray: the samples (complex128).
    ///
    /// Raises:
    ///     ValueError: for a packet that does not start with 0x47 or a partial packet.
    fn modulate<'py>(&self, py: Python<'py>, packets: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let p = bytevec(packets)?;
        Ok(carr(py, py.detach(|| self.inner.modulate(&p)).py()?))
    }

    /// The cells of the received samples of whole symbols (guard intervals removed).
    ///
    /// Args:
    ///     samples (array_like): the received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the cells, shape ``(K, n_symbols)`` (complex128).
    ///
    /// Raises:
    ///     ValueError: unless the samples hold whole symbols.
    fn demodulate<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<CMat<'py>> {
        let x = cvec(samples)?;
        let c = py.detach(|| self.inner.demodulate(&x)).py()?;
        grid_out(py, &c.concat(), self.inner.mode().carriers())
    }

    /// The bytes decided by the soft Viterbi decoder from the cells of a burst.
    ///
    /// Args:
    ///     cells (array_like): the received cells, shape ``(K, n_symbols)``.
    ///     n_packets (int): the packets of the burst.
    ///     noise_var (float): complex noise variance of a carrier.
    ///     channel (array_like, optional): the gain of every carrier (K values).
    ///
    /// Returns:
    ///     numpy.ndarray: the bytes (uint8).
    ///
    /// Raises:
    ///     ValueError: if the cells are not those of a burst of ``n_packets`` packets.
    #[pyo3(signature = (cells, n_packets, noise_var, channel = None))]
    fn inner_decode<'py>(
        &self,
        py: Python<'py>,
        cells: &Bound<'py, PyAny>,
        n_packets: usize,
        noise_var: f64,
        channel: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<BArr<'py>> {
        let symbols = self.symbols_in(cells)?;
        let h = channel_in(channel)?;
        let b = py.detach(|| self.inner.inner_decode(&symbols, n_packets, noise_var, h.as_deref())).py()?;
        Ok(b.into_pyarray(py))
    }

    /// The packets of the cells of a burst.
    ///
    /// Args:
    ///     cells (array_like): the received cells, shape ``(K, n_symbols)``.
    ///     n_packets (int): the packets of the burst.
    ///     noise_var (float): complex noise variance of a carrier.
    ///     channel (array_like, optional): the gain of every carrier (K values).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int, int]: the packets (uint8), the bytes corrected by
    ///     RS(204, 188) and the packets it could not correct.
    ///
    /// Raises:
    ///     ValueError: if the cells are not those of a burst of ``n_packets`` packets.
    #[pyo3(signature = (cells, n_packets, noise_var, channel = None))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        cells: &Bound<'py, PyAny>,
        n_packets: usize,
        noise_var: f64,
        channel: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(BArr<'py>, usize, usize)> {
        let symbols = self.symbols_in(cells)?;
        let h = channel_in(channel)?;
        let d = py.detach(|| self.inner.decode_cells(&symbols, n_packets, noise_var, h.as_deref())).py()?;
        Ok((d.packets.into_pyarray(py), d.corrected, d.failed))
    }

    /// The TPS of a received frame (its 68 symbols, the first one first).
    ///
    /// Args:
    ///     cells (array_like): the cells of the 68 symbols, shape ``(K, 68)``.
    ///     channel (array_like, optional): the gain of every carrier (K values).
    ///
    /// Returns:
    ///     dict: ``frame``, ``mode``, ``guard``, ``constellation``, ``hierarchy`` (alpha,
    ///     0 for non-hierarchical), ``rate_hp``, ``rate_lp`` (``None`` when
    ///     non-hierarchical) and ``cell_id`` (``None`` if not transmitted).
    ///
    /// Raises:
    ///     ValueError: for another number of symbols, a TPS that does not decode, or a
    ///     reserved value.
    #[pyo3(signature = (cells, channel = None))]
    fn decode_tps<'py>(
        &self,
        py: Python<'py>,
        cells: &Bound<'py, PyAny>,
        channel: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let symbols = self.symbols_in(cells)?;
        let h = channel_in(channel)?;
        let t: Tps = self.inner.decode_tps(&symbols, h.as_deref()).py()?;
        let d = PyDict::new(py);
        d.set_item("frame", t.frame)?;
        d.set_item("mode", t.mode.name())?;
        d.set_item("guard", t.guard.name())?;
        d.set_item("constellation", t.modulation.name())?;
        d.set_item(
            "hierarchy",
            match t.hierarchy {
                comnum::systems::dvbt::Hierarchy::None => 0,
                h => h.alpha(),
            },
        )?;
        d.set_item("rate_hp", t.rate_hp.name())?;
        d.set_item("rate_lp", t.rate_lp.map(InnerRate::name))?;
        d.set_item("cell_id", t.cell_id)?;
        Ok(d)
    }
}

impl PyDvbt {
    /// The symbols of a ``(K, n_symbols)`` array of cells.
    fn symbols_in(&self, cells: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<C64>>> {
        let (flat, k) = grid_in(cells)?;
        if k != self.inner.mode().carriers() {
            return Err(PyValueError::new_err(format!(
                "the cells have {k} carriers, the {} mode {}",
                self.inner.mode().name(),
                self.inner.mode().carriers()
            )));
        }
        Ok(flat.chunks(k).map(<[C64]>::to_vec).collect())
    }
}

/// The gains of the static channels of Annex B of EN 300 744 (DVB-T) at baseband
/// frequencies: the fixed reception F1 (20 echoes and a direct path, Ricean factor 10 dB)
/// and the portable reception P1 (the 20 echoes of Table B.1), normalised to a unit power
/// gain.
///
/// Args:
///     profile (str): ``"F1"`` or ``"P1"``.
///     frequencies (array_like): the frequencies (Hz), e.g. :meth:`Dvbt.carrier_frequencies`.
///
/// Returns:
///     numpy.ndarray: the gains H(f) (complex128).
///
/// Raises:
///     ValueError: for another profile.
#[pyfunction]
fn dvbt_channel_gains<'py>(py: Python<'py>, profile: &str, frequencies: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    let e = match profile.trim().to_ascii_uppercase().as_str() {
        "F1" => Echoes::dvbt_f1(),
        "P1" => Echoes::dvbt_p1(),
        _ => {
            return Err(PyValueError::new_err(format!("the profile must be F1 or P1, got {profile:?}")));
        }
    };
    Ok(carr(py, e.frequency_response(&fvec(frequencies)?)))
}

/// MPEG-2 transport packets for a simulation: the sync byte ``0x47`` and 187 random bytes
/// each.
///
/// Args:
///     n (int): number of packets.
///     rng (Rng): the generator.
///
/// Returns:
///     numpy.ndarray: ``188 n`` bytes (uint8).
///
/// Raises:
///     TypeError: if ``rng`` is not an :class:`Rng`.
#[pyfunction]
fn dvb_packets<'py>(py: Python<'py>, n: usize, mut rng: PyRefMut<'py, PyRng>) -> PyResult<BArr<'py>> {
    Ok(dvb::random_packets(n, &mut rng.inner).into_pyarray(py))
}

/// The BBFRAME of ``k_bch`` bits of a generic continuous stream: a BBHEADER (CCM, single
/// stream, DFL = ``k_bch - 80``, CRC-8) and the ``k_bch - 80`` data bits, randomised by the
/// PRBS of the standard.
///
/// Args:
///     data (array_like): the ``k_bch - 80`` data bits.
///     k_bch (int): the bits of the BBFRAME (``Dvbs2.k``).
///     rolloff (int): the roll-off field of the BBHEADER (0: 0.35, 1: 0.25, 2: 0.20).
///
/// Returns:
///     numpy.ndarray: the ``k_bch`` bits of the BBFRAME (uint8).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (data, k_bch, *, rolloff = 0))]
fn dvbs2_bbframe<'py>(py: Python<'py>, data: &Bound<'py, PyAny>, k_bch: usize, rolloff: u8) -> PyResult<BArr<'py>> {
    Ok(dvbs2::bbframe(&bvec(data)?, k_bch, rolloff).py()?.into_pyarray(py))
}

/// The data bits of a received BBFRAME and whether the CRC-8 of its BBHEADER holds.
///
/// Args:
///     bbframe (array_like): the bits of a BBFRAME.
///
/// Returns:
///     tuple[numpy.ndarray, bool]: the data bits (uint8, derandomised) and the CRC check.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn dvbs2_bbframe_data<'py>(py: Python<'py>, bbframe: &Bound<'py, PyAny>) -> PyResult<(BArr<'py>, bool)> {
    let (d, ok) = dvbs2::bbframe_data(&bvec(bbframe)?).py()?;
    Ok((d.into_pyarray(py), ok))
}

/// The MODCOD field, the short-frame flag and the pilot flag of a received PLHEADER (90
/// symbols), by maximum likelihood over the 128 PLS codewords.
///
/// Args:
///     rx (array_like): the 90 received symbols of the PLHEADER.
///
/// Returns:
///     tuple[int, bool, bool]: the MODCOD, the short-frame flag and the pilot flag.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn dvbs2_decode_plheader(rx: &Bound<'_, PyAny>) -> PyResult<(u8, bool, bool)> {
    let code = dvbs2::decode_plheader(&cvec(rx)?).py()?;
    Ok((code.modcod, code.short, code.pilots))
}

/// The convolutional code of CCSDS 131.0-B-5, Sec. 3: constraint length 7, ``G1 = 171``
/// then ``G2 = 133`` (octal), ``G2`` inverted at rate 1/2, punctured by Table 3-1 at
/// rates 2/3, 3/4, 5/6 and 7/8 (``C1`` and ``C2`` sent in the order of the table); the
/// encoder starts in state 0 and is not terminated, the decoder is the soft Viterbi
/// decoder of the whole stream.
///
/// Args:
///     rate (str): ``"1/2"``, ``"2/3"``, ``"3/4"``, ``"5/6"`` or ``"7/8"``.
///
/// Raises:
///     ValueError: for another rate.
///
/// Examples:
///     The first symbols of a one followed by zeros at rate 1/2: ``C1`` follows
///     ``G1 = 1111001``, ``C2`` the inverse of ``G2 = 1011011``:
///
///     >>> import comnum, numpy as np
///     >>> comnum.CcsdsConvolutional("1/2").encode([1, 0, 0])
///     array([1, 0, 1, 1, 1, 0], dtype=uint8)
///     >>> code = comnum.CcsdsConvolutional("3/4")
///     >>> b = comnum.bits(300, comnum.Rng(3))
///     >>> c = code.encode(b)
///     >>> len(c)
///     400
///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * c, 300), b))
///     True
#[pyclass(name = "CcsdsConvolutional", module = "comnum")]
pub struct PyCcsdsConvolutional {
    inner: ccsds::Convolutional,
}

#[pymethods]
impl PyCcsdsConvolutional {
    fn __repr__(&self) -> String {
        format!("CcsdsConvolutional(rate={:?})", self.inner.rate().name())
    }

    #[new]
    #[pyo3(signature = (rate = "1/2"))]
    fn new(rate: &str) -> PyResult<Self> {
        let rate: ccsds::ConvolutionalRate = rate.parse().py()?;
        Ok(PyCcsdsConvolutional { inner: ccsds::Convolutional::new(rate).py()? })
    }

    /// str: The rate.
    #[getter]
    fn rate(&self) -> &'static str {
        self.inner.rate().name()
    }

    /// The channel symbols of a stream of bits.
    ///
    /// Args:
    ///     bits (array_like): the input bits (a multiple of the puncturing period).
    ///
    /// Returns:
    ///     numpy.ndarray: the channel symbols (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of bits.
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.encode(&bvec(bits)?).py()?.into_pyarray(py))
    }

    /// The input bits of the channel LLRs of :meth:`encode`.
    ///
    /// Args:
    ///     llr (array_like): the LLRs of the channel symbols (positive for a 0).
    ///     n_bits (int): the number of input bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n_bits`` bits (uint8).
    ///
    /// Raises:
    ///     ValueError: if the number of LLRs is not that of ``n_bits`` bits.
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>, n_bits: usize) -> PyResult<BArr<'py>> {
        let l = fvec(llr)?;
        Ok(py.detach(|| self.inner.decode(&l, n_bits)).py()?.into_pyarray(py))
    }
}

/// A pseudo-randomizer of CCSDS 131.0-B-5, Sec. 10.4, from its name.
fn randomizer(name: &str) -> PyResult<Randomizer> {
    match name.trim().to_ascii_lowercase().as_str() {
        "long" | "131071" => Ok(Randomizer::Long),
        "short" | "255" => Ok(Randomizer::Short),
        _ => Err(PyValueError::new_err(format!("the randomizer is \"long\" or \"short\", got {name:?}"))),
    }
}

/// The first bits of a pseudo-randomizer of CCSDS 131.0-B-5, Sec. 10.4: ``"long"``,
/// ``h(x) = x^17 + x^14 + 1`` started at ``00011100011100011`` (period 131 071, Sec.
/// 10.4.2), or ``"short"``, ``h(x) = x^8 + x^7 + x^5 + x^3 + 1`` started at all ones
/// (period 255, Sec. 10.4.3).
///
/// Args:
///     sequence (str): ``"long"`` or ``"short"``.
///     n (int): the number of bits.
///
/// Returns:
///     numpy.ndarray: ``n`` bits (uint8).
///
/// Raises:
///     ValueError: for another sequence.
///
/// Examples:
///     The first 40 bits of the short sequence, ``FF 48 0E C0 9A`` (Sec. 10.4.3, NOTE 2):
///
///     >>> import comnum, numpy as np
///     >>> s = comnum.ccsds_randomizer("short", 40)
///     >>> np.packbits(s).tobytes().hex()
///     'ff480ec09a'
#[pyfunction]
fn ccsds_randomizer<'py>(py: Python<'py>, sequence: &str, n: usize) -> PyResult<BArr<'py>> {
    Ok(randomizer(sequence)?.sequence(n).into_pyarray(py))
}

/// A synchronization marker of CCSDS 131.0-B-5, Sec. 9, as octets.
///
/// Args:
///     name (str): ``"asm"`` (``1ACFFC1D``, Sec. 9.3.2: uncoded, convolutional,
///         Reed–Solomon and LDPC codes), ``"turbo_1_2"`` (``034776C7272895B0``, also the
///         marker of the LDPC codes of rate 1/2 to 4/5 is ``"asm"``), ``"turbo_1_3"``,
///         ``"turbo_1_4"``, ``"turbo_1_6"`` (Figures 9-3 to 9-6) or ``"embedded"``
///         (``352EF853``, the embedded data stream of Sec. 9.4).
///
/// Returns:
///     numpy.ndarray: the octets (uint8).
///
/// Raises:
///     ValueError: for another name.
///
/// Examples:
///     >>> import comnum
///     >>> comnum.ccsds_marker("asm").tobytes().hex()
///     '1acffc1d'
///     >>> len(comnum.ccsds_marker("turbo_1_6"))
///     24
#[pyfunction]
fn ccsds_marker<'py>(py: Python<'py>, name: &str) -> PyResult<BArr<'py>> {
    let m: &[u8] = match name.trim().to_ascii_lowercase().as_str() {
        "asm" => &ccsds::ASM,
        "embedded" => &ccsds::ASM_EMBEDDED,
        other => match other.strip_prefix("turbo_").map(|r| r.replace('_', "/").parse::<TurboRate>()) {
            Some(Ok(rate)) => ccsds::turbo_asm(rate),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "the marker is \"asm\", \"embedded\" or \"turbo_1_2\" … \"turbo_1_6\", got {name:?}"
                )));
            }
        },
    };
    Ok(m.to_vec().into_pyarray(py))
}

/// A channel access data unit (CCSDS 131.0-B-5, Table 9-1): the bits of a marker, then the
/// bits of a codeblock, randomized if ``randomizer`` is given (the marker never is).
///
/// Args:
///     marker (array_like): the octets of the marker (:func:`ccsds_marker`).
///     block (array_like): the bits of the codeblock, codeword or frame.
///     randomizer (str | None): ``"long"``, ``"short"`` or ``None``.
///
/// Returns:
///     numpy.ndarray: the bits (uint8).
///
/// Raises:
///     ValueError: for another randomizer or a marker value above 255.
///
/// Examples:
///     The randomized zeros carry the sequence; the receiver removes it from the LLRs by
///     their sign:
///
///     >>> import comnum, numpy as np
///     >>> cadu = comnum.ccsds_cadu(comnum.ccsds_marker("asm"), np.zeros(40, np.uint8), "short")
///     >>> np.packbits(cadu).tobytes().hex()
///     '1acffc1dff480ec09a'
#[pyfunction]
#[pyo3(signature = (marker, block, randomizer = None))]
fn ccsds_cadu<'py>(
    py: Python<'py>,
    marker: &Bound<'py, PyAny>,
    block: &Bound<'py, PyAny>,
    randomizer: Option<&str>,
) -> PyResult<BArr<'py>> {
    let r = randomizer.map(self::randomizer).transpose()?;
    Ok(ccsds::cadu(&bytevec(marker)?, &bvec(block)?, r).into_pyarray(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCcsdsConvolutional>()?;
    m.add_function(wrap_pyfunction!(ccsds_randomizer, m)?)?;
    m.add_function(wrap_pyfunction!(ccsds_marker, m)?)?;
    m.add_function(wrap_pyfunction!(ccsds_cadu, m)?)?;
    m.add_class::<PyDvbs>()?;
    m.add_class::<PyDvbs2>()?;
    m.add_class::<PyDvbt>()?;
    m.add_function(wrap_pyfunction!(dvbt_channel_gains, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_packets, m)?)?;
    m.add_function(wrap_pyfunction!(dvbs2_bbframe, m)?)?;
    m.add_function(wrap_pyfunction!(dvbs2_bbframe_data, m)?)?;
    m.add_function(wrap_pyfunction!(dvbs2_decode_plheader, m)?)?;
    Ok(())
}
