//! AIS (ITU-R M.1371-5): the packets, a receiver for a capture, the messages and the
//! `!AIVDM` sentences.

use crate::{BArr, CArr, IntoPy, bvec, carr, cvec};
use comnum::systems::ais::{
    self,
    message::{self, Message, Position},
    nmea,
};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn position_dict<'py>(py: Python<'py>, p: &Position) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("lon_deg", p.lon_deg)?;
    d.set_item("lat_deg", p.lat_deg)?;
    d.set_item("accurate", p.accurate)?;
    d.set_item("sog_knots", p.sog_knots)?;
    d.set_item("cog_deg", p.cog_deg)?;
    d.set_item("heading_deg", p.heading_deg)?;
    d.set_item("timestamp", p.timestamp)?;
    Ok(d)
}

fn dimensions_dict<'py>(py: Python<'py>, dim: &message::Dimensions) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("to_bow", dim.to_bow)?;
    d.set_item("to_stern", dim.to_stern)?;
    d.set_item("to_port", dim.to_port)?;
    d.set_item("to_starboard", dim.to_starboard)?;
    Ok(d)
}

/// The fields of a message as a dictionary: `msg_type`, `repeat`, `mmsi` always, then
/// the fields of its kind.
fn message_dict<'py>(py: Python<'py>, m: &Message) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("msg_type", m.msg_type())?;
    d.set_item("mmsi", m.mmsi())?;
    match m {
        Message::PositionReport { repeat, nav_status, rot_raw, position, manoeuvre, raim, .. } => {
            d.set_item("repeat", repeat)?;
            d.set_item("nav_status", nav_status)?;
            d.set_item("rot_raw", rot_raw)?;
            d.set_item("position", position_dict(py, position)?)?;
            d.set_item("manoeuvre", manoeuvre)?;
            d.set_item("raim", raim)?;
        }
        Message::BaseStation { repeat, utc, position, epfd, raim, .. } => {
            d.set_item("repeat", repeat)?;
            d.set_item("utc", (utc.0, utc.1, utc.2, utc.3, utc.4, utc.5))?;
            d.set_item("position", position_dict(py, position)?)?;
            d.set_item("epfd", epfd)?;
            d.set_item("raim", raim)?;
        }
        Message::StaticAndVoyage {
            repeat,
            ais_version,
            imo,
            callsign,
            shipname,
            ship_type,
            dimensions,
            epfd,
            eta,
            draught_m,
            destination,
            dte,
            ..
        } => {
            d.set_item("repeat", repeat)?;
            d.set_item("ais_version", ais_version)?;
            d.set_item("imo", imo)?;
            d.set_item("callsign", callsign)?;
            d.set_item("shipname", shipname)?;
            d.set_item("ship_type", ship_type)?;
            d.set_item("dimensions", dimensions_dict(py, dimensions)?)?;
            d.set_item("epfd", epfd)?;
            d.set_item("eta", (eta.0, eta.1, eta.2, eta.3))?;
            d.set_item("draught_m", draught_m)?;
            d.set_item("destination", destination)?;
            d.set_item("dte", dte)?;
        }
        Message::ClassBPosition { repeat, position, cs_unit, raim, .. } => {
            d.set_item("repeat", repeat)?;
            d.set_item("position", position_dict(py, position)?)?;
            d.set_item("cs_unit", cs_unit)?;
            d.set_item("raim", raim)?;
        }
        Message::ClassBName { repeat, shipname, .. } => {
            d.set_item("repeat", repeat)?;
            d.set_item("part", 0u8)?;
            d.set_item("shipname", shipname)?;
        }
        Message::ClassBStatic { repeat, ship_type, vendor, model, serial, callsign, dimensions, .. } => {
            d.set_item("repeat", repeat)?;
            d.set_item("part", 1u8)?;
            d.set_item("ship_type", ship_type)?;
            d.set_item("vendor", vendor)?;
            d.set_item("model", model)?;
            d.set_item("serial", serial)?;
            d.set_item("callsign", callsign)?;
            d.set_item("dimensions", dimensions_dict(py, dimensions)?)?;
        }
        Message::Other { repeat, .. } => {
            d.set_item("repeat", repeat)?;
        }
    }
    Ok(d)
}

/// Decode the bits of an AIS message (ITU-R M.1371-5, Annex 8): the position reports
/// (messages 1, 2, 3, 18), the base station report (4), the static and voyage data (5)
/// and the class B static data (24); any other message gives its type and MMSI.
///
/// Args:
///     bits (array_like): the message bits, most significant first (the payload of a
///         sentence, :func:`ais_messages`).
///
/// Returns:
///     dict: ``msg_type``, ``mmsi``, ``repeat`` and the fields of the message: a
///     ``position`` (``lon_deg``, ``lat_deg``, ``accurate``, ``sog_knots``, ``cog_deg``,
///     ``heading_deg``, ``timestamp``; ``None`` for a value "not available") and
///     ``nav_status``, ``rot_raw``, ``manoeuvre``, ``raim`` (1, 2, 3); ``utc``,
///     ``epfd`` (4); ``imo``, ``callsign``, ``shipname``, ``ship_type``, ``dimensions``,
///     ``eta``, ``draught_m``, ``destination``, ``dte`` (5); ``cs_unit`` (18); ``part``
///     and ``shipname`` or ``ship_type``, ``vendor``, ``model``, ``serial``, ``callsign``,
///     ``dimensions`` (24).
///
/// Raises:
///     ValueError: for fewer than 38 bits, a type outside 1 to 27 or a message shorter
///         than its table.
///
/// Examples:
///     >>> import comnum
///     >>> bits = comnum.ais_position_report(200000001, lat_deg=46.12, lon_deg=-9.88, sog_knots=8.0)
///     >>> m = comnum.ais_decode(bits)
///     >>> m["msg_type"], m["mmsi"], round(m["position"]["lat_deg"], 2)
///     (1, 200000001, 46.12)
#[pyfunction]
fn ais_decode<'py>(py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    message_dict(py, &message::decode(&bvec(bits)?).py()?)
}

/// The bits of a position report (message 1) with the given fields, the other fields
/// zero: the inverse of :func:`ais_decode` for that message.
///
/// Args:
///     mmsi (int): the MMSI.
///     lat_deg (float | None): latitude (degrees), ``None`` for "not available".
///     lon_deg (float | None): longitude (degrees).
///     sog_knots (float | None): speed over ground (knots).
///     cog_deg (float | None): course over ground (degrees).
///     heading_deg (int | None): true heading (degrees).
///     timestamp (int): the UTC second (60: not available).
///     nav_status (int): the navigational status (15: not defined).
///     accurate (bool): the position accuracy flag.
///
/// Returns:
///     numpy.ndarray: the 168 bits (uint8).
#[pyfunction]
#[pyo3(signature = (mmsi, *, lat_deg = None, lon_deg = None, sog_knots = None, cog_deg = None, heading_deg = None, timestamp = 60, nav_status = 15, accurate = false))]
#[allow(clippy::too_many_arguments, reason = "the fields of the report, by keyword")]
fn ais_position_report<'py>(
    py: Python<'py>,
    mmsi: u32,
    lat_deg: Option<f64>,
    lon_deg: Option<f64>,
    sog_knots: Option<f64>,
    cog_deg: Option<f64>,
    heading_deg: Option<u16>,
    timestamp: u8,
    nav_status: u8,
    accurate: bool,
) -> BArr<'py> {
    let p = Position { lon_deg, lat_deg, accurate, sog_knots, cog_deg, heading_deg, timestamp };
    message::position_report_bits(mmsi, nav_status, &p).into_pyarray(py)
}

/// The ``!AIVDM`` sentences of NMEA 0183 (IEC 61162-1) of a message: the six-bit
/// armouring of its bits, the fill bits and the checksum; one or several fragments.
///
/// Args:
///     bits (array_like): the message bits.
///     channel (str): the channel, ``'A'`` or ``'B'``.
///     sequence (int): the sequential identifier of a multi-fragment message (0 to 9).
///
/// Returns:
///     list[str]: the sentences.
///
/// Raises:
///     ValueError: for bits that are not 0 and 1.
#[pyfunction]
#[pyo3(signature = (bits, *, channel = "A", sequence = 0))]
fn ais_aivdm(bits: &Bound<'_, PyAny>, channel: &str, sequence: u8) -> PyResult<Vec<String>> {
    Ok(nmea::aivdm(&bvec(bits)?, channel.chars().next().unwrap_or('A'), sequence))
}

/// The messages of ``!AIVDM`` sentences: the checksums checked, the fragments of a
/// message assembled, the payload de-armoured.
///
/// Args:
///     sentences (list[str]): the sentences, in order.
///
/// Returns:
///     list[dict]: for every complete message, ``bits`` (uint8) and ``channel``.
///
/// Raises:
///     ValueError: for a sentence that is not ``!AIVDM``/``!AIVDO``, a wrong checksum
///         or a character outside the six-bit alphabet.
#[pyfunction]
fn ais_messages<'py>(py: Python<'py>, sentences: Vec<String>) -> PyResult<Bound<'py, PyList>> {
    let refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
    let out = PyList::empty(py);
    for (bits, channel) in nmea::messages(&refs).py()? {
        let d = PyDict::new(py);
        d.set_item("bits", bits.into_pyarray(py))?;
        d.set_item("channel", channel.to_string())?;
        out.append(d)?;
    }
    Ok(out)
}

/// The GMSK signal of the packets of messages (ITU-R M.1371-5, Annex 2: the training
/// sequence, the HDLC frame with its frame check sequence and bit stuffing, NRZI, GMSK
/// with ``BT = 0.4`` at 9600 bit/s), a silence of 32 bits before each, unit envelope.
///
/// Args:
///     messages (list[array_like]): the bits of every message.
///     fs (float): the sampling rate, an even multiple of 9600 Hz (``ais_front_end_rate()``
///         is 153 600 Hz, 16 samples per bit).
///
/// Returns:
///     numpy.ndarray: the samples (complex128).
///
/// Raises:
///     ValueError: for a sampling rate that is not an even multiple of 9600 Hz of 4
///         samples per bit at least.
#[pyfunction]
fn ais_transmit<'py>(py: Python<'py>, messages: Vec<Bound<'py, PyAny>>, fs: f64) -> PyResult<CArr<'py>> {
    let m: Vec<Vec<u8>> = messages.iter().map(bvec).collect::<PyResult<_>>()?;
    Ok(carr(py, ais::transmit(&m, fs).py()?))
}

/// The packets of an AIS channel at baseband: the polar discriminator, an integrator
/// over half a bit, the frequency offset of the transmitter removed, a bit clock from
/// the zero crossings, NRZI decoding and the HDLC frames whose frame check sequence is
/// correct, decoded by :func:`ais_decode`. The channel (161.975 or 162.025 MHz, 25 kHz
/// wide) is brought to baseband by :func:`select_channel` at ``fs``.
///
/// Args:
///     x (array_like): the channel at baseband.
///     fs (float): its sampling rate, an even multiple of 9600 Hz (16 samples per bit,
///         153 600 Hz, is the rate of the receiver's design).
///
/// Returns:
///     list[dict]: for every packet, ``start_bit`` (the bit of the line where its flag
///     starts; the time is ``start_bit / 9600``), ``bits`` (uint8), ``aivdm`` (the
///     sentences, channel ``A``), and ``message`` (the dictionary of
///     :func:`ais_decode`) or ``error`` (why it could not be decoded).
///
/// Raises:
///     ValueError: for the sampling rate, or fewer than 64 bits of signal.
///
/// Examples:
///     >>> import comnum
///     >>> fs = comnum.ais_front_end_rate()
///     >>> bits = comnum.ais_position_report(200000001, lat_deg=46.12, lon_deg=-9.88)
///     >>> x = comnum.ais_transmit([bits], fs)
///     >>> [p["message"]["mmsi"] for p in comnum.ais_receive(x, fs)]
///     [200000001]
#[pyfunction]
fn ais_receive<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64) -> PyResult<Bound<'py, PyList>> {
    let v = cvec(x)?;
    let packets = py.detach(|| ais::receive(&v, fs)).py()?;
    let out = PyList::empty(py);
    for p in packets {
        let d = PyDict::new(py);
        d.set_item("start_bit", p.start_bit)?;
        d.set_item("aivdm", nmea::aivdm(&p.bits, 'A', 0))?;
        match &p.message {
            Ok(m) => d.set_item("message", message_dict(py, m)?)?,
            Err(e) => d.set_item("error", e)?,
        }
        d.set_item("bits", p.bits.into_pyarray(py))?;
        out.append(d)?;
    }
    Ok(out)
}

/// The sampling rate of the AIS receiver: 16 samples per bit, 153 600 Hz.
///
/// Returns:
///     float: the rate (Hz).
#[pyfunction]
fn ais_front_end_rate() -> f64 {
    ais::FRONT_END_RATE
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ais_decode, m)?)?;
    m.add_function(wrap_pyfunction!(ais_position_report, m)?)?;
    m.add_function(wrap_pyfunction!(ais_aivdm, m)?)?;
    m.add_function(wrap_pyfunction!(ais_messages, m)?)?;
    m.add_function(wrap_pyfunction!(ais_transmit, m)?)?;
    m.add_function(wrap_pyfunction!(ais_receive, m)?)?;
    m.add_function(wrap_pyfunction!(ais_front_end_rate, m)?)?;
    Ok(())
}
