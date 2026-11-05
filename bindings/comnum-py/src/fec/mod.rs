//! Channel coding: CRC, scrambling, interleaving, convolutional, turbo, LDPC and polar
//! codes, and the finite-length limits (the Reed–Solomon and BCH codes and the
//! puncturing patterns are in [`crate::fec_codes`]).

mod ccsds;
mod conv;
mod crc;
mod ldpc_code;
mod limits;
mod nr;
mod polar_code;

use crate::fec_codes::{pattern_rows, puncturer};
use crate::{BArr, FArr, IArr, IntoPy, bvec, count, extract_error, farr, fvec, iarr, map_real, py_bool};
use comnum::fec::block::BlockCode;
use comnum::fec::bounds;
use comnum::fec::conv::{ConvolutionalCode, Trellis};
use comnum::fec::crc::Crc;
use comnum::fec::interleaving::Interleaver;
use comnum::fec::ldpc::{self, CheckRule, LdpcCode};
use comnum::fec::polar::PolarCode;
use comnum::fec::puncture::{Punctured, PuncturedTurbo, Puncturer};
use comnum::fec::scrambling::Scrambler;
use comnum::fec::turbo::{TurboCode, TurboDecoder};
use comnum::math::Jacobian;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Encode whole blocks of bits with a code, the GIL released.
fn encode<'py, C>(py: Python<'py>, code: &C, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>>
where
    C: BlockCode + Sync + ?Sized,
{
    let b = bvec(bits)?;
    Ok(py.detach(|| code.encode_stream(&b)).py()?.into_pyarray(py))
}

/// Decode whole blocks of LLRs with a code, the GIL released.
fn decode<'py, C>(py: Python<'py>, code: &C, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>>
where
    C: BlockCode + Sync + ?Sized,
{
    let l = fvec(llr)?;
    Ok(py.detach(|| code.decode_stream(&l)).py()?.into_pyarray(py))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crc::register(m)?;
    conv::register(m)?;
    ldpc_code::register(m)?;
    polar_code::register(m)?;
    nr::register(m)?;
    ccsds::register(m)?;
    limits::register(m)?;
    Ok(())
}
