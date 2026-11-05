//! Constant-envelope waveforms: continuous-phase modulation (CPFSK, MSK, GMSK),
//! constant-envelope OFDM and chirp spread spectrum.

mod ce_ofdm;
mod continuous_phase;
mod css;

use crate::comm::PyConstellation;
use crate::kind;
use crate::{BArr, CArr, FArr, IArr, IntoPy, bvec, carr, count, cvec, farr, fvec, iarr, map_real, uvec};
use comnum::math::C64;
use comnum::waveform::ce_ofdm::{CeOfdm, Receiver, Unwrap, gamp::Gamp, theory};
use comnum::waveform::cpm::{self, Cpm, CpmConfig, FrequencyPulse, ModulationIndex};
use comnum::waveform::css::{Css, Waveform, theory as css_theory};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    continuous_phase::register(m)?;
    ce_ofdm::register(m)?;
    css::register(m)?;
    Ok(())
}
