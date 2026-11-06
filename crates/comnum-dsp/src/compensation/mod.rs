//! Compensation of deterministic channel effects at the receiver.
//!
//! - [`cd`]: chromatic-dispersion compensation, an all-pass frequency-domain filter
//!   (exact on a block, or as an FIR);
//! - [`dbp`]: digital back-propagation, the split-step solver of `comnum_math::nlse` run
//!   with the coefficients of the link negated;
//! - [`dpd`]: digital predistortion of a power amplifier, memory polynomials trained
//!   by indirect learning, and the static predistorter of a memoryless model;
//! - [`iq`]: the quadrature imbalance of a coherent front end, removed by Gram–Schmidt
//!   or Löwdin orthonormalisation of the in-phase and quadrature parts.

pub mod cd;
pub mod dbp;
pub mod dpd;
pub mod iq;
