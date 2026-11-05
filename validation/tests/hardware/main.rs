//! The validation tests of hardware impairments: converters, amplifiers, predistortion,
//! oscillators: one module per reference, each loading its cards of `validation/cards/`
//! (`just card <id>` runs one).

mod banelli2000;
mod converters;
mod ding2004;
mod eun1997;
mod ieee1139;
