//! The validation tests of multiple antennas and multiple access: detection, capacity,
//! massive MIMO, channel estimation, NOMA, DS-CDMA: one module per reference, each loading its cards of `validation/cards/`
//! (`just card <id>` runs one).

mod alamouti1998;
mod biguesh2006;
mod ding2014;
mod dsss;
mod larsson2014;
mod loyka2001;
mod lu2014;
mod message_passing;
mod mimo;
mod ngo2013;
mod tse1999;
