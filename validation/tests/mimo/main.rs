//! The validation tests of multiple antennas: detection, capacity, massive MIMO, channel
//! estimation: one module per reference, each loading its cards of `validation/cards/`
//! (`just card <id>` runs one).

mod biguesh2006;
mod ding2014;
mod larsson2014;
mod loyka2001;
mod lu2014;
mod message_passing;
mod mimo;
mod ngo2013;
