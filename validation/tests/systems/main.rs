//! The validation tests of systems of standards: DVB, CCSDS, 5G NR: one module per
//! reference, each loading its cards of `validation/cards/` (`just card <id>` runs one).

mod ccsds131;
mod dvbs;
mod dvbt;
mod lin2019;
mod ts38101_4;
