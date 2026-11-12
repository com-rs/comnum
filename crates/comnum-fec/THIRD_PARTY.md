# Third-party data in the crate comnum-fec

comnum is released under the MIT or Apache-2.0 licence (`LICENSE-MIT`,
`LICENSE-APACHE`). The algorithms are written from their publications; the tables of
standards and papers are cited where they are used, in the code. This file lists the
data of this crate taken from other **software packages**, with their licences. The
`THIRD_PARTY.md` of the repository (<https://github.com/com-rs/comnum>) also lists the
recorded outputs of other tools that the tests of the repository compare with; they
are not part of this crate.

## Tables transcribed from Sionna (Apache License 2.0)

Sionna (NVIDIA, <https://github.com/NVlabs/sionna>, Copyright (c) 2021-2026 NVIDIA
CORPORATION & AFFILIATES) is released under the Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>; its text is the `LICENSE-APACHE` of this
crate). This crate ships, under the same licence:

- the QPP coefficients of `comnum_fec::turbo::lte_qpp` (3GPP TS 36.212, Table
  5.1.3-3), transcribed from Sionna's `turbo_coeffs.csv`;
- the row assignment of the LDPC base graphs of `comnum_fec::nr::tables` (3GPP TS
  38.212, Tables 5.3.2-2 and 5.3.2-3), which follows Sionna's `5G_bg1.csv` and
  `5G_bg2.csv`.

Every entry is checked against the 3GPP specifications, and the polar sequence of TS
38.212 Table 5.3.1.2-1, extracted from the standard, is identical to Sionna's
`polar_5G.csv`. No code of Sionna is copied.
