# Third-party data in the crate comnum-channel

comnum is released under the MIT or Apache-2.0 licence (`LICENSE-MIT`,
`LICENSE-APACHE`). The algorithms are written from their publications; the tables of
standards and papers are cited where they are used, in the code. This file lists the
data of this crate taken from other **software packages**, with their licences. The
`THIRD_PARTY.md` of the repository (<https://github.com/com-rs/comnum>) also lists the
recorded outputs of other tools that the tests of the repository compare with; they
are not part of this crate.

## The Raman gain spectrum of standard single-mode fibre (GNPy, BSD-3-Clause)

`crates/comnum-channel/src/optical/raman/gain.rs` (`RamanGain::ssmf`, the tables
`SSMF_OFFSETS_THZ` and `SSMF_GAIN_M_W`) transcribes, digit for digit, the default Raman
coefficient of GNPy (`gnpy.core.parameters.DEFAULT_RAMAN_COEFFICIENT`): the gain
efficiency $g_R A_{\mathrm{eff}}$ of standard single-mode fibre at 90 frequency offsets,
for a pump at 206.18 THz. GNPy is the optical network planning library of the Telecom
Infra Project, <https://github.com/Telecominfraproject/oopt-gnpy>, released under the
BSD 3-Clause licence reproduced below.

```
BSD 3-Clause License

Copyright (c) 2018, Telecom Infra Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
