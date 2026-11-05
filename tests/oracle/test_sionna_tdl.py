"""The TDL profiles of comnum against those of Sionna (NVIDIA, Apache-2.0).

Sionna implements the TDL models of 3GPP TR 38.901 (Tables 7.7.2-1 to 7.7.2-5, in
``models/v19_2``) and the simplified profiles of the conformance tests of 3GPP TS
38.101-4 (Tables B.2.1.1-2 to B.2.1.1-5, in ``models/ts_38_101_4_v19_2_2``) as JSON
tables. The recording reads these files (no simulation): their delays (normalised for TR
38.901, in ns for TS 38.101-4) and powers (dB), the specular (LOS) row first for TDL-D,
TDL-E and TDLD30. comnum derives the TR 38.901 taps from its CDL tables (TR 38.901
§7.7.4) and transcribes the TS 38.101-4 tables: both must give the same rows.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import comnum
from oracle import oracle

TR = {"tdl-a": "TDL-A", "tdl-b": "TDL-B", "tdl-c": "TDL-C", "tdl-d": "TDL-D", "tdl-e": "TDL-E"}
TS = {"tdla30": "TDL-A30", "tdlb100": "TDL-B100", "tdlc300": "TDL-C300", "tdld30": "TDL-D30"}


def _sionna_tables():
    spec = importlib.util.find_spec("sionna")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("sionna is not installed")
    models = Path(next(iter(spec.submodule_search_locations))) / "phy/channel/tr38901/models"
    out = {}
    for folder, names in [("v19_2", TR), ("ts_38_101_4_v19_2_2", TS)]:
        for name, file in names.items():
            d = json.loads((models / folder / f"{file}.json").read_text())
            out[name] = {"los": int(d["los"]), "delays": d["delays"], "powers": d["powers"]}
    return out


@pytest.fixture(scope="module")
def tables():
    return oracle("sionna_tdl", _sionna_tables, tool="Sionna")


@pytest.mark.parametrize("name", list(TR) + list(TS))
def test_tdl_profiles_are_those_of_sionna(tables, name):
    ref = tables[name]
    # TR 38.901 tables are normalised (unit RMS delay spread); TS 38.101-4 tables in ns
    scale = 1.0 if name in TR else 1e-9
    taps = comnum.power_delay_profile(name, delay_spread_s=1.0)
    delays, powers = list(taps[:, 0]), list(taps[:, 1])
    if ref["los"]:
        # the specular row of the first tap: its power is that of the TdlChannel
        ch = comnum.TdlChannel(name, 1e9, delay_spread_s=1.0)
        idx, p_s = ch.specular
        p_r = ch.powers[idx]
        k_db = 10 * np.log10(p_s / p_r)
        assert k_db == pytest.approx(ref["powers"][0] - ref["powers"][1], abs=1e-9)
        assert ref["delays"][0] == ref["delays"][1] == delays[0] / scale
        ref_delays, ref_powers = ref["delays"][1:], ref["powers"][1:]
    else:
        ref_delays, ref_powers = ref["delays"], ref["powers"]
    np.testing.assert_allclose(np.array(delays) / scale, ref_delays, rtol=0, atol=1e-9)
    np.testing.assert_allclose(powers, ref_powers, rtol=0, atol=1e-12)
