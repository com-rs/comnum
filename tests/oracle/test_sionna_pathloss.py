"""The path loss and line-of-sight probability of comnum against those of Sionna.

Sionna (NVIDIA, Apache-2.0) implements TR 38.901 §7.4 inside its system-level scenarios
(``UMaScenario``, ``UMiScenario``, ``RMaScenario``, ``InHScenario``). The recording places
one BS and UTs on a line at fixed horizontal distances (outdoors, but indoors for InH), sets
the LOS state, and reads the basic path loss and the LOS probability; the UTs are 1.5 m
high, so the effective environment height of UMa is 1 m (Table 7.4.1-1 NOTE 1) and no
draw enters. Sionna computes the breakpoints with the exact speed of light, comnum with
the 3.0e8 m/s that NOTE 1 and NOTE 5 prescribe: below the breakpoint the two agree to
1e-9 dB, beyond it the loss depends on the breakpoint distance and they differ by about
0.005 dB (``9 log10`` or ``40 log10`` of a distance 0.07 % apart); no point lies within
0.07 % of a breakpoint.
"""

import numpy as np
import pytest

import comnum
from oracle import oracle

# (scenario, h_BS, h_UT, carriers, distances)
CASES = [
    ("uma", 25.0, 1.5, [3.5e9, 28e9], [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 2000.0, 4000.0]),
    ("umi", 10.0, 1.5, [3.5e9, 28e9], [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 2000.0, 4000.0]),
    ("rma", 35.0, 1.5, [0.7e9, 3.5e9], [15.0, 60.0, 250.0, 1000.0, 3000.0, 4500.0]),
    ("inh", 3.0, 1.0, [3.5e9, 28e9], [2.0, 5.0, 10.0, 30.0, 60.0, 100.0]),
]


def _sionna_pathloss():
    import torch
    from sionna.phy.channel.tr38901 import (
        InHScenario,
        PanelArray,
        RMaScenario,
        UMaScenario,
        UMiScenario,
    )

    classes = {"uma": UMaScenario, "umi": UMiScenario, "rma": RMaScenario, "inh": InHScenario}
    out = {}
    for name, h_bs, h_ut, carriers, distances in CASES:
        for fc in carriers:
            arr = PanelArray(
                num_rows_per_panel=1,
                num_cols_per_panel=1,
                polarization="single",
                polarization_type="V",
                antenna_pattern="omni",
                carrier_frequency=fc,
                precision="double",
            )
            kwargs = {"inh": {"indoor_scenario": "mixed"}, "rma": {}}.get(
                name, {"o2i_model": "low"}
            )
            sc = classes[name](
                carrier_frequency=fc,
                ut_array=arr,
                bs_array=arr,
                direction="downlink",
                enable_shadow_fading=False,
                precision="double",
                **kwargs,
            )
            n = len(distances)
            ut = torch.tensor([[[d, 0.0, h_ut] for d in distances]], dtype=torch.float64)
            bs = torch.tensor([[[0.0, 0.0, h_bs]]], dtype=torch.float64)
            zeros = torch.zeros((1, n, 3), dtype=torch.float64)
            row = {"d2d_m": distances}
            for los in (True, False):
                sc.set_topology(
                    ut,
                    bs,
                    zeros,
                    torch.zeros((1, 1, 3), dtype=torch.float64),
                    zeros,
                    torch.full((1, n), name == "inh", dtype=torch.bool),
                    los,
                )
                row["los" if los else "nlos"] = sc.basic_pathloss[0, 0].tolist()
            row["p_los"] = sc.los_probability[0, 0].tolist()
            out[f"{name}_{fc:.0f}"] = row
    return out


@pytest.fixture(scope="module")
def recorded():
    return oracle("sionna_pathloss", _sionna_pathloss, tool="Sionna")


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_path_loss_and_los_probability_are_those_of_sionna(recorded, case):
    name, h_bs, h_ut, carriers, distances = case
    for fc in carriers:
        ref = recorded[f"{name}_{fc:.0f}"]
        # the breakpoint of NOTE 5 (RMa) and NOTE 1 (UMa, UMi, h_E = 1 m), c = 3e8 m/s
        bp = {
            "rma": 2 * np.pi * h_bs * h_ut,
            "uma": 4 * (h_bs - 1) * (h_ut - 1),
            "umi": 4 * (h_bs - 1) * (h_ut - 1),
        }
        d_bp = bp.get(name, np.inf) * fc / 3e8
        for i, d in enumerate(distances):
            assert abs(d / d_bp - 1) > 7e-4, (name, fc, d)
            tol = 1e-9 if d < d_bp else 0.01
            for los, key in ((True, "los"), (False, "nlos")):
                pl = comnum.path_loss_db(name, d, fc, los=los, h_bs_m=h_bs, h_ut_m=h_ut)
                assert pl == pytest.approx(ref[key][i], abs=tol), (name, fc, d, key)
            p = comnum.los_probability(name, d, h_ut_m=h_ut, h_bs_m=h_bs)
            assert p == pytest.approx(ref["p_los"][i], abs=1e-12), (name, fc, d)
