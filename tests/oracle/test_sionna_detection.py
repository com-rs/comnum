"""The demapper and the MIMO detectors of comnum against Sionna (NVIDIA, Apache-2.0).

Sionna (<https://github.com/NVlabs/sionna>) implements the bit LLRs of a constellation
(``Demapper``, exact ``"app"`` and ``"maxlog"``) and the MIMO detectors
``LinearDetector`` (LMMSE and ZF equalisation, then the nearest point),
``MaximumLikelihoodDetector`` (exhaustive) and ``KBestDetector`` (a tree search keeping
:math:`K` paths, complex representation). Their outputs are recorded fixtures
(``tests/oracle/sionna_detection.json`` and ``.npz``, see ``tests/oracle/__init__.py``),
with the received samples and channels they were given::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_sionna_detection.py

Sionna is given comnum's points as a custom constellation (``Constellation("custom",
...)``, labels the binary indices of the points, most significant bit first), in
double precision. Its LLR is :math:`\\ln P(b = 1 | y) / P(b = 0 | y)`, the opposite of
:meth:`comnum.Constellation.llr`.

Compared:

- the LLRs of 4- to 256-QAM, 8-PSK and 4-PAM at three noise variances, exact and
  max-log, 120 samples each;
- the decisions of :func:`comnum.mimo_detect` (``"mmse"``, ``"zf"``, ``"ml"``,
  ``"sphere"``, ``"kbest"`` with :math:`K` = 4 and 16) with those of Sionna on 100
  vectors of four configurations, 2 × 4 to 8 × 8, QPSK and 16-QAM, i.i.d. Rayleigh
  channels.

The cards ``sionna_llr`` and ``sionna_mimo_detection`` hold the comparisons.
"""

import numpy as np
import pytest

import comnum
from oracle import oracle

LLR = [("qam", 4), ("qam", 16), ("qam", 64), ("qam", 256), ("psk", 8), ("pam", 4)]
NOISE = [0.01, 0.1, 1.0]
# (n_rx, n_tx, kind, order, noise variance per receive antenna)
MIMO = [
    (4, 2, "qam", 16, 0.1),
    (4, 4, "qam", 4, 0.2),
    (4, 4, "qam", 16, 0.05),
    (8, 8, "qam", 4, 0.3),
]
KBEST = [4, 16]


def mkey(n_rx, n_tx, kind, m, nv):
    return f"{n_rx}x{n_tx}_{kind}{m}_{nv:g}"


def _custom(points):
    import torch
    from sionna.phy.mapping import Constellation

    bits = int(np.log2(len(points)))
    return Constellation(
        "custom", bits, points=torch.tensor(points, dtype=torch.complex128), precision="double"
    )


def _compute():
    import torch
    from sionna.phy.mapping import Demapper
    from sionna.phy.mimo import KBestDetector, LinearDetector, MaximumLikelihoodDetector

    rng = np.random.default_rng(20261117)
    out = {"llr": {}, "mimo": {}}
    for kind, m in LLR:
        pts = np.asarray(comnum.Constellation(kind, m).normalized().points, complex)
        sc = _custom(pts)
        for nv in NOISE:
            x = pts[rng.integers(0, m, 120)]
            w = np.sqrt(nv / 2) * (rng.standard_normal(120) + 1j * rng.standard_normal(120))
            y = x + w if kind != "pam" else x + np.sqrt(2) * w.real  # a real channel
            entry = {"y": y, "points": pts}  # the points of the card
            for method in ("app", "maxlog"):
                d = Demapper(method, constellation=sc, precision="double")
                no = torch.tensor(nv, dtype=torch.float64)
                entry[method] = -np.asarray(d(torch.tensor(y), no)).reshape(-1)  # comnum's sign
            out["llr"][f"{kind}{m}_{nv:g}"] = entry
    for n_rx, n_tx, kind, m, nv in MIMO:
        pts = np.asarray(comnum.Constellation(kind, m).normalized().points, complex)
        sc = _custom(pts)
        n = 100
        h = (
            rng.standard_normal((n, n_rx, n_tx)) + 1j * rng.standard_normal((n, n_rx, n_tx))
        ) / np.sqrt(2)
        idx = rng.integers(0, m, (n, n_tx))
        noise = rng.standard_normal((n, n_rx)) + 1j * rng.standard_normal((n, n_rx))
        y = np.einsum("nij,nj->ni", h, pts[idx]) + np.sqrt(nv / 2) * noise
        s = torch.tensor(np.tile(nv * np.eye(n_rx), (n, 1, 1)), dtype=torch.complex128)
        th, ty = torch.tensor(h), torch.tensor(y)
        # in comnum's layout: the streams along the rows, (n_rx, n) and (n_tx, n)
        entry = {"h": h, "y": y.T, "sent": idx.T, "points": pts}
        for eq in ("lmmse", "zf"):
            det = LinearDetector(
                eq, "symbol", "maxlog", constellation=sc, hard_out=True, precision="double"
            )
            entry[eq] = np.asarray(det(ty, th, s), np.int64).T
        det = MaximumLikelihoodDetector(
            "symbol", "maxlog", n_tx, constellation=sc, hard_out=True, precision="double"
        )
        entry["ml"] = np.asarray(det(ty, th, s), np.int64).T
        for k in KBEST:
            det = KBestDetector(
                "symbol", n_tx, k, constellation=sc, hard_out=True, precision="double"
            )
            entry[f"kbest{k}"] = np.asarray(det(ty, th, s), np.int64).T
        out["mimo"][mkey(n_rx, n_tx, kind, m, nv)] = entry
    return out


@pytest.fixture(scope="module")
def sionna():
    return oracle("sionna_detection", _compute, tool="Sionna")


@pytest.mark.parametrize("max_log", [False, True])
@pytest.mark.parametrize("nv", NOISE)
@pytest.mark.parametrize(("kind", "m"), LLR)
def test_the_llrs_are_those_of_sionna(sionna, kind, m, nv, max_log):
    ref = sionna["llr"][f"{kind}{m}_{nv:g}"]
    c = comnum.Constellation(kind, m).normalized()
    got = c.llr(ref["y"], nv, max_log=max_log)
    want = ref["maxlog" if max_log else "app"]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-10)


def _ours(ref, kind, m, nv, method, **kw):
    c = comnum.Constellation(kind, m).normalized()
    return np.asarray(comnum.mimo_detect(ref["h"], ref["y"], c, nv, method, **kw))


@pytest.mark.parametrize(
    ("ours", "theirs"), [("mmse", "lmmse"), ("zf", "zf"), ("ml", "ml"), ("sphere", "ml")]
)
@pytest.mark.parametrize("config", MIMO)
def test_the_mimo_decisions_are_those_of_sionna(sionna, config, ours, theirs):
    _n_rx, _n_tx, kind, m, nv = config
    ref = sionna["mimo"][mkey(*config)]
    np.testing.assert_array_equal(_ours(ref, kind, m, nv, ours), ref[theirs])


def test_the_kbest_search_of_16_paths_is_that_of_sionna(sionna):
    for config in MIMO:
        _n_rx, _n_tx, kind, m, nv = config
        ref = sionna["mimo"][mkey(*config)]
        np.testing.assert_array_equal(_ours(ref, kind, m, nv, "kbest", k=16), ref["kbest16"])


def test_the_kbest_search_of_4_paths_finds_ml_at_least_as_often_as_sionna(sionna):
    """Sionna sorts the columns of the channel by decreasing norm before its QR
    decomposition (its documentation); comnum keeps the order of the streams. With 4
    paths the two searches prune different branches: their decisions differ on a few
    vectors, neither always closer to y, and comnum's is the ML decision at least as
    often."""
    for config in MIMO:
        _n_rx, _n_tx, kind, m, nv = config
        ref = sionna["mimo"][mkey(*config)]
        ours = _ours(ref, kind, m, nv, "kbest", k=4)
        theirs, ml = np.asarray(ref["kbest4"]), np.asarray(ref["ml"])
        assert np.any(ours != theirs, axis=0).mean() <= 0.1, config
        assert np.all(ours == ml, axis=0).mean() >= np.all(theirs == ml, axis=0).mean(), config
