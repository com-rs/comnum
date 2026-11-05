"""Invalid input from Python raises an exception, never a Rust panic.

A panic of the extension reaches Python as ``pyo3_runtime.PanicException``, a subclass of
``BaseException`` that ``except Exception`` does not catch; ``docs/ARCHITECTURE.md``
("Errors and panics") reserves panics for broken invariants. These tests call every
public function of the package, and every method of an instance of every class, with
degenerate arguments (empty, NaN, infinite, negative, zero, of the wrong type or shape)
and accept any exception but a panic. The instances are built by the constructors that
the docstrings show, and by ``EXTRA`` for the classes whose docstrings show none.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import comnum

DEGENERATE: list[Any] = [
    np.array([], dtype=complex),
    np.array([np.nan + 0j] * 7),
    np.array([np.inf] * 8),
    np.array([1.0, 2.0]),
    np.zeros((2, 3)),
    np.array([5, 9, 300]),
    -1,
    0,
    7,
    float("nan"),
    "zzz",
    [0.5],
]

# classes whose docstrings construct no instance
EXTRA = {
    "Edfa": "comnum.Edfa()",
    "LowRank": "comnum.LowRank(np.eye(4, dtype=complex), 0.1, 2)",
    "ManakovPmd": "comnum.ManakovPmd(comnum.FiberParams(n_spans=1))",
    "ManakovSsfm": "comnum.ManakovSsfm(comnum.FiberParams(n_spans=1))",
    "NrHarqProcess": "comnum.NrHarqProcess(comnum.NrTransportBlock(1000, 0.5))",
    "Photodiode": "comnum.Photodiode()",
    "PointingError": "comnum.PointingError(1.0, 0.1, 0.2)",
    "Wiener2d": "comnum.Wiener2d([(0, 0), (0, 4), (3, 0)], 0.01, 1e-3, 0.01, 2)",
}

CONSTRUCTOR = re.compile(r"comnum\.[A-Z]\w*(?:\.\w+)?\([^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)")


def is_panic(e: BaseException) -> bool:
    return "Panic" in type(e).__name__


def panics(f: Callable[..., Any]) -> Iterator[str]:
    """The calls of `f` with one to three copies of a degenerate argument that panic."""
    for n in (1, 2, 3):
        for v in DEGENERATE:
            try:
                f(*([v] * n))
            except Exception:  # any exception but a panic is accepted
                pass
            except BaseException as e:
                if not is_panic(e):
                    raise
                yield f"{n} x {v!r:.30}: {str(e).splitlines()[0][:80]}"


def instances() -> dict[str, Any]:
    """One instance per class of the package, built from the docstrings and EXTRA."""
    doc = []
    for _, cls in inspect.getmembers(comnum, inspect.isclass):
        doc.append(cls.__doc__ or "")
        doc.extend(getattr(m, "__doc__", "") or "" for _, m in inspect.getmembers(cls))
    names = {"comnum": comnum, "np": np}
    out: dict[str, Any] = {}
    for expr in [*sorted(set(CONSTRUCTOR.findall("\n".join(doc)))), *EXTRA.values()]:
        try:
            v = eval(expr, names)  # the constructors of the package's docstrings
        except Exception:  # an example that needs its context is skipped
            continue
        if type(v).__module__.startswith("comnum"):
            out.setdefault(type(v).__name__, v)
    return out


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # NaN cast to integers
def test_no_function_panics_on_degenerate_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a function given a file name writes here
    found = {}
    for name, f in inspect.getmembers(comnum):
        if name.startswith("_") or inspect.isclass(f) or inspect.ismodule(f) or not callable(f):
            continue
        if bad := list(panics(f)):
            found[name] = bad[0]
    assert not found, found


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # NaN cast to integers
def test_no_method_panics_on_degenerate_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    objs = instances()
    classes = [
        n
        for n, c in inspect.getmembers(comnum, inspect.isclass)
        if not issubclass(c, BaseException)
    ]
    # the classes built only by a chain, a runner or a channel: Block, Chain, ChannelTaps,
    # OwcIllumination, OwcImpulseResponse, Results, Trace, Var
    assert len(objs) >= len(classes) - 8, sorted(set(classes) - set(objs))
    found = {}
    for cname, obj in sorted(objs.items()):
        for mname, _ in inspect.getmembers(type(obj)):
            if mname.startswith("_"):
                continue
            try:
                m = getattr(obj, mname)
            except Exception:  # a property that needs a state
                continue
            if callable(m) and (bad := list(panics(m))):
                found[f"{cname}.{mname}"] = bad[0]
    assert not found, found


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: comnum.Constellation("qam", 4).energy_with([0.5]), ValueError),
        (lambda: comnum.LowRank(np.eye(4, dtype=complex), 0.1, 2).mse_floor(np.eye(2)), ValueError),
    ],
    ids=["energy_with", "mse_floor"],
)
def test_the_panics_found_by_the_probe_are_errors(call, error):
    with pytest.raises(error):
        call()
