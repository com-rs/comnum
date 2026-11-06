"""Shared machinery of the library benchmarks: arguments, threads, timing, intervals,
machine and version information, the runs of comnum on a specification (``comnum.run_spec``
and the ``comnum`` binary), JSON output.

Every scenario script runs one library at a time, in its own process, so that the
thread settings (read by NumPy, numba, PyTorch and rayon when they start) apply from
the first import and that one library's imports cannot slow another one down. A
script started by hand re-executes itself with the environment of its ``--threads``
option; ``run_all.py`` starts the scripts with that environment directly.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

# The variants of the libraries, in the order of the tables.
LIBRARIES = {
    "comnum": "comnum from Python: the functions of the comnum package, called block by block",
    "comnum-spec": "comnum from Python: comnum.run_spec, the parallel Monte Carlo runner of Rust",
    "comnum-rust": "comnum from Rust: the comnum command-line binary on the same specification",
    "komm": "komm (NumPy)",
    "opticommpy": "OptiCommPy with numba (its @njit kernels compiled; its CuPy split-step on the GPU in the mode gpu)",
    "opticommpy-nojit": "OptiCommPy without numba (NUMBA_DISABLE_JIT=1: the same @njit kernels run as plain Python)",
    "sionna": "Sionna PHY (PyTorch, eager mode; the CPU, or the GPU in the mode gpu)",
}

# The modes of a run: one thread for every pool, the defaults of the library (every
# core), or the GPU (the libraries a scenario lists in ``GPU_LIBS``, Sionna by default,
# and OptiCommPy's CuPy split-step; comnum has no GPU path; the CPU pools at their
# defaults).
MODES = ["1", "all", "gpu"]
GPU_LIBS_DEFAULT = {"sionna"}

# The variables that set the size of the thread pools, all set to 1 for the
# single-thread runs and left unset (the library default: every core) otherwise.
THREAD_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
]

REPEATS = 5


def available_cpus() -> int:
    """The CPUs this process may use: ``os.cpu_count()`` bounded by the affinity mask
    and by the cgroup CPU quota (a container, a RunPod pod: the host has far more cores
    than the pod may use, and a pool sized by ``os.cpu_count()`` oversubscribes the
    quota)."""
    n = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    quota: float | None = None
    try:  # cgroup v2: "max 100000" or "<quota> <period>"
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, math.ceil(quota)))
    return n


def thread_env(lib: str, threads: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of a run of ``lib`` in the mode ``threads`` ("1", "all" or "gpu").
    In the modes "all" and "gpu" the pools are left at the library defaults, except
    inside a container whose CPU quota is below ``os.cpu_count()``: the pools are then
    sized to the quota (the default would oversubscribe it)."""
    env = dict(os.environ if base is None else base)
    for v in THREAD_VARS:
        env.pop(v, None)
    if threads == "1":
        env.update({v: "1" for v in THREAD_VARS})
    elif (n := available_cpus()) < (os.cpu_count() or 1):
        env.update({v: str(n) for v in THREAD_VARS})
    env.pop("NUMBA_DISABLE_JIT", None)
    if lib == "opticommpy-nojit":
        env["NUMBA_DISABLE_JIT"] = "1"
    if lib == "opticommpy":
        # a fresh cache: the first call compiles the kernels instead of loading them
        cache = RESULTS / "_work" / f"numba_cache_{os.getpid()}_{time.time_ns()}"
        env["NUMBA_CACHE_DIR"] = str(cache)
    env["BENCH_ENV"] = f"{lib}:{threads}"
    return env


def parse_args(cases: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=sys.modules["__main__"].__doc__)
    p.add_argument("--lib", required=True, choices=list(LIBRARIES))
    p.add_argument("--threads", default="1", choices=MODES, help="1, all (every core) or gpu")
    p.add_argument("--quick", action="store_true", help="small sizes, to check that it runs")
    p.add_argument("--timing-only", action="store_true", help="skip the error-rate sweep")
    p.add_argument("--cases", default=",".join(cases), help="comma-separated cases")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--work", default=str(RESULTS / "_work"), help="scratch directory")
    p.add_argument("--out", required=True, help="JSON file of this run")
    args = p.parse_args()
    args.cases = [c for c in args.cases.split(",") if c]
    if os.environ.get("BENCH_ENV") != f"{args.lib}:{args.threads}":
        env = thread_env(args.lib, args.threads)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return args


TORCH_DEVICE = "cpu"
"""The device of the Sionna runs, set by ``setup_torch``: the scripts put the tensors
they create themselves on it."""


def setup_torch(threads: str) -> dict:
    """Set the PyTorch thread pools (before any computation) and the device of Sionna:
    ``config.device`` of Sionna, which every block of Sionna creates its tensors on.
    It is ``cpu`` in the modes "1" and "all" (Sionna's own default is the first CUDA
    device when there is one, which would put the CPU modes on the GPU), and in the
    mode "gpu" ``COMNUM_BENCH_DEVICE``, ``cuda:0`` by default (a bare ``cuda`` becomes
    ``cuda:0``, the form Sionna accepts). ``torch.set_default_device`` is not used: its
    device context adds a Python overhead to every operation. Returns what was set."""
    global TORCH_DEVICE
    import torch

    info: dict = {}
    if threads == "1":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    elif (n := available_cpus()) < (os.cpu_count() or 1):
        torch.set_num_threads(n)  # a container: the quota, not the host's cores
    device = "cpu"
    if threads == "gpu":
        device = os.environ.get("COMNUM_BENCH_DEVICE", "cuda:0")
        if device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("the mode gpu needs a CUDA device: torch.cuda.is_available() is False")
    try:
        from sionna.phy import config

        if hasattr(config, "device"):
            config.device = device
    except ImportError:
        pass
    TORCH_DEVICE = device
    info["device"] = device
    if device.startswith("cuda"):
        i = torch.device(device).index or 0
        info["gpu"] = torch.cuda.get_device_name(i)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
        info["cuda"] = torch.version.cuda
    info.update(
        torch_intra_op_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
    )
    return info


def to_numpy(t):
    """A PyTorch tensor (on any device) as a NumPy array; anything else through
    ``np.asarray``."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    import numpy as np

    return np.asarray(t)


def time_it(fn: Callable[[], object], repeats: int = REPEATS) -> dict:
    """One warm-up call (JIT compilation, caches, first allocations), reported apart,
    then the median of ``repeats`` timed calls (``time.perf_counter``)."""
    t0 = time.perf_counter()
    fn()
    warmup = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "warmup_s": warmup,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "times_s": times,
        "repeats": repeats,
    }


def wilson(errors: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion (95 % by default)."""
    if total == 0:
        return (0.0, 1.0)
    p = errors / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    h = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate_point(errors: int, total: int, theory: float | None = None, **extra) -> dict:
    """An error rate of independent trials, its 95 % interval and its z-score against
    ``theory`` (binomial standard error at the theory)."""
    lo, hi = wilson(errors, total)
    p = errors / total if total else float("nan")
    d = {
        "errors": int(errors),
        "total": int(total),
        "value": p,
        "lo": lo,
        "hi": hi,
        "ci": "binomial",
        **extra,
    }
    if theory is not None:
        d["theory"] = float(theory)
        sd = math.sqrt(max(theory * (1 - theory), 1e-300) / total)
        d["z"] = (p - theory) / sd
    return d


def frame_ber_point(errors_per_frame, k: int, theory: float | None = None, **extra) -> dict:
    """The bit error rate of coded frames: the errors of a frame are not independent
    (they come in bursts), so the interval is that of the mean of the per-frame error
    counts (central limit theorem over the frames), not a binomial interval."""
    import numpy as np

    e = np.asarray(errors_per_frame, dtype=float)
    f = e.size
    p = e.sum() / (f * k)
    se = e.std(ddof=1) / math.sqrt(f) / k if f > 1 else float("inf")
    # no error: the rule of three (95 %) on the frames, the interval of the mean is empty
    hi = 3.0 / (f * k) if e.sum() == 0 else p + 1.96 * se
    d = {
        "errors": int(e.sum()),
        "total": int(f * k),
        "frames": int(f),
        "value": float(p),
        "lo": float(max(0.0, p - 1.96 * se)),
        "hi": float(hi),
        "se": float(se),
        "ci": "per-frame",
        **extra,
    }
    if theory is not None:
        d["theory"] = float(theory)
        d["z"] = float((p - theory) / se) if se > 0 else float("nan")
    return d


def setup_cupy() -> dict:
    """Check that CuPy sees a CUDA device and return the device, CuPy and CUDA
    versions, for a run of OptiCommPy in the mode ``gpu``."""
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("the mode gpu needs a CUDA device: cupy sees none")
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    return {
        "device": "cuda",
        "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "gpu_memory_gb": round(props["totalGlobalMem"] / 2**30, 1),
        "cuda": str(cp.cuda.runtime.runtimeGetVersion()),
        "cupy": cp.__version__,
    }


def versions() -> dict[str, str]:
    out = {}
    for pkg in [
        "comnum",
        "komm",
        "OptiCommPy",
        "sionna",
        "torch",
        "numba",
        "cupy",
        "cupy-cuda12x",
        "cupy-cuda13x",
        "llvmlite",
        "numpy",
        "scipy",
    ]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_info() -> dict:
    """The machine, gathered automatically (macOS: sysctl and sw_vers; Linux:
    /proc/cpuinfo and os-release; the GPU from nvidia-smi when there is one)."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        or _linux_cpu()
        or platform.processor(),
        "logical_cores": os.cpu_count(),
        "cpus_available": available_cpus(),
        "physical_cores": _run(["sysctl", "-n", "hw.physicalcpu"]) or _linux_physical_cores(),
        "core_levels": [],
        "ram_gb": None,
        "macos": _run(["sw_vers", "-productVersion"]),
        "linux": _linux_release(),
        "rustc": _run(["rustc", "--version"]),
    }
    for i in range(int(_run(["sysctl", "-n", "hw.nperflevels"]) or 0)):
        name = _run(["sysctl", "-n", f"hw.perflevel{i}.name"])
        n = _run(["sysctl", "-n", f"hw.perflevel{i}.physicalcpu"])
        info["core_levels"].append(f"{n} {name}")
    mem = _run(["sysctl", "-n", "hw.memsize"])
    if mem:
        info["ram_gb"] = round(int(mem) / 2**30, 1)
    elif Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["ram_gb"] = round(int(line.split()[1]) / 2**20, 1)
    smi = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    if smi:
        name, mem, driver = [x.strip() for x in smi.splitlines()[0].split(",")]
        info["gpu"] = {"name": name, "memory": mem, "driver": driver}
    return info


def _linux_cpu() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _linux_physical_cores() -> str | None:
    out = _run(["lscpu", "-p=CORE,SOCKET"])
    if not out:
        return None
    cores = {line for line in out.splitlines() if not line.startswith("#")}
    return str(len(cores)) if cores else None


def _linux_release() -> str | None:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        return None
    return None


def git_commit() -> dict:
    commit = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": bool(status)}


def threading_info() -> dict:
    """The thread pools of the libraries loaded by the run, as they ended up."""
    info: dict = {"cpu_count": os.cpu_count(), "cpus_available": available_cpus()}
    if "numba" in sys.modules:
        import numba

        info["numba_threads"] = numba.get_num_threads()
        info["numba_jit_disabled"] = bool(numba.config.DISABLE_JIT)
    if "torch" in sys.modules:
        import torch

        info["torch_intra_op_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
        info["torch_device"] = TORCH_DEVICE
    return info


def spec_threads(args: argparse.Namespace) -> int:
    """The ``threads`` option of ``comnum.run_spec`` and of the ``comnum`` binary for
    ``--threads``: 1, or 0 (every core)."""
    return 1 if args.threads == "1" else 0


def rust_binary() -> Path:
    """The ``comnum`` command-line binary of the repository (``run_all.py`` builds it)."""
    return ROOT / "target" / "release" / "comnum"


def run_rust_cli(spec_dict: dict, work: Path, threads: int) -> tuple[dict, float]:
    """Run the comnum binary on a specification: its JSON report and the wall time of
    the process."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{spec_dict['name']}.json"
    path.write_text(json.dumps(spec_dict), encoding="utf-8")
    out_dir = work / f"{spec_dict['name']}_out"
    cmd = [
        str(rust_binary()),
        "run",
        str(path),
        "--out",
        str(out_dir),
        "--threads",
        str(threads),
        "--quiet",
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True)
    wall = time.perf_counter() - t0
    report = json.loads((out_dir / f"{spec_dict['name']}.json").read_text(encoding="utf-8"))
    return report, wall


def spec_case(
    args: argparse.Namespace,
    res: dict,
    sweep: dict,
    timing: dict,
    points: Callable[[list[dict]], list[dict]],
    *,
    rust: bool = False,
    key: str = "timing",
    **workload: object,
) -> dict:
    """One case of the comnum runners on specifications: ``comnum.run_spec``
    (``comnum-spec``) or, with ``rust``, the ``comnum`` binary (``comnum-rust``, the wall
    time of the process).

    Unless ``--timing-only``, the ``sweep`` specification is run once (``sweep_s``) and
    its conditions become ``points``; the ``timing`` specification is timed by
    :func:`time_it` under ``key``, with the ``workload`` fields (``units``, ``unit``,
    ``workload``). ``res`` (the case, its precision…) gets ``threads_option``, with
    ``rust`` the ``binary`` (relative to the repository), and these results."""
    threads = spec_threads(args)
    res["threads_option"] = threads
    if rust:
        res["binary"] = str(rust_binary().relative_to(ROOT))
        work = Path(args.work) / "rust"

        def run(s: dict) -> tuple[list[dict], float]:
            report, wall = run_rust_cli(s, work, threads)
            return report["conditions"], wall
    else:
        import comnum

        def run(s: dict) -> tuple[list[dict], float]:
            t0 = time.perf_counter()
            rows = comnum.run_spec(s, threads=threads)
            return rows, time.perf_counter() - t0

    if not args.timing_only:
        rows, wall = run(sweep)
        res["sweep_s"] = wall
        res["points"] = points(rows)
    res[key] = time_it(lambda: run(timing))
    res[key].update(workload)
    return res


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def main(scenario: str, cases: dict[str, dict], runners: dict[str, Callable]) -> None:
    """The entry point of a scenario script: run the cases of ``--lib`` that it
    supports and write the JSON of the run."""
    args = parse_args(list(cases))
    todo = [c for c in args.cases if cases[c].get(args.lib) is True]
    meta = {
        "scenario": scenario,
        "library": args.lib,
        "threads": args.threads,
        "quick": args.quick,
        "timing_only": args.timing_only,
        # the cache directory of numba is left out: a local path, new for every run
        "env": {v: os.environ.get(v) for v in [*THREAD_VARS, "NUMBA_DISABLE_JIT"]},
        "versions": versions(),
    }
    t0 = time.perf_counter()
    results = runners[args.lib](args, todo) if todo else []
    meta["wall_s"] = time.perf_counter() - t0
    meta["threading"] = threading_info()
    write_json(args.out, {**meta, "cases": results})
