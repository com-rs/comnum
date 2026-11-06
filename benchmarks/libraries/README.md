# comnum against other Python link-level libraries

The same links simulated with comnum and with
[komm](https://github.com/rwnobrega/komm),
[OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) (with and without numba)
and [Sionna](https://github.com/NVlabs/sionna) (PyTorch, CPU): do the results agree,
with the closed form when there is one and with each other otherwise, and how long
does each library take? `tools/benchmarks_page.py` renders the results into
`docs/python/benchmarks.rst`, which states the rules (same work, intervals, warm-up,
median of five runs, threads, precision).

| Script | Scenario | komm | OptiCommPy | Sionna | comnum |
|---|---|:-:|:-:|:-:|:-:|
| `s1_awgn.py` | uncoded Gray QPSK and 16-QAM over AWGN, BER vs the exact closed form | yes | yes | yes | yes |
| `s2_rayleigh.py` | BPSK and QPSK over flat Rayleigh fading, BER vs theory | — | — | yes | yes |
| `s3_mimo.py` | 4 × 4 16-QAM, i.i.d. Rayleigh: ZF (vs theory), MMSE, ML | — | — | yes | yes |
| `s4_coding.py` | (133, 171) soft Viterbi; 5G NR LDPC (1024, 2048); one LDPC code and the same LLRs given to three BP decoders | Viterbi | BP decoder | yes | yes |
| `s5_ssfm.py` | split-step Fourier, 16-QAM, 32 GBd, 10 × 80 km, 1 km step: NMSE of the fields | — | yes | yes | yes |

comnum runs three ways: the functions of the `comnum` package called block by block
(`comnum`), its parallel Monte Carlo runner from Python (`comnum-spec`, `comnum.run_spec`)
and the `comnum` binary of Rust on the same specification (`comnum-rust`, scenarios 1
and 4). "OptiCommPy without numba" is `NUMBA_DISABLE_JIT=1`, numba's own switch: the
`@njit` functions of OptiCommPy run as plain Python (OptiCommPy has no option of its
own); its results are those of the compiled code, so it is only timed. OptiCommPy is
GPL-3.0: the scripts import it, nothing of it is copied.

## Reproducing

On a quiet machine, from the root of the repository (Python 3.12; the versions are
pinned in `requirements.txt`):

```bash
uv venv --python 3.12 benchmarks/libraries/.venv
source benchmarks/libraries/.venv/bin/activate
uv pip install -r benchmarks/libraries/requirements.txt
maturin develop --release                    # comnum, from this checkout
cd benchmarks/libraries
python run_all.py                            # builds target/release/comnum first
python footprint.py                          # results/footprint.json: what each library installs
python ../../tools/benchmarks_page.py        # docs/python/benchmarks.rst
python ../../tools/highlights.py             # the highlights of the front pages, docs/python/why.rst
```

`python run_all.py --quick` runs everything at small sizes in about two minutes, to
check that it works; its timings mean nothing. Other options: `--scenarios
s1_awgn,s4_coding`, `--libs comnum,sionna`, `--threads 1` (default `1,all`),
`--no-rust-build`, `--keep-work`. One scenario and one library:

```bash
python s3_mimo.py --lib sionna --threads 1 --out /tmp/s3_sionna.json [--quick]
```

(a script started by hand re-executes itself with the thread settings of `--threads`).

## What `run_all.py` does

Every (scenario, library, threads) runs in its own process, with every thread pool at
one thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS` = 1,
`torch.set_num_threads(1)`, `threads=1` for comnum) or at the defaults of the
libraries (every core). The error rates come from the single-thread runs; the
all-thread runs only time. OptiCommPy with numba starts from an empty numba cache, so
that its first call measures the compilation. Each scenario writes
`results/<scenario>.json`: the error rates with their 95 % intervals and the closed
form, the timings (first call, then the median of five runs), the thread settings,
the precision of every library, the machine (`sysctl`, `sw_vers`), the Python and
package versions, the comnum commit and the command of every run.

The repository's `.gitignore` ignores every `results/` directory: commit the result
files with `git add -f benchmarks/libraries/results/*.json`.

## Expected duration

About two minutes with `--quick`. The full run (10⁷ bits per point, 2000 frames per
point, 2¹⁴ symbols through the fibre, both thread modes) is estimated at 15 to 20
minutes by scaling a quick run, plus the build of `target/release/comnum` when it is
out of date. Most of it is Sionna's exhaustive ML detector (scenario 3) and its
decoders (scenario 4), and the exhaustive ML search of comnum (scenario 3).
`--scenarios` and `--libs` run a part of it.

## On a GPU (RunPod or any Linux machine with an NVIDIA GPU)

Sionna is designed for GPUs; the comparison above runs it on the CPU of a laptop. The
mode `--threads gpu` runs it on the CUDA device (Sionna's `config.device`, which
every block of Sionna creates its tensors on, set to `cuda:0`; the result brought back
to the host inside the timed call; in the modes `1` and `all` it is set to `cpu`,
since Sionna's own default is the GPU as soon as it sees one), OptiCommPy's CuPy split-step (`optic.models.modelsGPU.ssfm`, scenario 5, in
complex128 and complex64; a scenario names the libraries with a GPU path in
`GPU_LIBS`) on the same device, and comnum on the CPU of the same machine, the only
path it has. Nothing is compiled on the machine: comnum, Sionna, OptiCommPy and CuPy
come from PyPI, and the PyTorch of the image (the RunPod "PyTorch" template ships it
with CUDA) is kept:

```bash
git clone https://github.com/com-rs/comnum && cd comnum
bash benchmarks/libraries/runpod.sh --quick          # a few minutes: checks that everything runs
bash benchmarks/libraries/runpod.sh                  # the full run, 20 to 40 minutes
```

The script installs `requirements-gpu.txt` (Sionna, OptiCommPy and the packages of
the scripts) and the CuPy wheel of the image's CUDA version, `pip install
comnum==<the version of the checkout>`, and runs `run_all.py --threads 1,all,gpu
--results results/runpod` with comnum, its runner, OptiCommPy and Sionna (komm is not
needed there). For a checkout that is not on PyPI yet, `just
runpod-wheel` builds its Linux wheel on the laptop (maturin with the zig linker: `uv pip
install ziglang` once) to send with `scp` and install with `--wheel <file>`; `--source`
installs the Rust toolchain and builds on the machine instead (the only mode that also
runs the `comnum` binary, the library comnum-rust).

The results land in `results/runpod/*.json`, with the GPU and the driver recorded next
to the CPU, packed into `results-runpod.tar.gz`; the script prints the command that
fetches them: `just runpod-fetch <host> <ssh port>` copies the archive, unpacks it into
`results/runpod/` and renders the page, where `python tools/benchmarks_page.py` adds a
section "On another machine: runpod", the same tables with the column *GPU*. Commit
them with `git add -f benchmarks/libraries/results/runpod/*.json`. The relative times
stay against comnum on one thread of that machine, so that the two machines are not
compared with each other.

Which pod: the GPU hardly matters for these sizes (2¹⁴ to 2¹⁶ samples through the
fibre, batches of 10⁶ bits), the CPU does, since every comnum time is a CPU time and
RunPod pods get a slice of the host's CPU that depends on the GPU type. A secure-cloud
RTX 4090 pod (24 GB, the GPU of a lab workstation) and an H100 or A100 pod (the
data-centre reference) make two useful points, each under an hour; take the pod with
the most vCPUs offered for that GPU, and note the CPU model the result file records.
Inside a pod `os.cpu_count()` is the host's core count, far above the pod's quota:
the scripts size the pools of the mode *all threads* to the quota (`cgroup cpu.max`,
the affinity mask), recorded as `cpus_available` next to `logical_cores`.
The "RunPod PyTorch" template (CUDA 12) is the one the script assumes.
