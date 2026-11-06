#!/usr/bin/env bash
# The library benchmark on a GPU machine (a RunPod pod, or any Linux box with an NVIDIA
# GPU): Sionna on the GPU, OptiCommPy's CuPy split-step on the GPU (scenario 5), comnum
# on the CPU of the same machine, the same scripts as on the laptop. Nothing is compiled
# on the machine: comnum comes from PyPI (or from a wheel sent to the machine), Sionna,
# OptiCommPy and CuPy from PyPI, and the PyTorch of the pod's image is kept (the RunPod
# "PyTorch" template has it with CUDA). From a fresh pod:
#
#   git clone https://github.com/com-rs/comnum && cd comnum
#   bash benchmarks/libraries/runpod.sh --quick      # a few minutes: checks that everything runs
#   bash benchmarks/libraries/runpod.sh              # the full run, 20 to 40 minutes
#
# At the end the script packs benchmarks/libraries/results/runpod/*.json into
# results-runpod.tar.gz and prints the command that fetches it from the laptop
# (`just runpod-fetch`), which also renders the page. Nothing of the pod is needed
# afterwards.
#
# Where comnum comes from (in this order of preference):
#   (default)        pip install comnum==<the version of this checkout> from PyPI;
#   --wheel <file>   a wheel built on the laptop (`just runpod-wheel`) and sent with scp,
#                    for a checkout that is not released yet;
#   --source         the Rust toolchain and maturin on the machine, built from this
#                    checkout (5 to 10 minutes more); the only mode that runs the `comnum`
#                    binary (the library comnum-rust).
# Other options: --quick (small sizes), --scenarios s1_awgn,s4_coding, --libs ...,
# --name <dir> (the subdirectory of results/, default runpod), --modes 1,all,gpu,
# --venv (an isolated environment instead of the Python of the image; torch is then
# downloaded, 2 to 3 GB).
set -euo pipefail

QUICK=""
SCENARIOS="s1_awgn,s2_rayleigh,s3_mimo,s4_coding,s5_ssfm"
LIBS=""
NAME="runpod"
MODES="1,all,gpu"
WHEEL=""
SOURCE=""
VENV=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK="--quick" ;;
    --scenarios) SCENARIOS="$2"; shift ;;
    --libs) LIBS="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --modes) MODES="$2"; shift ;;
    --wheel) WHEEL="$2"; shift ;;
    --source) SOURCE="1" ;;
    --venv) VENV="1" ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -z "$LIBS" ] && LIBS="comnum,comnum-spec,opticommpy,sionna"
[ -n "$SOURCE" ] && [ "$LIBS" = "comnum,comnum-spec,opticommpy,sionna" ] && LIBS="comnum,comnum-spec,comnum-rust,opticommpy,sionna"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('Cargo.toml', 'rb'))['workspace']['package']['version'])")

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "no NVIDIA GPU: nvidia-smi failed" >&2; exit 1; }

echo "== Python"
if [ -n "$VENV" ]; then
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  [ -d benchmarks/libraries/.venv ] || uv venv --python 3.12 benchmarks/libraries/.venv
  # shellcheck disable=SC1091
  . benchmarks/libraries/.venv/bin/activate
  PIP="uv pip install"
else
  PIP="python3 -m pip install -q"
fi
# Sionna and the few packages of the scripts; the torch of the image is kept when it
# satisfies Sionna, downloaded otherwise
$PIP -r benchmarks/libraries/requirements-gpu.txt
# CuPy for OptiCommPy's GPU split-step: the wheel of the CUDA major version of torch
CUDA_MAJOR=$(python3 -c "import torch; print((torch.version.cuda or '12').split('.')[0])")
$PIP "cupy-cuda${CUDA_MAJOR}x"
python3 - <<'PY'
import torch, sionna, cupy, optic
print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available())
print("sionna", sionna.__version__)
print("cupy", cupy.__version__, "devices", cupy.cuda.runtime.getDeviceCount())
print("OptiCommPy", getattr(optic, "__version__", ""))
assert torch.cuda.is_available(), "torch has no CUDA device: install the CUDA build of torch (pip install torch --index-url https://download.pytorch.org/whl/cu126)"
assert cupy.cuda.runtime.getDeviceCount() > 0, "cupy sees no CUDA device"
PY

echo "== comnum $VERSION"
if [ -n "$SOURCE" ]; then
  if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  fi
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
  if ! command -v cc >/dev/null; then
    apt-get update -qq && apt-get install -y -qq build-essential >/dev/null
  fi
  $PIP maturin
  maturin develop --release -m bindings/comnum-py/Cargo.toml
  cargo build --release -p comnum-spec --bin comnum
elif [ -n "$WHEEL" ]; then
  $PIP "$WHEEL"
else
  $PIP "comnum==$VERSION" || {
    echo "comnum $VERSION is not on PyPI: send a wheel (just runpod-wheel, then --wheel) or build it here (--source)" >&2
    exit 1
  }
fi
python3 -c "import comnum; print('comnum', comnum.__version__, comnum.__file__)"

echo "== the benchmark: modes $MODES, libraries $LIBS, scenarios $SCENARIOS -> results/$NAME"
cd benchmarks/libraries
python3 run_all.py $QUICK --scenarios "$SCENARIOS" --libs "$LIBS" --threads "$MODES" \
  --results "results/$NAME" --no-rust-build
tar -czf "$ROOT/results-$NAME.tar.gz" -C results "$NAME"
echo "== done: $ROOT/results-$NAME.tar.gz"
if [ -n "${RUNPOD_PUBLIC_IP:-}" ] && [ -n "${RUNPOD_TCP_PORT_22:-}" ]; then
  echo "   from the laptop: just runpod-fetch $RUNPOD_PUBLIC_IP $RUNPOD_TCP_PORT_22 $NAME"
else
  echo "   from the laptop: just runpod-fetch <host> <ssh port> $NAME"
fi
