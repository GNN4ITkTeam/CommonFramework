#!/usr/bin/env bash
#
# install_frnn.sh - build and install FRNN (and its `prefix_sum` runtime
# dependency) into an existing acorn uv environment.
#
# Why this script exists
# ----------------------
# FRNN is not published on PyPI and is intentionally kept OUT of
# pyproject.toml / uv.lock: it is optional and its CUDA extension build is
# environment-specific. Because uv does not know about it, a plain `uv sync`
# treats the manually-installed `frnn` / `prefix_sum` packages as extraneous
# and REMOVES them. After any plain `uv sync` (or `make install*`) re-run this
# script, or sync with `uv sync --inexact` to leave them in place.
#
# The build has a few sharp edges this script hides:
#   * the host `c++` on Perlmutter is GCC 7.5, too old -> we use g++-13;
#   * the CUDA toolkit must share the *major* version of torch's CUDA build;
#   * the HPC SDK keeps cuBLAS headers under math_libs/<cuda>/include, so a
#     stale CPATH from a different toolkit mixes headers and fails with
#     "cudaEmulation... has not been declared";
#   * `frnn` imports `prefix_sum` at runtime, so both must be installed;
#   * the extension must be compiled for the local GPU arch.
#
# Usage
# -----
#   scripts/install_frnn.sh [ENV_DIR]
#
# ENV_DIR is the uv environment to install into. If omitted it defaults, in
# order, to $UV_PROJECT_ENVIRONMENT, ./.venv, then ./.venv-gpu. Relative paths
# are resolved against the repository root.
#
# Overrides (environment variables)
# ---------------------------------
#   CUDA_HOME              CUDA toolkit to build against. Must share the major
#                          version of the env's torch CUDA build. Auto-detected.
#   FRNN_CUDA_MODULE       Lmod module loaded when CUDA_HOME is unset or has the
#                          wrong major (default: cudatoolkit/12.9).
#   TORCH_CUDA_ARCH_LIST   GPU arch list (default: detected via nvidia-smi,
#                          else 8.0).
#   FRNN_REPO              FRNN git URL (default: https://github.com/xju2/FRNN.git).
#   FRNN_BUILD_DIR         scratch directory (default: a fresh mktemp -d).
#   MAX_JOBS               parallel compile jobs (default: 8).
#   CXX / CC               host compiler (default: g++-13 / gcc-13 when present).
#
set -euo pipefail

log()  { printf '\033[1;34m[frnn]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[frnn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[frnn] error:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- resolve the target environment ---------------------------------------
if [[ -n "${1:-}" ]]; then
    ENV_DIR="$1"
elif [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    ENV_DIR="$UV_PROJECT_ENVIRONMENT"
elif [[ -d "$REPO_ROOT/.venv" ]]; then
    ENV_DIR="$REPO_ROOT/.venv"
elif [[ -d "$REPO_ROOT/.venv-gpu" ]]; then
    ENV_DIR="$REPO_ROOT/.venv-gpu"
else
    ENV_DIR="$REPO_ROOT/.venv"
fi
[[ "$ENV_DIR" = /* ]] || ENV_DIR="$REPO_ROOT/$ENV_DIR"

PY="$ENV_DIR/bin/python"
[[ -x "$PY" ]] || die "no python at $PY - create the environment first (e.g. 'make install')."

# --- inspect the target environment ---------------------------------------
TORCH_VER="$("$PY" -c 'import torch; print(torch.__version__)')" \
    || die "torch is not importable in $ENV_DIR"
CUDA_TAG="$("$PY" -c 'import torch; print(torch.version.cuda or "")')"
[[ -n "$CUDA_TAG" ]] || die "torch in $ENV_DIR was built without CUDA; FRNN requires a CUDA torch build."
TORCH_CUDA_MAJOR="${CUDA_TAG%%.*}"

log "target env : $ENV_DIR"
log "torch      : $TORCH_VER (CUDA $CUDA_TAG)"

# --- pick a CUDA toolkit matching torch's CUDA major ----------------------
nvcc_major() { "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1; }

ensure_module_cmd() {
    command -v module >/dev/null 2>&1 && return 0
    local f
    for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash \
             /opt/cray/pe/lmod/lmod/init/bash; do
        if [[ -r "$f" ]]; then
            # shellcheck disable=SC1090
            source "$f" >/dev/null 2>&1 || true
            command -v module >/dev/null 2>&1 && return 0
        fi
    done
    return 1
}

resolve_cuda_home() {
    local mod="${FRNN_CUDA_MODULE:-cudatoolkit/12.9}" cand=""

    if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
        if [[ "$(nvcc_major "$CUDA_HOME")" == "$TORCH_CUDA_MAJOR" ]]; then
            echo "$CUDA_HOME"; return
        fi
        warn "CUDA_HOME=$CUDA_HOME is CUDA $(nvcc_major "$CUDA_HOME").x but torch needs CUDA ${TORCH_CUDA_MAJOR}.x - looking for a matching toolkit."
    fi

    if ensure_module_cmd; then
        if module load "$mod" >/dev/null 2>&1; then
            local v
            for v in CUDA_HOME CUDATOOLKIT_HOME; do
                cand="${!v:-}"
                if [[ -n "$cand" && -x "$cand/bin/nvcc" && "$(nvcc_major "$cand")" == "$TORCH_CUDA_MAJOR" ]]; then
                    echo "$cand"; return
                fi
            done
            cand="$(module show "$mod" 2>/dev/null | sed -n 's/.*setenv("CUDATOOLKIT_HOME","\([^"]*\)".*/\1/p' | head -1)"
            if [[ -n "$cand" && -x "$cand/bin/nvcc" && "$(nvcc_major "$cand")" == "$TORCH_CUDA_MAJOR" ]]; then
                echo "$cand"; return
            fi
        fi
    fi

    if command -v nvcc >/dev/null 2>&1; then
        cand="$(dirname "$(dirname "$(command -v nvcc)")")"
        if [[ "$(nvcc_major "$cand")" != "$TORCH_CUDA_MAJOR" ]]; then
            warn "using CUDA $(nvcc_major "$cand").x with a torch built for CUDA ${TORCH_CUDA_MAJOR}.x - the build may fail."
        fi
        echo "$cand"; return
    fi

    echo ""
}

CUDA_HOME="$(resolve_cuda_home)"
[[ -n "$CUDA_HOME" ]] || die "no CUDA toolkit found. Load one matching torch (e.g. 'module load ${FRNN_CUDA_MODULE:-cudatoolkit/12.9}') or set CUDA_HOME."

export CUDA_HOME CUDATOOLKIT_HOME="$CUDA_HOME" NVHPC_CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# HPC SDK keeps cuBLAS/cuSPARSE headers in math_libs/<cuda>/include, a sibling
# of cuda/<cuda>. If CPATH points at a different toolkit's math_libs the host
# compiler mixes headers and fails. Pin CPATH to this toolkit.
_cuda_ver="$(basename "$CUDA_HOME")"
_sdk_root="$(dirname "$(dirname "$CUDA_HOME")")"
_math_libs="$_sdk_root/math_libs/$_cuda_ver"
if [[ -d "$_math_libs/include" ]]; then
    export CPATH="$_math_libs/include:$CUDA_HOME/include"
    export LD_LIBRARY_PATH="$_math_libs/lib64:$LD_LIBRARY_PATH"
else
    export CPATH="$CUDA_HOME/include"
fi
log "CUDA toolkit: $CUDA_HOME (nvcc $(nvcc_major "$CUDA_HOME").x)"

# --- host compiler (default c++ may be far too old) -----------------------
pick() { local c; for c in "$@"; do command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }; done; echo ""; }

if [[ -z "${CXX:-}" ]]; then
    CXX="$(pick g++-13 g++-12 g++)"
fi
[[ -n "${CXX:-}" ]] || die "no C++ compiler found (need g++ >= 12); set CXX explicitly."
if [[ -z "${CC:-}" ]]; then
    CC="$(pick gcc-13 gcc-12 gcc)"
fi
export CXX CC CUDAHOSTCXX="$CXX"
log "host CXX    : $CXX"

# --- target GPU arch -------------------------------------------------------
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
log "arch list   : $TORCH_CUDA_ARCH_LIST"

# --- fetch FRNN source -----------------------------------------------------
FRNN_BUILD_DIR="${FRNN_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/frnn-build.XXXXXX")}"
mkdir -p "$FRNN_BUILD_DIR"
FRNN_SRC="$FRNN_BUILD_DIR/FRNN"
FRNN_REPO="${FRNN_REPO:-https://github.com/xju2/FRNN.git}"
if [[ ! -d "$FRNN_SRC/.git" ]]; then
    log "cloning     : $FRNN_REPO"
    git clone --depth 1 "$FRNN_REPO" "$FRNN_SRC"
fi
git -C "$FRNN_SRC" submodule update --init --depth 1 external/prefix_sum

# --- build in a scratch venv (leaves the target env untouched) -------------
BUILD_VENV="$FRNN_BUILD_DIR/venv"
BASE_PREFIX="$("$PY" -c 'import sys; print(sys.base_prefix)')"
BASE_PY="$BASE_PREFIX/bin/python$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ -x "$BASE_PY" ]] || BASE_PY="$PY"
log "build venv  : $BUILD_VENV"
uv venv --python "$BASE_PY" "$BUILD_VENV"
uv pip install --python "$BUILD_VENV/bin/python" setuptools ninja wheel
uv pip install --python "$BUILD_VENV/bin/python" "torch==$TORCH_VER" \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_TAG//./}"

log "building    : prefix_sum"
MAX_JOBS="${MAX_JOBS:-8}" uv pip install --python "$BUILD_VENV/bin/python" \
    --no-build-isolation --no-deps "$FRNN_SRC/external/prefix_sum"

log "building    : frnn (this takes a couple of minutes)"
MAX_JOBS="${MAX_JOBS:-8}" uv pip install --python "$BUILD_VENV/bin/python" \
    --no-build-isolation --no-deps "$FRNN_SRC"

# --- install the self-contained packages into the target env ---------------
BUILD_SP="$("$BUILD_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
DEST_SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
log "installing  : $DEST_SP"

mkdir -p "$DEST_SP/frnn"
cp -rf "$BUILD_SP/frnn/." "$DEST_SP/frnn/"
cp -rf "$BUILD_SP"/frnn-*.dist-info "$DEST_SP/" 2>/dev/null || true
cp -f  "$BUILD_SP"/prefix_sum*.so "$DEST_SP/"
cp -rf "$BUILD_SP"/prefix_sum-*.dist-info "$DEST_SP/" 2>/dev/null || true

# --- verify ----------------------------------------------------------------
log "verifying   :"
"$PY" - <<'PYEOF'
import torch, frnn
assert torch.cuda.is_available(), "CUDA is not available in this environment"
d = torch.rand(4096, 3, device="cuda")
dist, _, _, _ = frnn.frnn_grid_points(
    points1=d.unsqueeze(0), points2=d.unsqueeze(0),
    lengths1=None, lengths2=None, K=8, r=0.2,
    grid=None, return_nn=False, return_sorted=True,
)
print("  frnn_grid_points OK, dists:", tuple(dist.shape))
try:
    from acorn.stages.graph_construction.models.utils import FRNN_AVAILABLE
    print("  acorn FRNN_AVAILABLE:", FRNN_AVAILABLE)
except Exception as exc:  # pragma: no cover - acorn may not be importable here
    print("  (skipped acorn import:", exc, ")")
PYEOF

cat <<'EOF'

FRNN installed.  Reminder:
  * `uv sync` (and `make install*`) consider frnn/prefix_sum "extraneous" and
    REMOVE them. After such a sync, re-run this script, or protect them with
        uv sync --inexact
  * `uv run` (used by `make run` / `make check`) does NOT remove them.

EOF
