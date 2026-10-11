HAS_GPU := $(shell nvidia-smi -L >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_GPU),1)
    DEFAULT_ENV := .venv-gpu
else
    DEFAULT_ENV := .venv-cpu
endif

.PHONY: install install-cpu install-xrootd install-cuda-ext install-frnn check run clean

install: ENV ?= .venv-gpu
install:
	UV_PROJECT_ENVIRONMENT=$(ENV) uv sync --extra gpu --frozen
	UV_PROJECT_ENVIRONMENT=$(ENV) uv run --frozen python check_acorn.py

install-cpu: ENV ?= .venv-cpu
install-cpu:
	UV_PROJECT_ENVIRONMENT=$(ENV) uv sync --frozen
	UV_PROJECT_ENVIRONMENT=$(ENV) uv run --frozen python check_acorn.py

install-xrootd: ENV ?= .venv-xrootd
install-xrootd:
	UV_PROJECT_ENVIRONMENT=$(ENV) uv sync --extra gpu --extra xrootd --frozen
	UV_PROJECT_ENVIRONMENT=$(ENV) uv run --frozen python check_acorn.py

install-cuda-ext: ENV ?= .venv-gpu
install-cuda-ext:
	@command -v nvcc >/dev/null 2>&1 || (echo "ERROR: nvcc not found on PATH. Load a CUDA toolkit matching torch (2.4.0+cu121), e.g. 'module load cudatoolkit/12.2 gcc/12.2.0', then rerun."; exit 1)
	@test -f setup.py || (echo "ERROR: setup.py not found (CUDA-ext build requires the DWALK setup.py merged into dev)."; exit 1)
	ACORN_BUILD_CUDA_EXT=1 UV_PROJECT_ENVIRONMENT=$(ENV) uv sync --extra gpu --reinstall-package acorn --frozen
	UV_PROJECT_ENVIRONMENT=$(ENV) uv run --frozen python check_acorn.py

# Build and install the optional FRNN (+ prefix_sum) GPU nearest-neighbour
# library into the uv environment. FRNN is not managed by uv, so a plain
# `uv sync` removes it again - re-run this target afterwards, or sync with
# `uv sync --inexact`. Pass ENV=<dir> to target a specific environment.
install-frnn:
	scripts/install_frnn.sh $(ENV)

check: ENV ?= $(DEFAULT_ENV)
check:
	UV_PROJECT_ENVIRONMENT=$(ENV) uv run --frozen python check_acorn.py

run: ENV ?= $(DEFAULT_ENV)
run:
	@if [ -z "$(filter-out run,$(MAKECMDGOALS))" ]; then \
	  echo "Usage: make run [ENV=<env>] <command> [args...]"; exit 1; fi
	UV_PROJECT_ENVIRONMENT=$(ENV) uv run $(filter-out run,$(MAKECMDGOALS))

%:
	@:

clean:
	rm -rf .venv-gpu .venv-cpu .venv-xrootd
