#!/usr/bin/env bash

# Convenience wrapper for installing ACORN with optional CUDA/C++ extensions.
# Input env name, if empty, use default acorn.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

source "${repo_dir}/create_conda_environment.sh" "${1:-acorn}" --cuda-ext
