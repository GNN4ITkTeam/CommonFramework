import importlib

import torch


_IMPORT_ERROR = None
_NATIVE_MODULE = None


def _load_native_module():
    global _IMPORT_ERROR, _NATIVE_MODULE

    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE

    if _IMPORT_ERROR is not None:
        return None

    try:
        _NATIVE_MODULE = importlib.import_module(
            "acorn.cuda_ext._trace_selected_paths_cuda"
        )
    except ImportError as exc:
        _IMPORT_ERROR = exc
        return None

    return _NATIVE_MODULE


def trace_selected_paths_available():
    return _load_native_module() is not None


def trace_selected_paths_cuda(best_child, selected_roots, num_nodes=None):
    module = _load_native_module()
    if module is None:
        raise RuntimeError(
            "The optional ACORN CUDA path-tracing extension is not available. "
            "Build with ACORN_BUILD_CUDA_EXT=1 and --no-build-isolation to enable it."
        ) from _IMPORT_ERROR

    if not best_child.is_cuda:
        raise ValueError("best_child must be a CUDA tensor")
    if not selected_roots.is_cuda:
        raise ValueError("selected_roots must be a CUDA tensor")
    if best_child.ndim != 1:
        raise ValueError("best_child must be 1D")
    if selected_roots.ndim != 1:
        raise ValueError("selected_roots must be 1D")

    best_child = best_child.contiguous().to(dtype=torch.long)
    selected_roots = selected_roots.contiguous().to(dtype=torch.long)

    if num_nodes is None:
        num_nodes = int(best_child.numel())
    if num_nodes != int(best_child.numel()):
        raise ValueError("num_nodes must match best_child.shape[0]")

    if selected_roots.numel() == 0:
        return selected_roots, selected_roots

    selected_track_labels, selected_nodes, selected_count = module.trace_selected_paths_cuda(
        best_child,
        selected_roots,
    )
    selected_count = int(selected_count.item())
    return (
        selected_track_labels[:selected_count].long(),
        selected_nodes[:selected_count].long(),
    )