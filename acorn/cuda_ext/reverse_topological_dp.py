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
            "acorn.cuda_ext._reverse_topological_dp_cuda"
        )
    except ImportError as exc:
        _IMPORT_ERROR = exc
        return None

    return _NATIVE_MODULE


def reverse_topological_dp_available():
    return _load_native_module() is not None


def reverse_topological_dp(
    row_ptr,
    col_idx,
    edge_weight,
    incoming_row_ptr,
    incoming_col_idx,
    topo_order,
    active_nodes=None,
):
    module = _load_native_module()
    if module is None:
        raise RuntimeError(
            "The optional ACORN CUDA reverse-topological-DP extension is not available. "
            "Build with ACORN_BUILD_CUDA_EXT=1 and --no-build-isolation to enable it."
        ) from _IMPORT_ERROR

    tensors = {
        "row_ptr": row_ptr,
        "col_idx": col_idx,
        "edge_weight": edge_weight,
        "incoming_row_ptr": incoming_row_ptr,
        "incoming_col_idx": incoming_col_idx,
        "topo_order": topo_order,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1D")

    row_ptr = row_ptr.contiguous().to(dtype=torch.int32)
    col_idx = col_idx.contiguous().to(dtype=torch.int32)
    edge_weight = edge_weight.contiguous().to(dtype=torch.float32)
    incoming_row_ptr = incoming_row_ptr.contiguous().to(dtype=torch.int32)
    incoming_col_idx = incoming_col_idx.contiguous().to(dtype=torch.int32)
    topo_order = topo_order.contiguous().to(dtype=torch.int32)

    if active_nodes is None:
        active_nodes = torch.ones(
            topo_order.shape[0],
            device=topo_order.device,
            dtype=torch.bool,
        )
    else:
        if not active_nodes.is_cuda:
            raise ValueError("active_nodes must be a CUDA tensor")
        if active_nodes.ndim != 1:
            raise ValueError("active_nodes must be 1D")
        active_nodes = active_nodes.contiguous().to(dtype=torch.bool)

    best_score, best_child, source_mask = module.reverse_topological_dp_cuda(
        row_ptr,
        col_idx,
        edge_weight,
        incoming_row_ptr,
        incoming_col_idx,
        topo_order,
        active_nodes,
    )
    return best_score, best_child.long(), source_mask.bool()