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
            "acorn.cuda_ext._connected_components_cuda"
        )
    except ImportError as exc:
        _IMPORT_ERROR = exc
        return None

    return _NATIVE_MODULE


def connected_components_available():
    return _load_native_module() is not None


def _prepare_edge_tensors(src, dst, *, make_undirected):
    if not src.is_cuda or not dst.is_cuda:
        raise ValueError("connected_components_weak expects CUDA edge tensors")

    if src.ndim != 1 or dst.ndim != 1:
        raise ValueError("connected_components_weak expects 1D edge tensors")

    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")

    if make_undirected:
        original_src = src
        src = torch.cat([original_src, dst], dim=0)
        dst = torch.cat([dst, original_src], dim=0)

    if src.dtype != torch.int32:
        src = src.to(dtype=torch.int32)
        dst = dst.to(dtype=torch.int32)

    return src.contiguous(), dst.contiguous()


def connected_components_weak_int32(src, dst, num_nodes, *, make_undirected=True):
    module = _load_native_module()
    if module is None:
        raise RuntimeError(
            "The optional ACORN CUDA connected-components extension is not available. "
            "Build with ACORN_BUILD_CUDA_EXT=1 and --no-build-isolation to enable it."
        ) from _IMPORT_ERROR

    if src.numel() == 0:
        labels = torch.arange(num_nodes, device=src.device, dtype=torch.int32)
        return labels, num_nodes

    src, dst = _prepare_edge_tensors(src, dst, make_undirected=make_undirected)
    labels, num_components = module.connected_components_cuda(src, dst, int(num_nodes))
    return labels, int(num_components)


def connected_components_weak(src, dst, num_nodes, *, make_undirected=True):
    labels, num_components = connected_components_weak_int32(
        src,
        dst,
        num_nodes,
        make_undirected=make_undirected,
    )
    return labels.long(), int(num_components)


def process_components_cuda(src, dst, labels, large_component_mask, num_nodes, num_components):
    module = _load_native_module()
    if module is None:
        raise RuntimeError(
            "The optional ACORN CUDA connected-components extension is not available. "
            "Build with ACORN_BUILD_CUDA_EXT=1 and --no-build-isolation to enable it."
        ) from _IMPORT_ERROR

    src, dst = _prepare_edge_tensors(src, dst, make_undirected=False)

    if not labels.is_cuda or labels.ndim != 1:
        raise ValueError("process_components_cuda expects 1D CUDA labels")

    if labels.numel() != num_nodes:
        raise ValueError("labels length must equal num_nodes")

    if labels.dtype != torch.int32:
        labels = labels.to(dtype=torch.int32)

    if not large_component_mask.is_cuda or large_component_mask.ndim != 1:
        raise ValueError("process_components_cuda expects a 1D CUDA node mask")

    if large_component_mask.numel() != num_nodes:
        raise ValueError("large_component_mask length must equal num_nodes")

    if large_component_mask.dtype != torch.bool:
        large_component_mask = large_component_mask.to(dtype=torch.bool)

    return module.process_components_cuda(
        src,
        dst,
        labels.contiguous(),
        large_component_mask.contiguous(),
        int(num_nodes),
        int(num_components),
    )