import torch
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_scipy_sparse_matrix

from acorn.cuda_ext import (
    connected_components_available,
    connected_components_weak,
)

try:
    import cupy as cp
    import cudf
    import cugraph
except ImportError:
    cp = None
    cudf = None
    cugraph = None


VALID_CC_BACKENDS = {"auto", "scipy", "cugraph", "custom_cuda"}


def normalize_cc_backend(cc_backend):
    backend = (cc_backend or "auto").lower()
    if backend not in VALID_CC_BACKENDS:
        raise ValueError(
            f"Unsupported connected components backend '{cc_backend}'. "
            "Expected one of: auto, scipy, cugraph, custom_cuda."
        )
    return backend


def can_use_gpu_cc(device, use_gpu):
    return (
        use_gpu
        and device.type == "cuda"
        and cp is not None
        and cudf is not None
        and cugraph is not None
    )


def can_use_custom_cuda_cc(device, use_gpu):
    return use_gpu and device.type == "cuda" and connected_components_available()


def torch_tensor_to_cupy(tensor):
    tensor = tensor.detach().contiguous()
    if tensor.device.type == "cuda":
        return cp.from_dlpack(tensor)
    return cp.asarray(tensor.cpu().numpy())


def torch_tensor_to_cudf_series(tensor):
    tensor = tensor.detach().contiguous()
    if tensor.device.type == "cuda":
        return cudf.from_dlpack(tensor.__dlpack__())
    return cudf.Series(tensor.cpu().numpy())


def cudf_series_to_torch(series):
    return torch.from_dlpack(series.to_dlpack())


def should_use_cugraph(device, use_gpu=False, cc_backend="auto"):
    backend = normalize_cc_backend(cc_backend)
    if backend == "scipy":
        return False
    if backend == "cugraph":
        if not can_use_gpu_cc(device, use_gpu):
            raise RuntimeError(
                "cc_backend='cugraph' requires use_gpu=True, a CUDA device, and "
                "cupy/cudf/cugraph to be installed."
            )
        return True
    return can_use_gpu_cc(device, use_gpu)


def should_use_custom_cuda(device, use_gpu=False, cc_backend="auto"):
    backend = normalize_cc_backend(cc_backend)
    if backend == "custom_cuda":
        if not can_use_custom_cuda_cc(device, use_gpu):
            raise RuntimeError(
                "cc_backend='custom_cuda' requires use_gpu=True, a CUDA device, "
                "and a built ACORN CUDA connected-components extension."
            )
        return True
    return False


def get_component_labels_scipy(edge_index, num_nodes, output_device=None):
    cpu_edge_index = edge_index.cpu() if edge_index.device.type != "cpu" else edge_index
    adjacency = to_scipy_sparse_matrix(cpu_edge_index, num_nodes=num_nodes)
    num_components, labels = connected_components(
        csgraph=adjacency,
        directed=True,
        connection="weak",
    )
    labels = torch.from_numpy(labels).long()
    if output_device is not None:
        labels = labels.to(output_device)
    return labels, num_components


def get_component_labels_cugraph(active_src, active_dst, num_nodes):
    undirected_src = torch.cat([active_src, active_dst])
    undirected_dst = torch.cat([active_dst, active_src])

    edge_df = cudf.DataFrame(
        {
            "src": torch_tensor_to_cudf_series(undirected_src),
            "dst": torch_tensor_to_cudf_series(undirected_dst),
        }
    )
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(
        edge_df,
        source="src",
        destination="dst",
        renumber=False,
    )

    label_df = cugraph.connected_components(graph)

    labels = torch.full((num_nodes,), -1, device=active_src.device, dtype=torch.long)
    vertices = cudf_series_to_torch(label_df["vertex"])
    components = cudf_series_to_torch(label_df["labels"])
    labels[vertices.long()] = components.long()
    next_component = int(components.max().item()) + 1 if components.numel() > 0 else 0
    isolated_mask = labels < 0
    num_isolated = int(isolated_mask.sum().item())
    if num_isolated:
        labels[isolated_mask] = torch.arange(
            next_component,
            next_component + num_isolated,
            device=active_src.device,
            dtype=torch.long,
        )
    num_components = next_component + num_isolated
    return labels, num_components


def get_component_labels_custom_cuda(active_src, active_dst, num_nodes):
    return connected_components_weak(active_src, active_dst, num_nodes)


def compute_component_labels_from_tensors(
    src,
    dst,
    num_nodes,
    use_gpu=False,
    cc_backend="auto",
):
    if src.numel() == 0:
        labels = torch.arange(num_nodes, device=src.device, dtype=torch.long)
        return labels, num_nodes

    if should_use_custom_cuda(src.device, use_gpu=use_gpu, cc_backend=cc_backend):
        return get_component_labels_custom_cuda(src, dst, num_nodes)

    if should_use_cugraph(src.device, use_gpu=use_gpu, cc_backend=cc_backend):
        return get_component_labels_cugraph(src, dst, num_nodes)

    edge_index = torch.stack([src, dst], dim=0)
    return get_component_labels_scipy(
        edge_index,
        num_nodes,
        output_device=src.device,
    )


def compute_component_labels(edge_index, num_nodes, use_gpu=False, cc_backend="auto"):
    return compute_component_labels_from_tensors(
        edge_index[0],
        edge_index[1],
        num_nodes,
        use_gpu=use_gpu,
        cc_backend=cc_backend,
    )
