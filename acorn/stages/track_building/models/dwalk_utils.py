# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_scatter import scatter_max

from acorn.cuda_ext import (
    connected_components_weak_int32,
    reverse_topological_dp,
    reverse_topological_dp_available,
    trace_selected_paths_available,
    trace_selected_paths_cuda,
)
from .cc_backend_utils import (
    compute_component_labels as compute_component_labels_from_backend,
    cudf_series_to_torch,
    should_use_custom_cuda,
    torch_tensor_to_cudf_series,
)

try:
    import cupy as cp
    import cudf
except ImportError:
    cp = None
    cudf = None


SMALL_GRAPH_DP_NODE_THRESHOLD = 2048
SMALL_GRAPH_DP_EDGE_THRESHOLD = 4096


def get_dp_metric_weights(graph, score_name, path_metrics):
    if path_metrics == "score_weighted_length":
        return graph[score_name]
    if path_metrics == "length":
        return torch.ones_like(graph[score_name])
    raise ValueError(f"Unsupported path_metrics '{path_metrics}'")


def get_min_root_score(path_metrics):
    if path_metrics == "score_weighted_length":
        return 0
    if path_metrics == "length":
        return 2
    raise ValueError(f"Unsupported path_metrics '{path_metrics}'")


def build_csr_graph_tensors(src, dst, weight, num_nodes):
    stable_order = torch.argsort(src, stable=True)
    sorted_src = src[stable_order]
    sorted_dst = dst[stable_order]
    sorted_weight = weight[stable_order]
    row_counts = torch.bincount(sorted_src, minlength=num_nodes)
    row_ptr = torch.empty(num_nodes + 1, dtype=torch.int32, device=src.device)
    row_ptr[0] = 0
    row_ptr[1:] = torch.cumsum(row_counts, dim=0).to(torch.int32)

    incoming_order = torch.argsort(dst, stable=True)
    incoming_dst = dst[incoming_order]
    incoming_src = src[incoming_order]
    incoming_counts = torch.bincount(incoming_dst, minlength=num_nodes)
    incoming_row_ptr = torch.empty(
        num_nodes + 1,
        dtype=torch.int32,
        device=src.device,
    )
    incoming_row_ptr[0] = 0
    incoming_row_ptr[1:] = torch.cumsum(incoming_counts, dim=0).to(torch.int32)

    return {
        "row_ptr": row_ptr,
        "col_idx": sorted_dst.to(torch.int32),
        "edge_weight": sorted_weight.to(torch.float32),
        "incoming_row_ptr": incoming_row_ptr,
        "incoming_col_idx": incoming_src.to(torch.int32),
        "sort_order": stable_order,
    }


def build_dp_graph_state(graph, score_name):
    topo_order = get_reverse_topological_order(graph).to(
        device=graph.edge_index.device,
        dtype=torch.int32,
    )
    csr_graph = build_csr_graph_tensors(
        graph.edge_index[0],
        graph.edge_index[1],
        graph[score_name],
        int(graph.num_nodes),
    )
    csr_graph["topo_order"] = topo_order
    csr_graph["num_nodes"] = int(graph.num_nodes)
    return csr_graph


def walk_through(
    graph,
    score_name,
    th_min,
    th_add,
    path_metrics="score_weighted_length",
    use_gpu=False,
    use_cudf=False,
    cc_backend="auto",
):
    graph = max_add_cuts(graph, score_name, th_min, th_add)
    dp_state = dp(graph, score_name, path_metrics)
    tracks = get_tracks(
        graph,
        score_name,
        dp_state,
        path_metrics=path_metrics,
        use_gpu=use_gpu,
        use_cudf=use_cudf,
        cc_backend=cc_backend,
    )
    return tracks


def max_add_cuts(graph, score_name, th_min, th_add):
    edge_scores = graph[score_name]
    edge_index = graph.edge_index

    mask_min = edge_scores > th_min
    mask_add = edge_scores > th_add

    out, argmax = scatter_max(edge_scores, edge_index[0], dim=0)
    mask_max = torch.zeros_like(mask_min, dtype=torch.bool)
    mask_max[argmax[out >= th_min]] = True

    final_mask = mask_max | mask_add

    in_, argmax = scatter_max(edge_scores, edge_index[1], dim=0)
    mask_max = torch.zeros_like(mask_min, dtype=torch.bool)
    mask_max[argmax[in_ >= th_min]] = True
    final_mask = (mask_max | mask_add) & final_mask

    subgraph = graph.edge_subgraph(final_mask)

    transform = RemoveIsolatedNodes()
    subgraph = transform(subgraph)

    return subgraph


def reverse_topological_dp_numpy(
    num_nodes,
    edge_starts,
    sorted_dst,
    sorted_weight,
    topo_order,
    active_nodes,
    in_degree,
):
    best_score = np.zeros(num_nodes, dtype=sorted_weight.dtype)
    best_child = np.full(num_nodes, -1, dtype=np.int64)
    source_mask = np.zeros(num_nodes, dtype=np.bool_)

    for order_idx in range(topo_order.shape[0] - 1, -1, -1):
        node = topo_order[order_idx]
        if not active_nodes[node]:
            continue

        best_value = 0.0
        best_next = -1
        for edge_idx in range(edge_starts[node], edge_starts[node + 1]):
            child = sorted_dst[edge_idx]
            if not active_nodes[child]:
                continue

            candidate = sorted_weight[edge_idx] + best_score[child]
            if candidate > best_value:
                best_value = candidate
                best_next = child

        best_score[node] = best_value
        best_child[node] = best_next

    for node in range(num_nodes):
        if active_nodes[node] and in_degree[node] == 0 and best_score[node] > 0:
            source_mask[node] = True

    return best_score, best_child, source_mask


def get_reverse_topological_order(graph):
    if hasattr(graph, "_dp_topological_order"):
        cached_order = graph._dp_topological_order
        if (
            int(cached_order.numel()) == int(graph.num_nodes)
            and cached_order.numel() == 0
            or (
                int(cached_order.numel()) == int(graph.num_nodes)
                and int(cached_order.min()) >= 0
                and int(cached_order.max()) < int(graph.num_nodes)
            )
        ):
            return cached_order

    if not hasattr(graph, "hit_r") or not hasattr(graph, "hit_z"):
        raise ValueError(
            "Reverse-topological DP requires graph.hit_r and graph.hit_z."
        )

    radial_key = graph.hit_r.square() + graph.hit_z.square()
    order = torch.argsort(radial_key, stable=True)

    graph._dp_topological_order = order.detach().cpu()
    return graph._dp_topological_order


def run_reverse_topological_dp(src, dst, weight, num_nodes, topo_order, active_nodes=None):
    use_small_graph_fallback = (
        src.device.type == "cuda"
        and num_nodes <= SMALL_GRAPH_DP_NODE_THRESHOLD
        and int(weight.numel()) <= SMALL_GRAPH_DP_EDGE_THRESHOLD
    )

    if (
        src.device.type == "cuda"
        and reverse_topological_dp_available()
        and not use_small_graph_fallback
    ):
        csr_graph = build_csr_graph_tensors(src, dst, weight, num_nodes)
        topo_order_cuda = torch.as_tensor(
            topo_order,
            device=src.device,
            dtype=torch.int32,
        )
        native_active_nodes = None
        if active_nodes is not None:
            native_active_nodes = active_nodes.to(device=src.device, dtype=torch.bool)
        best_score, best_child, source_mask = reverse_topological_dp(
            csr_graph["row_ptr"],
            csr_graph["col_idx"],
            csr_graph["edge_weight"],
            csr_graph["incoming_row_ptr"],
            csr_graph["incoming_col_idx"],
            topo_order_cuda,
            native_active_nodes,
        )
        return {
            "best_score": best_score.to(dtype=weight.dtype),
            "best_child": best_child.to(dtype=torch.long),
            "source_mask": source_mask,
        }

    src_cpu = src.detach().cpu().numpy().astype(np.int64, copy=False)
    dst_cpu = dst.detach().cpu().numpy().astype(np.int64, copy=False)
    weight_cpu = weight.detach().cpu().numpy()
    topo_order_cpu = np.asarray(topo_order, dtype=np.int64)

    if active_nodes is None:
        active_nodes_cpu = np.ones(num_nodes, dtype=np.bool_)
    else:
        active_nodes_cpu = active_nodes.detach().cpu().numpy().astype(np.bool_, copy=False)

    if src_cpu.size == 0:
        best_score = torch.zeros(num_nodes, dtype=weight.dtype, device=weight.device)
        best_child = torch.full((num_nodes,), -1, dtype=torch.long, device=src.device)
        source_mask = torch.zeros(num_nodes, dtype=torch.bool, device=src.device)
        return {
            "best_score": best_score,
            "best_child": best_child,
            "source_mask": source_mask,
        }

    sort_order = np.argsort(src_cpu, kind="stable")
    sorted_src = src_cpu[sort_order]
    sorted_dst = dst_cpu[sort_order]
    sorted_weight = weight_cpu[sort_order]
    counts = np.bincount(sorted_src, minlength=num_nodes)
    edge_starts = np.zeros(num_nodes + 1, dtype=np.int64)
    edge_starts[1:] = np.cumsum(counts)
    in_degree = np.bincount(dst_cpu, minlength=num_nodes)

    best_score_cpu, best_child_cpu, source_mask_cpu = reverse_topological_dp_numpy(
        num_nodes,
        edge_starts,
        sorted_dst,
        sorted_weight,
        topo_order_cpu,
        active_nodes_cpu,
        in_degree,
    )

    return {
        "best_score": torch.from_numpy(best_score_cpu).to(device=weight.device, dtype=weight.dtype),
        "best_child": torch.from_numpy(best_child_cpu).to(device=src.device, dtype=torch.long),
        "source_mask": torch.from_numpy(source_mask_cpu).to(device=src.device, dtype=torch.bool),
    }


def dp_from_tensors(
    src,
    dst,
    weight,
    num_nodes,
    topo_order=None,
    active_nodes=None,
    active_edges=None,
):
    if active_edges is not None:
        src = src[active_edges]
        dst = dst[active_edges]
        weight = weight[active_edges]

    if topo_order is None:
        topo_order = torch.arange(num_nodes, device=src.device)

    return run_reverse_topological_dp(
        src,
        dst,
        weight,
        num_nodes,
        topo_order,
        active_nodes=active_nodes,
    )


def dp(graph, score_name, path_metrics="score_weighted_length"):
    src, dst = graph.edge_index
    weight = get_dp_metric_weights(graph, score_name, path_metrics)
    num_nodes = int(graph.num_nodes)

    return dp_from_tensors(
        src,
        dst,
        weight,
        num_nodes,
        topo_order=get_reverse_topological_order(graph),
    )


def _collect_tracks(
    initial_state,
    get_hit_id,
    get_num_nodes,
    compute_components,
    select_roots,
    advance_structure,
    recompute_dp_state,
    use_cudf=False,
):
    tracks = []
    current_state = initial_state

    while True:
        hit_id = get_hit_id()
        num_nodes = get_num_nodes()

        component_labels, num_components = compute_components()

        selected_roots = select_roots(
            current_state,
            component_labels,
            num_components,
        )
        if selected_roots.numel() == 0:
            break

        selected_track_labels, selected_nodes = trace_selected_paths(
            current_state["best_child"],
            selected_roots,
            num_nodes,
        )

        tracks.extend(
            convert_paths_to_hit_ids(
                hit_id,
                selected_nodes,
                selected_track_labels,
                selected_roots,
                use_cudf=use_cudf,
            )
        )

        should_stop = advance_structure(selected_nodes, num_nodes)
        if should_stop:
            break

        current_state = recompute_dp_state()

    return tracks


def get_tracks(
    graph,
    score_name,
    dp_state,
    path_metrics="score_weighted_length",
    use_gpu=False,
    use_cudf=False,
    cc_backend="auto",
):
    current_graph = graph
    use_cached_components_once = hasattr(graph, "cached_component_labels")

    def get_hit_id():
        return current_graph.hit_id.long()

    def get_num_nodes():
        return int(current_graph.num_nodes)

    def get_device():
        return current_graph.edge_index.device

    def compute_components():
        nonlocal use_cached_components_once
        if use_cached_components_once:
            use_cached_components_once = False
            return current_graph.cached_component_labels, current_graph.cached_num_components
        return compute_component_labels(
            current_graph,
            use_gpu,
            cc_backend,
        )

    def select_roots(dp_state, component_labels, num_components):
        return select_component_roots(
            dp_state,
            component_labels,
            num_components,
            path_metrics=path_metrics,
        )

    def advance_structure(selected_nodes, num_nodes):
        nonlocal current_graph
        selected_node_mask = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=get_device(),
        )
        selected_node_mask[selected_nodes] = True
        updated_graph = update_active_edges(
            current_graph,
            selected_node_mask,
            score_name=score_name,
        )
        should_stop = updated_graph.num_nodes == current_graph.num_nodes
        current_graph = updated_graph
        return should_stop

    def recompute_dp_state():
        return dp(current_graph, score_name, path_metrics)

    return _collect_tracks(
        dp_state,
        get_hit_id,
        get_num_nodes,
        compute_components,
        select_roots,
        advance_structure,
        recompute_dp_state,
        use_cudf=use_cudf,
    )

def get_active_degrees(src, dst, num_nodes):
    if src.numel() > 0:
        in_degree = torch.bincount(dst, minlength=num_nodes)
        out_degree = torch.bincount(src, minlength=num_nodes)
    else:
        in_degree = torch.zeros(num_nodes, dtype=torch.long, device=src.device)
        out_degree = torch.zeros(num_nodes, dtype=torch.long, device=src.device)
    return in_degree, out_degree


def compute_component_labels(graph, use_gpu=False, cc_backend="auto"):
    num_nodes = int(graph.num_nodes)
    num_edges = int(graph.edge_index.shape[1])

    if use_gpu and cc_backend == "custom_cuda":
        if num_nodes <= 10000 and num_edges <= 10000:
            return compute_component_labels_from_backend(
                graph.edge_index,
                num_nodes,
                use_gpu=False,
                cc_backend="scipy",
            )

    if should_use_custom_cuda(
        graph.edge_index.device,
        use_gpu=use_gpu,
        cc_backend=cc_backend,
    ):
        return connected_components_weak_int32(
            graph.edge_index[0],
            graph.edge_index[1],
            num_nodes,
        )

    return compute_component_labels_from_backend(
        graph.edge_index,
        num_nodes,
        use_gpu=use_gpu,
        cc_backend=cc_backend,
    )

def select_component_roots(
    dp_state,
    component_labels,
    num_components,
    path_metrics="score_weighted_length",
):
    source_mask = dp_state["source_mask"]
    component_index = (
        component_labels
        if component_labels.dtype == torch.long
        else component_labels.long()
    )
    masked_scores = dp_state["best_score"].clone()
    masked_scores.masked_fill_(
        ~source_mask,
        torch.finfo(masked_scores.dtype).min,
    )
    component_scores, root_nodes = scatter_max(
        masked_scores,
        component_index,
        dim=0,
        dim_size=num_components,
    )
    valid_components = torch.nonzero(component_scores > get_min_root_score(path_metrics), as_tuple=False).flatten()
    return root_nodes[valid_components]


def trace_selected_paths_python(best_child, selected_roots, num_nodes):
    num_paths = selected_roots.shape[0]
    if num_paths == 0:
        return selected_roots, selected_roots

    current_nodes = selected_roots.clone()
    path_ids = torch.arange(num_paths, device=selected_roots.device, dtype=torch.long)
    selected_nodes = torch.empty(
        num_nodes,
        dtype=selected_roots.dtype,
        device=selected_roots.device,
    )
    selected_track_labels = torch.empty(
        num_nodes,
        dtype=torch.long,
        device=selected_roots.device,
    )
    num_selected = 0
    step = 0

    while current_nodes.numel() > 0 and step < num_nodes:
        next_num_selected = num_selected + current_nodes.numel()
        selected_nodes[num_selected:next_num_selected] = current_nodes
        selected_track_labels[num_selected:next_num_selected] = path_ids
        num_selected = next_num_selected

        next_nodes = best_child[current_nodes]
        active_mask = next_nodes != -1
        current_nodes = next_nodes[active_mask]
        path_ids = path_ids[active_mask]
        step += 1

    return selected_track_labels[:num_selected], selected_nodes[:num_selected]


def trace_selected_paths(best_child, selected_roots, num_nodes):
    if best_child.device.type == "cuda" and trace_selected_paths_available():
        return trace_selected_paths_cuda(best_child, selected_roots, num_nodes)

    return trace_selected_paths_python(best_child, selected_roots, num_nodes)


def convert_paths_to_hit_ids(
    hit_id,
    selected_nodes,
    selected_track_labels,
    selected_roots,
    use_cudf=False,
):
    num_tracks = int(selected_roots.numel())
    if num_tracks == 0:
        return ()

    if selected_nodes.numel() == 0:
        return ()

    selected_hit_ids = hit_id[selected_nodes]

    if can_use_cudf(selected_track_labels.device, use_cudf):
        return convert_paths_to_hit_ids_cudf(
            selected_hit_ids,
            selected_track_labels,
            num_tracks,
        )

    grouped_order = torch.argsort(selected_track_labels, stable=True)
    grouped_hit_ids = selected_hit_ids[grouped_order]
    grouped_track_labels = selected_track_labels[grouped_order]

    track_sizes = torch.bincount(grouped_track_labels, minlength=num_tracks)
    non_empty_track_sizes = track_sizes[track_sizes > 0].detach().cpu().tolist()
    grouped_tracks = torch.split(grouped_hit_ids, non_empty_track_sizes)

    return grouped_tracks


def convert_paths_to_hit_ids_cudf(selected_hit_ids, selected_track_labels, num_tracks):
    sorted_df = cudf.DataFrame(
        {
            "label": torch_tensor_to_cudf_series(selected_track_labels),
            "hit_id": torch_tensor_to_cudf_series(selected_hit_ids),
        }
    ).sort_values("label")
    sorted_labels = cudf_series_to_torch(sorted_df["label"]).long()
    _, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
    grouped_hit_ids = cudf_series_to_torch(sorted_df["hit_id"])
    non_empty_track_sizes = counts.tolist()
    return torch.split(grouped_hit_ids, non_empty_track_sizes)


def update_active_edges(graph, selected_node_mask, score_name=None):
    if selected_node_mask.numel() == 0:
        return graph

    if not selected_node_mask.any():
        return graph

    keep_node_mask = ~selected_node_mask
    keep_node_ids = torch.nonzero(keep_node_mask, as_tuple=False).flatten()
    num_nodes = int(keep_node_ids.numel())
    if num_nodes == int(graph.num_nodes):
        return graph

    src, dst = graph.edge_index
    keep_edge_mask = keep_node_mask[src] & keep_node_mask[dst]
    new_node_ids = torch.full(
        (selected_node_mask.numel(),),
        -1,
        dtype=torch.long,
        device=selected_node_mask.device,
    )
    new_node_ids[keep_node_ids] = torch.arange(num_nodes, device=selected_node_mask.device)

    updated_graph = Data(
        edge_index=torch.stack(
            (
                new_node_ids[src[keep_edge_mask]],
                new_node_ids[dst[keep_edge_mask]],
            ),
            dim=0,
        ),
        num_nodes=num_nodes,
    )

    for key in ("hit_id", "hit_r", "hit_z"):
        if hasattr(graph, key):
            updated_graph[key] = graph[key][keep_node_ids]

    if score_name is not None and hasattr(graph, score_name):
        updated_graph[score_name] = graph[score_name][keep_edge_mask]

    return updated_graph

def can_use_cudf(device, use_cudf):
    return (
        use_cudf
        and device.type == "cuda"
        and cp is not None
        and cudf is not None
    )
