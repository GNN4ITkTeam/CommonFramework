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

import torch
import numpy as np
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.data import Data
from torch_scatter import scatter_max
from numba import njit, types
from numba.typed import Dict, List


def filter_graph(graph, score_name, threshold):
    mask = graph[score_name] > threshold
    edge_index = graph.edge_index[:, mask]
    edge_scores = graph[score_name][mask]
    transform = RemoveIsolatedNodes()
    new_graph = Data(
        edge_index=edge_index,
        edge_scores=edge_scores,
        hit_id=graph.hit_id,
        num_nodes=len(graph.hit_id),
        hit_r=graph.hit_r,
        hit_z=graph.hit_z,
    )
    new_graph = transform(new_graph)

    return new_graph


def process_components(graph, labels, large_component_labels):
    in_degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
    out_degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
    in_degrees.index_add_(
        0, graph.edge_index[1], torch.ones(graph.num_edges, dtype=torch.long)
    )
    out_degrees.index_add_(
        0, graph.edge_index[0], torch.ones(graph.num_edges, dtype=torch.long)
    )

    large_component_mask = torch.isin(labels, large_component_labels)
    small_component_mask = ~large_component_mask

    in_degrees_max = scatter_max(in_degrees, labels, dim=0)[0]
    out_degrees_max = scatter_max(out_degrees, labels, dim=0)[0]

    simple_path_components = (in_degrees_max <= 1) & (out_degrees_max <= 1)

    simple_path_mask = simple_path_components[labels]

    large_component_simple_path_mask = simple_path_mask & large_component_mask
    large_component_complex_path_mask = ~simple_path_mask & large_component_mask

    assert torch.all(
        large_component_simple_path_mask
        | large_component_complex_path_mask
        | small_component_mask
        == torch.ones_like(labels, dtype=torch.bool)
    ), "Categorization is not complete and mutually exclusive"

    subgraph_simple_paths = graph.subgraph(large_component_simple_path_mask)
    subgraph_complex_paths = graph.subgraph(large_component_complex_path_mask)

    return subgraph_simple_paths, subgraph_complex_paths


def labels_to_lists(simple_path_graph):
    labels = simple_path_graph.labels
    hit_ids = simple_path_graph.hit_id

    unique_labels, counts = torch.unique(labels, return_counts=True)
    mask = labels.unsqueeze(0) == unique_labels.unsqueeze(1)
    grouped_hit_ids = hit_ids.unsqueeze(0).expand(len(unique_labels), -1)[mask]
    result = torch.split(grouped_hit_ids, counts.tolist())
    result = [track.tolist() for track in result]

    return result


def get_simple_path(graph):
    from scipy.sparse.csgraph import connected_components
    from torch_geometric.utils import to_scipy_sparse_matrix

    adj_matrix = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)

    n_components, labels = connected_components(
        csgraph=adj_matrix, directed=True, connection="weak"
    )
    labels = torch.from_numpy(labels).long()
    graph.labels = labels

    unique_labels, counts = torch.unique(labels, return_counts=True)
    large_component_labels = unique_labels[counts >= 3]

    subgraph_simple_paths, subgraph_rest = process_components(
        graph, labels, large_component_labels
    )

    simple_path_lists = labels_to_lists(subgraph_simple_paths)

    return simple_path_lists, subgraph_rest


@njit
def topological_sort_numba(numba_edges):
    in_degree = Dict.empty(key_type=types.int64, value_type=types.int64)
    for node in numba_edges:
        if node not in in_degree:
            in_degree[node] = 0
        for neighbor in numba_edges[node]:
            if neighbor not in in_degree:
                in_degree[neighbor] = 1
            else:
                in_degree[neighbor] += 1

    queue = List()
    for node in in_degree:
        if in_degree[node] == 0:
            queue.append(node)

    result = List()
    while queue:
        node = queue.pop(0)
        result.append(node)

        if node in numba_edges:
            for neighbor in numba_edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    # TODO: Handle this later
    # if len(result) != len(numba_edges):
    #     return None  # Graph has a cycle

    return result


def topological_sort_graph(graph, numba_edges):
    sorted_hit_ids = topological_sort_numba(numba_edges)
    return sorted_hit_ids


def resolve_ambiguities(tracks, max_ambi_hits):
    # 1. Order the optimized tracks by their length, in descending order
    sorted_tracks = sorted(tracks, key=lambda x: len(x), reverse=True)

    # 2. Iterate through each track, keeping a dict of hit_ids, and remove any hit ids from tracks if they have already been used more than once
    used_hit_ids = {}
    resolved_tracks = []

    for track in sorted_tracks:
        updated_track = [
            hit_id for hit_id in track if used_hit_ids.get(hit_id, 0) < max_ambi_hits
        ]
        resolved_tracks.append(updated_track)
        for hit_id in updated_track:
            used_hit_ids[hit_id] = used_hit_ids.get(hit_id, 0) + 1

    return resolved_tracks


def walk_through(graph, score_name, th_min, th_add, allow_node_reuse):
    graph = max_add_cuts(graph, score_name, th_min, th_add)
    numba_edges = convert_pyg_graph_to_numba(graph, score_name)
    sorted_hit_ids = topological_sort_graph(graph, numba_edges=numba_edges)
    tracks = get_tracks(
        numba_edges,
        sorted_hit_ids,
        allow_node_reuse,
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

    subgraph = graph.edge_subgraph(final_mask)

    transform = RemoveIsolatedNodes()
    subgraph = transform(subgraph)

    return subgraph


@njit
def find_longest_path(complete_paths):
    longest_path = List.empty_list(types.int64)
    max_length = 0
    for path in complete_paths:
        if len(path) > max_length:
            max_length = len(path)
            longest_path.clear()
            longest_path.extend(path)
    return longest_path


@njit
def process_sorted_nodes(sorted_hit_ids, numba_edges, allow_node_reuse):
    tracks = List()
    used_nodes = Dict.empty(key_type=types.int64, value_type=types.boolean)
    for hit_id in sorted_hit_ids:
        if hit_id in used_nodes:
            continue

        complete_paths = find_paths(hit_id, numba_edges, used_nodes, allow_node_reuse)

        if complete_paths:
            longest_path = find_longest_path(complete_paths)
            if len(longest_path) > 1:
                tracks.append(longest_path)
                for node in longest_path:
                    used_nodes[node] = True

    return tracks


def get_tracks(numba_edges, sorted_hit_ids, allow_node_reuse):
    numba_sorted_hit_ids = List(sorted_hit_ids)
    tracks = process_sorted_nodes(numba_sorted_hit_ids, numba_edges, allow_node_reuse)
    return tracks


@njit
def find_paths(start_node, edges, used_nodes, allow_node_reuse):
    paths = List()
    paths.append([start_node])
    complete_paths = List()

    while len(paths) > 0:
        path = paths.pop(0)
        current_node = path[-1]

        if current_node not in edges:
            complete_paths.append(path)
            continue

        for neighbor in edges[current_node]:
            if not allow_node_reuse and neighbor in used_nodes:
                continue
            new_path = path.copy()
            new_path.append(neighbor)
            paths.append(new_path)

    return complete_paths


inner_dict_type = types.DictType(types.int64, types.float64)
outer_dict_type = types.DictType(types.int64, inner_dict_type)


@njit
def pyg_to_dict_numba(
    edge_index_src: np.ndarray,
    edge_index_dst: np.ndarray,
    edge_attr: np.ndarray,
    hit_ids: np.ndarray,
):
    edge_index_src = edge_index_src.astype(np.int64)
    edge_index_dst = edge_index_dst.astype(np.int64)
    edge_attr = edge_attr.astype(np.float64)
    hit_ids = hit_ids.astype(np.int64)

    numba_edges = Dict.empty(key_type=types.int64, value_type=inner_dict_type)

    for i in range(len(edge_index_src)):
        src = hit_ids[edge_index_src[i]]
        dst = hit_ids[edge_index_dst[i]]
        attr = edge_attr[i]

        if src not in numba_edges:
            numba_edges[src] = Dict.empty(
                key_type=types.int64, value_type=types.float64
            )

        numba_edges[src][dst] = attr

    return numba_edges


def convert_pyg_graph_to_numba(pyg_graph, score_name):
    edge_index = pyg_graph.edge_index.numpy()
    edge_index_src = edge_index[0]
    edge_index_dst = edge_index[1]
    edge_attr = pyg_graph[score_name].numpy()
    hit_ids = pyg_graph.hit_id.numpy()

    numba_edges = pyg_to_dict_numba(edge_index_src, edge_index_dst, edge_attr, hit_ids)

    return numba_edges
