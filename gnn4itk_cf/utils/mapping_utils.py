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
from torch_scatter import scatter
from torch_geometric.data import Data


def get_condition_lambda(condition_key, condition_val):
    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "not_in": lambda event: ~torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "within": lambda event: (condition_val[0] <= event[condition_key].float())
        & (event[condition_key].float() <= condition_val[1]),
        "not_within": lambda event: not (
            (condition_val[0] <= event[condition_key].float())
            & (event[condition_key].float() <= condition_val[1])
        ),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
        return lambda event: (condition_val[0] <= event[condition_key].float()) & (
            event[condition_key].float() <= condition_val[1]
        )
    elif isinstance(condition_val, list):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")


def map_tensor_handler(
    input_tensor: torch.Tensor,
    output_type: str,
    input_type: str = None,
    truth_map: torch.Tensor = None,
    edge_index: torch.Tensor = None,
    track_edges: torch.Tensor = None,
    num_nodes: int = None,
    num_edges: int = None,
    num_track_edges: int = None,
    aggr: str = None,
):
    """
    A general function to handle arbitrary maps of one tensor type to another
    Types are "node-like", "edge-like" and "track-like".
    - node-like: The input tensor is of the same size as the 
        number of nodes in the graph
    - edge-like: The input tensor is of the same size as the 
        number of edges in the graph, that is, the *constructed* graph
    - track-like: The input tensor is of the same size as the 
        number of true track edges in the event, that is, the *truth* graph

    To visualize:
                    (n)
                     ^
                    / \
      edge_to_node /   \ track_to_node
                  /     \
                 /       \
                /         \
               /           \
              /             \
node_to_edge /               \ node_to_track
            /                 \
           v     edge_to_track v
          (e) <-------------> (t)
            track_to_edge

    Args:
        input_tensor (torch.Tensor): The input tensor to be mapped
        output_type (str): The type of the output tensor. 
            One of "node-like", "edge-like" or "track-like"
        input_type (str, optional): The type of the input tensor. 
            One of "node-like", "edge-like" or "track-like". Defaults to None,
            and will try to infer the type from the input tensor, if num_nodes
            and/or num_edges are provided.
        truth_map (torch.Tensor, optional): The truth map tensor. 
            Defaults to None. Used for mappings to/from track-like tensors.
        num_nodes (int, optional): The number of nodes in the graph. 
            Defaults to None. Used for inferring the input type.
        num_edges (int, optional): The number of edges in the graph. 
            Defaults to None. Used for inferring the input type.
        num_track_edges (int, optional): The number of track edges in the graph 
            Defaults to None. Used for inferring the input type.
    """

    if num_track_edges is None and truth_map is not None:
        num_track_edges = truth_map.shape[0]
    if num_track_edges is None and track_edges is not None:
        num_track_edges = track_edges.shape[1]
    if num_edges is None and edge_index is not None:
        num_edges = edge_index.shape[1]
    if input_type is None:
        input_type, input_tensor = infer_input_type(
            input_tensor, num_nodes, num_edges, num_track_edges
        )
    if input_type == output_type:
        return input_tensor

    input_args = {
        "truth_map": truth_map,
        "edge_index": edge_index,
        "track_edges": track_edges,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_track_edges": num_track_edges,
        "aggr": aggr,
    }

    mapping_functions = {
        ("node-like", "edge-like"): map_nodes_to_edges,
        ("edge-like", "node-like"): map_edges_to_nodes,
        ("node-like", "track-like"): map_nodes_to_tracks,
        ("track-like", "node-like"): map_tracks_to_nodes,
        ("edge-like", "track-like"): map_edges_to_tracks,
        ("track-like", "edge-like"): map_tracks_to_edges,
    }
    if (input_type, output_type) not in mapping_functions:
        raise ValueError(f"Mapping from {input_type} to {output_type} not supported")

    return mapping_functions[(input_type, output_type)](input_tensor, **input_args)


# Returns string and tensor
def infer_input_type(
    input_tensor: torch.Tensor,
    num_nodes: int = None,
    num_edges: int = None,
    num_track_edges: int = None,
) -> (str, torch.Tensor):
    """
    Infers the type of the input tensor based on its shape and the provided number of nodes, edges, and track edges.

    Args:
        input_tensor (torch.Tensor): The tensor whose type needs to be inferred.
        num_nodes (int, optional): Number of nodes in the graph.
        num_edges (int, optional): Number of edges in the graph.
        num_track_edges (int, optional): Number of track edges in the graph.

    Returns:
        str: The inferred type of the input tensor. One of ["node-like", "edge-like", "track-like"].
    """

    NODE_LIKE = "node-like"
    EDGE_LIKE = "edge-like"
    TRACK_LIKE = "track-like"

    if num_nodes is not None and input_tensor.shape[0] == num_nodes:
        return NODE_LIKE, input_tensor
    elif num_edges is not None and num_edges in input_tensor.shape:
        return EDGE_LIKE, input_tensor
    elif num_track_edges is not None and num_track_edges in input_tensor.shape:
        return TRACK_LIKE, input_tensor
    elif num_track_edges is not None and num_track_edges // 2 in input_tensor.shape:
        return TRACK_LIKE, torch.cat([input_tensor, input_tensor], dim=0)
    else:
        raise ValueError("Unable to infer the type of the input tensor.")


def map_nodes_to_edges(
    nodelike_input: torch.Tensor, edge_index: torch.Tensor, aggr: str = None, **kwargs
):
    """
    Map a node-like tensor to an edge-like tensor. If the aggregation is None, this is simply done by sending node values to the edges, thus returning a tensor of shape (2, num_edges).
    If the aggregation is not None, the node values are aggregated to the edges, and the resulting tensor is of shape (num_edges,).
    """

    if aggr is None:
        return nodelike_input[edge_index]

    edgelike_tensor = nodelike_input[edge_index]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(edgelike_tensor, dim=0)


def map_edges_to_nodes(
    edgelike_input: torch.Tensor,
    edge_index: torch.Tensor,
    aggr: str = None,
    num_nodes: int = None,
    **kwargs,
):
    """
    Map an edge-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending edge values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the edge values are aggregated to the nodes at the destination node (edge_index[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(edge_index.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(
            num_nodes, dtype=edgelike_input.dtype, device=edgelike_input.device
        )
        nodelike_output[edge_index] = edgelike_input
        return nodelike_output

    return scatter(
        edgelike_input, edge_index[1], dim=0, dim_size=num_nodes, reduce=aggr
    )


def map_nodes_to_tracks(
    nodelike_input: torch.Tensor, track_edges: torch.Tensor, aggr: str = None, **kwargs
):
    """
    Map a node-like tensor to a track-like tensor. If the aggregation is None, this is simply done by sending node values to the tracks, thus returning a tensor of shape (2, num_track_edges).
    If the aggregation is not None, the node values are aggregated to the tracks, and the resulting tensor is of shape (num_track_edges,).
    """

    if aggr is None:
        return nodelike_input[track_edges]

    tracklike_tensor = nodelike_input[track_edges]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(tracklike_tensor, dim=0)


def map_tracks_to_nodes(
    tracklike_input: torch.Tensor,
    track_edges: torch.Tensor,
    aggr: str = None,
    num_nodes: int = None,
    **kwargs,
):
    """
    Map a track-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending track values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the track values are aggregated to the nodes at the destination node (track_edges[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(track_edges.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(
            num_nodes, dtype=tracklike_input.dtype, device=tracklike_input.device
        )
        nodelike_output[track_edges] = tracklike_input
        return nodelike_output

    return scatter(
        tracklike_input.repeat(2),
        torch.cat([track_edges[0], track_edges[1]]),
        dim=0,
        dim_size=num_nodes,
        reduce=aggr,
    )


def map_tracks_to_edges(
    tracklike_input: torch.Tensor,
    truth_map: torch.Tensor,
    num_edges: int = None,
    **kwargs,
):
    """
    Map an track-like tensor to a edge-like tensor. This is done by sending the track value through the truth map, where the truth map is >= 0. Note that where truth_map == -1,
    the true edge has not been constructed in the edge_index. In that case, the value is set to NaN.
    """

    if num_edges is None:
        num_edges = int(truth_map.max().item() + 1)
    edgelike_output = torch.zeros(
        num_edges, dtype=tracklike_input.dtype, device=tracklike_input.device
    )
    edgelike_output[truth_map[truth_map >= 0]] = tracklike_input[truth_map >= 0]
    edgelike_output[truth_map[truth_map == -1]] = float("nan")
    return edgelike_output


def map_edges_to_tracks(
    edgelike_input: torch.Tensor, truth_map: torch.Tensor, **kwargs
):
    """
    TODO: Implement this. I don't think it is a meaningful operation, but it is needed for completeness.
    """
    raise NotImplementedError(
        "This is not a meaningful operation, but it is needed for completeness"
    )


def remap_from_mask(event, edge_mask):
    """
    Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
    """

    truth_map_to_edges = torch.ones(edge_mask.shape[0], dtype=torch.long) * -1
    truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(
        event.truth_map.shape[0]
    )[event.truth_map >= 0]
    truth_map_to_edges = truth_map_to_edges[edge_mask]

    new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
    new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(
        truth_map_to_edges.shape[0]
    )[truth_map_to_edges >= 0]
    event.truth_map = new_map.to(event.truth_map.device)


def map_to_edges(event):
    """
    A function that takes any event and maps edges to track-like tensors
    """

    edge_mask = event.edge_mask
    truth_map = event.truth_map
    edge_index = event.edge_index
    edge_attr = event.edge_attr
    x = event.x

    # First, we need to remap the truth map to match the edge mask
    remap_from_mask(event, edge_mask)

    # Now, we can map the nodes to the edges
    x = map_nodes_to_edges(x, edge_index, aggr="mean")
    edge_attr = map_nodes_to_edges(edge_attr, edge_index, aggr="mean")

    # Now, we can map the tracks to the edges
    x = map_tracks_to_edges(x, truth_map)
    edge_attr = map_tracks_to_edges(edge_attr, truth_map)

    event.x = x
    event.edge_attr = edge_attr

    return event


def get_directed_prediction(event: Data, edge_pred, edge_index):
    """Apply score cut on edge_index of a totally bidirectional graph, i.e. every edge is repeated
    in the graph

    Args:
        event (_type_): _description_
        scores (_type_): _description_
        score_cut (_type_): _description_
        edge_index: must be sorted by distance
    """
    num_edges = edge_pred.shape[0]

    # sort edge index by the node id of source node
    inner_sorted_indices = torch.argsort(edge_index[1])

    # rearrange prediction and edge index by this order
    edge_pred = edge_pred[inner_sorted_indices]
    edge_index = edge_index[:, inner_sorted_indices]

    # sort edge index by the node id of destination node
    outter_sorted_indices = torch.argsort(edge_index[0])

    # rearrange
    edge_pred = edge_pred[outter_sorted_indices]
    print(
        edge_index.shape, edge_index.max(), edge_index[:, outter_sorted_indices].T.shape
    )
    event["edge_index"] = edge_index[:, outter_sorted_indices].T.view(2, -1, 2)[0].T

    # rearrange edge-level features as well
    for key in event.keys:
        if not isinstance(event[key], torch.Tensor) or not event[key].shape:
            continue
        if event[key].shape[0] == num_edges:
            event[key] = event[key][inner_sorted_indices][outter_sorted_indices].view(
                2, -1
            )[0]

    # do paired prediction
    paired_pred = edge_pred.view(-1, 2)
    for matching in ["loose", "tight"]:
        if matching == "loose":
            event.passing_edge_mask_loose = torch.any(paired_pred, dim=1)
        elif matching == "tight":
            event.passing_edge_mask_tight = torch.all(paired_pred, dim=1)
    # print(event.passing_edge_mask_loose, event.passing_edge_mask_tight)
