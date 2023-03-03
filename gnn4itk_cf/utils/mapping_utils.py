# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch_scatter import scatter

def get_condition_lambda(condition_key, condition_val):

    # Refactor the above switch case into a dictionary
    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: torch.isin(event[condition_key], torch.tensor(condition_val[1])),
        "not_in": lambda event: ~torch.isin(event[condition_key], torch.tensor(condition_val[1])),
        "within": lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1]),
        "not_within": lambda event: not ((condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
        return lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])
    elif isinstance(condition_val, list):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")

def map_tensor_handler(input_tensor: torch.Tensor, 
                       output_type: str, 
                       input_type: str = None, 
                       truth_map: torch.Tensor = None, 
                       edge_index: torch.Tensor = None,
                       track_edges: torch.Tensor = None,
                       num_nodes: int = None, 
                       num_edges: int = None, 
                       num_track_edges: int = None,
                       aggr: str = None):
    """
    A general function to handle arbitrary maps of one tensor type to another. Types are "node-like", "edge-like" and "track-like".
    - Node-like: The input tensor is of the same size as the number of nodes in the graph
    - Edge-like: The input tensor is of the same size as the number of edges in the graph, that is, the *constructed* graph
    - Track-like: The input tensor is of the same size as the number of true track edges in the event, that is, the *truth* graph

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
           |                   | 
           v     edge_to_track v
          (e) <-------------> (t)
            track_to_edge

    Args:
        input_tensor (torch.Tensor): The input tensor to be mapped
        output_type (str): The type of the output tensor. One of "node-like", "edge-like" or "track-like"
        input_type (str, optional): The type of the input tensor. One of "node-like", "edge-like" or "track-like". Defaults to None, and will try to infer the type from the input tensor, if num_nodes and/or num_edges are provided.
        truth_map (torch.Tensor, optional): The truth map tensor. Defaults to None. Used for mappings to/from track-like tensors.
        num_nodes (int, optional): The number of nodes in the graph. Defaults to None. Used for inferring the input type.
        num_edges (int, optional): The number of edges in the graph. Defaults to None. Used for inferring the input type.
        num_track_edges (int, optional): The number of track edges in the graph. Defaults to None. Used for inferring the input type.
    """

    # Refactor the above switch case into a dictionary
    mapping_dict = {
        ("node-like", "edge-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_nodes_to_edges(input_tensor, edge_index, aggr),
        ("edge-like", "node-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_edges_to_nodes(input_tensor, edge_index, aggr, num_nodes),
        ("node-like", "track-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_nodes_to_tracks(input_tensor, track_edges, aggr),
        ("track-like", "node-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_tracks_to_nodes(input_tensor, track_edges, aggr, num_nodes),
        ("edge-like", "track-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_edges_to_tracks(input_tensor, truth_map),
        ("track-like", "edge-like"): lambda input_tensor, truth_map, edge_index, track_index, num_nodes, num_edges, num_track_edges, aggr: map_tracks_to_edges(input_tensor, truth_map, num_edges),
    }

    if num_track_edges is None and truth_map is not None:
        num_track_edges = truth_map.shape[0]
    if num_edges is None and edge_index is not None:
        num_edges = edge_index.shape[1]
    if input_type is None:
        input_type = infer_input_type(input_tensor, num_nodes, num_edges, num_track_edges)

    if input_type == output_type:
        return input_tensor
    elif (input_type, output_type) in mapping_dict:
        return mapping_dict[(input_type, output_type)](input_tensor, truth_map, edge_index, track_edges, num_nodes, num_edges, num_track_edges, aggr)
    else:
        raise ValueError(f"Mapping from {input_type} to {output_type} not supported")

def infer_input_type(input_tensor: torch.Tensor, num_nodes: int = None, num_edges: int = None, num_track_edges: int = None):
    """
    Tries to infer the input type from the input tensor and the number of nodes, edges and track-edges in the graph.
    If the input tensor cannot be matched to any of the provided types, it is assumed to be node-like.
    """

    if num_nodes is not None and input_tensor.shape[0] == num_nodes:
        return "node-like"
    elif num_edges is not None and num_edges in input_tensor.shape:
        return "edge-like"
    elif num_track_edges is not None and num_track_edges in input_tensor.shape:
        return "track-like"
    else:
        return "node-like"

def map_nodes_to_edges(nodelike_input: torch.Tensor, edge_index: torch.Tensor, aggr: str = None):
    """
    Map a node-like tensor to an edge-like tensor. If the aggregation is None, this is simply done by sending node values to the edges, thus returning a tensor of shape (2, num_edges).
    If the aggregation is not None, the node values are aggregated to the edges, and the resulting tensor is of shape (num_edges,).
    """

    if aggr is None:
        return nodelike_input[edge_index]
    
    edgelike_tensor = nodelike_input[edge_index]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(edgelike_tensor, dim=0)
    
def map_edges_to_nodes(edgelike_input: torch.Tensor, edge_index: torch.Tensor, aggr: str = None, num_nodes: int = None):
    """
    Map an edge-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending edge values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the edge values are aggregated to the nodes at the destination node (edge_index[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(edge_index.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(num_nodes, dtype=edgelike_input.dtype, device=edgelike_input.device)
        nodelike_output[edge_index] = edgelike_input
        return nodelike_output
    
    return scatter(edgelike_input, edge_index[1], dim=0, dim_size=num_nodes, reduce=aggr)

def map_nodes_to_tracks(nodelike_input: torch.Tensor, track_edges: torch.Tensor, aggr: str = None):
    """
    Map a node-like tensor to a track-like tensor. If the aggregation is None, this is simply done by sending node values to the tracks, thus returning a tensor of shape (2, num_track_edges).
    If the aggregation is not None, the node values are aggregated to the tracks, and the resulting tensor is of shape (num_track_edges,).
    """
    
    if aggr is None:
        return nodelike_input[track_edges]
    
    tracklike_tensor = nodelike_input[track_edges]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(tracklike_tensor, dim=0)

def map_tracks_to_nodes(tracklike_input: torch.Tensor, track_edges: torch.Tensor, aggr: str = None, num_nodes: int = None):
    """
    Map a track-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending track values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the track values are aggregated to the nodes at the destination node (track_edges[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(track_edges.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(num_nodes, dtype=tracklike_input.dtype, device=tracklike_input.device)
        nodelike_output[track_edges] = tracklike_input
        return nodelike_output
    
    return scatter(tracklike_input, track_edges[1], dim=0, dim_size=num_nodes, reduce=aggr)
    
def map_tracks_to_edges(tracklike_input: torch.Tensor, truth_map: torch.Tensor, num_edges: int = None):
    """
    Map an track-like tensor to a edge-like tensor. This is done by sending the track value through the truth map, where the truth map is >= 0. Note that where truth_map == -1,
    the true edge has not been constructed in the edge_index. In that case, the value is set to NaN.
    """

    if num_edges is None:
        num_edges = int(truth_map.max().item() + 1)
    edgelike_output = torch.zeros(num_edges, dtype=tracklike_input.dtype, device=tracklike_input.device)
    edgelike_output[truth_map[truth_map >= 0]] = tracklike_input[truth_map >= 0]
    edgelike_output[truth_map[truth_map == -1]] = float("nan")
    return edgelike_output

def map_edges_to_tracks(edgelike_input: torch.Tensor, truth_map: torch.Tensor):
    """
    TODO: Implement this. I don't think it is a meaningful operation, but it is needed for completeness.
    """
    raise NotImplementedError("This is not a meaningful operation, but it is needed for completeness")

def remap_from_mask(event, edge_mask):
    """ 
    Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
    """

    # print(event, edge_mask, edge_mask.sum())
    truth_map_to_edges = torch.ones(edge_mask.shape[0], dtype=torch.long) * -1
    truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(event.truth_map.shape[0])[event.truth_map >= 0]
    truth_map_to_edges = truth_map_to_edges[edge_mask]

    new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
    new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(truth_map_to_edges.shape[0])[truth_map_to_edges >= 0]
    event.truth_map = new_map.to(event.truth_map.device)