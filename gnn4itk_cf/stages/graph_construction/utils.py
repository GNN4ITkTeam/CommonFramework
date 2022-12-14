import torch
import logging

def get_weight_mask(event, edges, weight_conditions, true_edges=None, truth_map=None):

    graph_mask = torch.ones_like(edges[0], dtype=torch.bool)

    for condition_key, condition_val in weight_conditions.items():
        assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        print(graph_mask.shape, value_mask.shape, edges.shape)
        if true_edges is not None:
            print("True edges is not None")
            print(true_edges.shape, truth_map.shape)
        graph_mask = graph_mask * map_value_to_graph(event, value_mask, edges, true_edges, truth_map)

    return graph_mask

def build_signal_edges(event, weighting_config):
    signal_mask = torch.zeros_like(event.track_edges[0], dtype=torch.bool)
    for weight_spec in weighting_config:
        signal_mask = signal_mask | get_weight_mask(event, event.track_edges, weight_spec["conditions"])
    event.signal_track_edges = event.track_edges[:, signal_mask]


def handle_weighting(event, edges, truth, weighting_config, true_edges, truth_map):
    print(event, edges.shape, truth.shape, weighting_config, true_edges.shape, truth_map.shape)
    """
    Take the specification of the weighting and convert this into float values. The default is:
    - True edges have weight 1.0
    - Negative edges have weight 1.0

    The weighting_config can be used to change this behaviour. For example, we might up-weight target particles - that is edges that pass:
    - y == 1
    - primary == True
    - pt > 1 GeV
    - etc. As desired.

    We can also down-weight (i.e. mask) edges that are true, but not of interest. For example, we might mask:
    - y == 1
    - primary == False
    - pt < 1 GeV
    - etc. As desired.
    """

    # Set the default values, which will be overwritten if specified in the config
    weights = torch.zeros_like(edges[0], dtype=torch.float)
    weights[truth == 0] = 1.0

    for weight_spec in weighting_config:
        weight_val = weight_spec["weight"]
        weights[get_weight_mask(event, edges, weight_spec["conditions"], true_edges, truth_map)] = weight_val

    event.weights = weights

def handle_hard_node_cuts(event, hard_cuts_config):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """

    node_mask = torch.ones_like(event.x, dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        node_val_mask = torch.zeros_like(event.x, dtype=torch.bool)
        node_val_mask[event.track_edges[0]], node_val_mask[event.track_edges[1]] = value_mask, value_mask
        node_mask = node_mask * node_val_mask

    logging.info(f"Masking the following number of nodes with the HARD CUT: {node_mask.sum()} / {node_mask.shape[0]}")
    
    # TODO: Refactor the below to use the remap_from_mask function
    num_nodes = event.x.shape[0]
    for feature in event.keys:
        if isinstance(event[feature], torch.Tensor) and event[feature].shape[0] == num_nodes:
            event[feature] = event[feature][node_mask]

    num_tracks = event.track_edges.shape[1]
    track_mask = node_mask[event.track_edges].all(0)
    node_lookup = torch.cumsum(node_mask, dim=0) - 1
    for feature in event.keys:
        if isinstance(event[feature], torch.Tensor) and event[feature].shape[-1] == num_tracks:
            event[feature] = event[feature][..., track_mask]

    event.track_edges = node_lookup[event.track_edges]     

def handle_hard_cuts(event, hard_cuts_config):
    true_track_mask = torch.ones_like(event.truth_map, dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        true_track_mask = true_track_mask * value_mask

    graph_mask = torch.isin(event.edge_index, event.track_edges[:, true_track_mask]).all(0)
    remap_from_mask(event, graph_mask)

    for edge_key in ["edge_index", "y", "weight", "scores"]:
        if edge_key in event.keys:
            event[edge_key] = event[edge_key][..., graph_mask]

    num_track_edges = event.track_edges.shape[1]
    for track_feature in event.keys:
        if isinstance(event[track_feature], torch.Tensor) and (event[track_feature].shape[-1] == num_track_edges):
            event[track_feature] = event[track_feature][..., true_track_mask]

def map_value_to_graph(event, value_mask, edges=None, true_edges=None, truth_map=None):
    """
    Map the value mask to the graph. This is done by testing which dimension the value fits. 
    - If it is already equal to the graph size, nothing needs to be done
    - If it is equal to the track edges, it needs to be mapped to the graph edges
    - If it is equal to node list size, it needs to be mapped to the incoming/outgoing graph edges 
    """

    if edges is not None and value_mask.shape[0] == edges.shape[1]:
        return value_mask
    elif true_edges is not None and value_mask.shape[0] == true_edges.shape[1]:
        return map_tracks_to_graph(value_mask, edges, truth_map)
    elif value_mask.shape[0] == event.x.shape[0]:
        return map_nodes_to_graph(value_mask, edges)
    else:
        raise ValueError(f"Value mask has shape {value_mask.shape}, which is not compatible with the graph")

def map_tracks_to_graph(track_mask, edges, truth_map):

    graph_mask = torch.zeros_like(edges[0], dtype=torch.bool)
    graph_mask[truth_map[truth_map >= 0]] = track_mask[truth_map >= 0]

    return graph_mask

def map_nodes_to_graph(node_mask, edges):

    return node_mask[edges[0]]

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

def remap_from_mask(event, edge_mask):
    """ 
    Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
    """

    truth_map_to_edges = torch.ones(event.edge_index.shape[1], dtype=torch.long) * -1
    truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(event.truth_map.shape[0])[event.truth_map >= 0]
    truth_map_to_edges = truth_map_to_edges[edge_mask]

    new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
    new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(truth_map_to_edges.shape[0])[truth_map_to_edges >= 0]
    event.truth_map = new_map.to(event.truth_map.device)