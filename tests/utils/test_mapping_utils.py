import pytest
import torch
from acorn.utils.mapping_utils import (
    get_variable_type,
    VariableType,
    map_nodes_to_edges,
    map_nodes_to_tracks,
    map_tensor_handler,
)


# Mock data for testing
@pytest.fixture
def mock_data():
    num_nodes = 10
    num_edges = 20
    num_track_edges = 10
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    hit_x = torch.rand((num_nodes, 3))
    # track_to_edge_map = torch.randint(0, num_nodes, (num_edges,))
    track_edges = torch.randint(0, num_nodes, (2, num_track_edges))

    return {
        "edge_index": edge_index,
        # "track_to_edge_map": track_to_edge_map,
        "hit_x": hit_x,
        "track_edges": track_edges,
    }


def test_infer_input_type(mock_data):
    input_type = get_variable_type("hit_x")
    assert input_type == VariableType.NODE_LIKE

    input_type = get_variable_type("edge_index")
    assert input_type == VariableType.EDGE_LIKE

    input_type = get_variable_type("track_edges")
    assert input_type == VariableType.TRACK_LIKE


def test_map_nodes_to_edges(mock_data):
    nodelike_input = mock_data["hit_x"]
    edge_index = mock_data["edge_index"]
    result = map_nodes_to_edges(nodelike_input, edge_index, aggr="mean")
    assert result.shape == (20, 3)

    result = map_tensor_handler(
        input_tensor=nodelike_input,
        output_type=VariableType.EDGE_LIKE,
        input_type=VariableType.NODE_LIKE,
        edge_index=edge_index,
        aggr="mean",
    )
    assert result.shape == (20, 3)


def test_map_nodes_to_tracks(mock_data):
    nodelike_input = mock_data["hit_x"]
    track_edges = mock_data["track_edges"]
    result = map_nodes_to_tracks(nodelike_input, track_edges, aggr="mean")
    assert result.shape == (10, 3)

    result = map_tensor_handler(
        input_tensor=nodelike_input,
        output_type=VariableType.TRACK_LIKE,
        input_type=VariableType.NODE_LIKE,
        track_edges=track_edges,
        aggr="mean",
    )
    assert result.shape == (10, 3)
