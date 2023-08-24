import pytest
import torch
from gnn4itk_cf.utils.mapping_utils import (
    infer_input_type,
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
    x = torch.rand((num_nodes, 3))
    truth_map = torch.randint(0, num_nodes, (num_edges,))
    track_edges = torch.randint(0, num_nodes, (2, num_track_edges))

    return {
        "edge_index": edge_index,
        "truth_map": truth_map,
        "x": x,
        "track_edges": track_edges,
    }


def test_infer_input_type(mock_data):
    input_tensor = mock_data["x"]
    input_type, _ = infer_input_type(
        input_tensor, num_nodes=10, num_edges=20, num_track_edges=10
    )
    assert input_type == "node-like"

    input_tensor = mock_data["truth_map"]
    input_type, _ = infer_input_type(
        input_tensor, num_nodes=10, num_edges=20, num_track_edges=10
    )
    assert input_type == "edge-like"

    input_tensor = mock_data["track_edges"]
    input_type, _ = infer_input_type(
        input_tensor, num_nodes=10, num_edges=20, num_track_edges=10
    )
    assert input_type == "track-like"


def test_map_nodes_to_edges(mock_data):
    nodelike_input = mock_data["x"]
    edge_index = mock_data["edge_index"]
    result = map_nodes_to_edges(nodelike_input, edge_index, aggr="mean")
    assert result.shape == (20, 3)

    result = map_tensor_handler(
        input_tensor=nodelike_input,
        output_type="edge-like",
        input_type="node-like",
        edge_index=edge_index,
        aggr="mean",
    )
    assert result.shape == (20, 3)


def test_map_nodes_to_tracks(mock_data):
    nodelike_input = mock_data["x"]
    track_edges = mock_data["track_edges"]
    result = map_nodes_to_tracks(nodelike_input, track_edges, aggr="mean")
    assert result.shape == (10, 3)

    result = map_tensor_handler(
        input_tensor=nodelike_input,
        output_type="track-like",
        input_type="node-like",
        track_edges=track_edges,
        aggr="mean",
    )
    assert result.shape == (10, 3)
