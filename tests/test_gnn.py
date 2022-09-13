import sys
sys.path.append("../")

import yaml
import pytest

def test_model_load():
    from stages.GNN.gnn_stage import GNNStage

    # load test_gnn_config.yaml
    with open("test_gnn_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = GNNStage(config)

    assert model is not None
    assert model.model is not None 