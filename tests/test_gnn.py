import sys
sys.path.append("../")

import pytest

def test_model_load():
    from stages.GNN.gnn_stage import GNNStage

    config = {
        "model_name": "InteractionGNN"
    }

    model = GNNStage(config)

    assert model is not None
    assert model.model is not None 
