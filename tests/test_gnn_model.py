import sys
sys.path.append("../")

import yaml
import pytest

def test_model_load():
    """
    Test the model to ensure it is of the right format and loaded correctly.
    """
    from stages.GNN.gnn_stage import GNNStage

    # load test_gnn_config.yaml
    with open("test_gnn_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = GNNStage(config)

    assert model is not None
    assert model.model is not None 

def test_data_load():
    """
    Test the data to ensure it is of the right format and loaded correctly.
    Runs X tests:
    1. Test a correct data load
    2. Test a data load with no testset
    3. Test a data load without enough events
    4. Missing directory
    """
    from stages.GNN.gnn_stage import GNNStage

    # load test_gnn_config.yaml
    with open("test_gnn_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Test 1
    config["data_split"] = [1, 1, 1]
    model = GNNStage(config)
    model.setup(stage="fit")

    assert model.trainset is not None
    assert model.valset is not None
    assert model.testset is not None

    # Test 2
    config["data_split"] = [1, 1, 0]
    model = GNNStage(config)
    model.setup(stage="fit")

    assert model.trainset is not None
    assert model.valset is not None
    assert model.testset is None

    # Test 3
    config["data_split"] = [100, 1, 1]
    model = GNNStage(config)

    pytest.raises(AssertionError, model.setup, stage="fit")

    # Test 4
    config["data_split"] = [1, 1, 1]
    config["input_dir"] = "a_missing_directory"
    model = GNNStage(config)

    pytest.raises(FileNotFoundError, model.setup, stage="fit")


def test_construct_weighting():
    """
    TODO
    """
    
    pass