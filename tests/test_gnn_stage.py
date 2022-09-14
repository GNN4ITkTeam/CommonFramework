import sys
sys.path.append("../")

import yaml
import pytest

def test_stage_load():
    """
    Test the model to ensure it is of the right format and loaded correctly.
    """
    from src.train_stage import train

    train("test_gnn_config.yaml")

    



