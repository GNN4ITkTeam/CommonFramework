def test_stage_load():
    """
    Test the model to ensure it is of the right format and loaded correctly.
    """
    from gnn4itk_cf.core.train_stage import train

    train("stages/edge_classifier/test_gnn_config.yaml")
