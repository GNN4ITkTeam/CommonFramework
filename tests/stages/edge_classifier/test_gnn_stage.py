def test_stage_load():
    """
    Test the model to ensure it is of the right format and loaded correctly.
    """
    from acorn.core.train_stage import train

    train("stages/edge_classifier/test_gnn_config.yaml")

    train("stages/edge_classifier/test_ignn2_config.yaml")


def test_gnn_infer():
    import os

    import yaml

    from acorn.core.infer_stage import infer

    with open("stages/edge_classifier/test_ignn2_config.yaml", "r") as f:
        config = yaml.full_load(f)

    infer("stages/edge_classifier/test_ignn2_config.yaml", verbose=True)

    assert "trainset" in os.listdir(config["stage_dir"]) and (
        len(os.listdir(os.path.join(config["stage_dir"], "trainset")))
        == config["data_split"][0]
    )

    assert "valset" in os.listdir(config["stage_dir"]) and (
        len(os.listdir(os.path.join(config["stage_dir"], "valset")))
        == config["data_split"][1]
    )

    assert "testset" in os.listdir(config["stage_dir"]) and (
        len(os.listdir(os.path.join(config["stage_dir"], "testset")))
        == config["data_split"][2]
    )
