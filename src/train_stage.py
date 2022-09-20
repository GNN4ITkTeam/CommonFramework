"""
This script:
1. Loads a training config
2. Checks the stage to train
3. Loads the stage module
4. Trains the stage
5. Tests the output
"""

import sys
import os

import yaml
import click

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .utils import str_to_class

@click.command()
@click.option("--config_file", "-c", default="config.yaml", help="Path to the training config file")

def main(config_file):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    train(config_file)

def train(config_file):
    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load stage
    stage = config["stage"]
    stage_module = str_to_class(f"{stage}")(config)

    # setup stage
    stage_module.setup(stage="fit")

    checkpoint_callback = ModelCheckpoint(
        monitor="auc", mode="max", save_top_k=2, save_last=True
    )

    # train stage
    trainer = Trainer(
        gpus=config["gpus"],
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback]
    )
    
    # TODO:
    trainer.fit(stage_module)


if __name__ == "__main__":
    main()