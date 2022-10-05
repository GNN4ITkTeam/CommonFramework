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
    infer(config_file)

# def infer(config_file):
#     # load config
#     with open(config_file, "r") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     # load stage
#     stage = config["stage"]

#     #  infer stage
#     stage_module = str_to_class(f"{stage}Stage").load_from_checkpoint(
#         os.path.join(config["input_dir"], "checkpoints", "last.ckpt")
#     )

#     stage_module.setup(stage="infer")
#     stage_module.build_infer_data()

def infer(config_file):

    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load stage
    stage = config["stage"]
    str_to_class(f"{stage}Stage").infer(config_file)


if __name__ == "__main__":
    main()