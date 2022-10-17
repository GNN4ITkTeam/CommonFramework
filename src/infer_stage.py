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
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .utils import str_to_class

@click.command()
@click.option("--config_file", "-c", default="config.yaml", help="Path to the training config file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")

def main(config_file, verbose):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    # set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    infer(config_file)


def infer(config_file):

    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module = str_to_class(stage, model).infer(config)


if __name__ == "__main__":
    main()