# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script:
1. Loads a training config
2. Checks the stage to train
3. Loads the stage module
4. Trains the stage
5. Tests the output
"""

import sys

import yaml
import click
import logging

from pytorch_lightning import LightningModule
import torch

from .core_utils import str_to_class, find_latest_checkpoint


@click.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
@click.option(
    "--dataset",
    "-d",
    default="valset",
    type=click.Choice(["trainset", "valset", "testset"], case_sensitive=True),
)
def main(config_file, verbose, checkpoint, dataset):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """

    evaluate(config_file, verbose, checkpoint, dataset)


def evaluate(config_file, verbose=None, checkpoint=None, dataset="valset"):
    # set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module = str_to_class(stage, model)
    config["dataset"] = dataset

    if issubclass(stage_module, LightningModule):
        checkpoint_path = (
            find_latest_checkpoint(
                config["stage_dir"], templates=["best*.ckpt", "*.ckpt"]
            )
            if checkpoint is None
            else checkpoint
        )
        if not checkpoint_path:
            print("No checkpoint found")
            sys.exit(1)
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint_config = torch.load(
            checkpoint_path, map_location=torch.device("cpu")
        )["hyper_parameters"]
        config = {**checkpoint_config, **config}

    stage_module.evaluate(config)


if __name__ == "__main__":
    main()
