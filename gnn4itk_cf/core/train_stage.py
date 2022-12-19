"""
This script:
1. Loads a training config
2. Checks the stage to train
3. Loads the stage module
4. Trains the stage
5. Tests the output
"""

import logging
import os

import yaml
import click
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule

from gnn4itk_cf.utils import str_to_class

try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    logging.info("Wandb found, using WandbLogger")
except ImportError:
    wandb = None
    logging.info("Wandb not found, using CSVLogger")
    from pytorch_lightning.loggers import CSVLogger

@click.command()
@click.argument("config_file")

def main(config_file):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    train(config_file)

def train(config_file):
    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module = str_to_class(stage, model)(config)

    # setup stage
    stage_module.setup(stage="fit")

    # run training, depending on whether we are using a Lightning trainable model or not
    if isinstance(stage_module, LightningModule):
        lightning_train(config, stage_module)
    else:
        stage_module.train()


def lightning_train(config, stage_module):
        # setup logger
    logger = (
        WandbLogger(save_dir=config["stage_dir"], project=config["project"])
        if wandb is not None
        else CSVLogger(save_dir=config["stage_dir"])
    )
    metric_to_monitor = config["metric_to_monitor"] if "metric_to_monitor" in config else "val_loss"
    metric_mode = config["metric_mode"] if "metric_mode" in config else "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["stage_dir"], "artifacts"),
        filename='best',
        monitor=metric_to_monitor, 
        mode=metric_mode, 
        save_top_k=1, 
        save_last=True
    )

    accelerator = "gpu" if torch.cuda.is_available() else None
    devices = config["gpus"] if "gpus" in config else 1

    # train stage
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(stage_module)


if __name__ == "__main__":
    main()