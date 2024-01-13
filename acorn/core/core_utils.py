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

import os
from pathlib import Path

import torch
import logging

try:
    import wandb
except ImportError:
    wandb = None
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from acorn import stages
from acorn.stages import *  # noqa
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger


def str_to_class(stage, model):
    """
    Convert a string to a class in the stages directory
    """

    return getattr(getattr(stages, stage), model)


def get_default_root_dir():
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(".", os.environ["SLURM_JOB_ID"])
    else:
        return None


def load_config_and_checkpoint(config_path, default_root_dir):
    # Check if there is a checkpoint to load
    checkpoint = (
        find_latest_checkpoint(default_root_dir)
        if default_root_dir is not None
        else None
    )
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        return (
            torch.load(checkpoint, map_location=torch.device("cpu"))[
                "hyper_parameters"
            ],
            checkpoint,
        )
    else:
        print("No checkpoint found, loading config from file")
        with open(config_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader), None


def find_latest_checkpoint(checkpoint_base, templates=None):
    if templates is None:
        templates = ["*.ckpt"]
    elif isinstance(templates, str):
        templates = [templates]
    checkpoint_paths = []
    for template in templates:
        checkpoint_paths = checkpoint_paths or [
            str(path) for path in Path(checkpoint_base).rglob(template)
        ]
    return max(checkpoint_paths, key=os.path.getctime) if checkpoint_paths else None


def get_trainer(config, default_root_dir):
    metric_to_monitor = (
        config["metric_to_monitor"] if "metric_to_monitor" in config else "val_loss"
    )
    metric_mode = config["metric_mode"] if "metric_mode" in config else "min"

    logging.info(f"Setting default root dir: {default_root_dir}")
    resume = "allow"

    job_id = (
        os.environ["SLURM_JOB_ID"]
        if "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
        else None
    )

    if (
        isinstance(default_root_dir, str)
        and find_latest_checkpoint(default_root_dir) is not None
    ):
        logging.info(
            f"Found checkpoint from a previous run in {default_root_dir}, resuming from"
            f" {find_latest_checkpoint(default_root_dir)}"
        )

    logging.info(f"Job ID: {job_id}, resume: {resume}")

    # handle wandb logging
    logger = (
        WandbLogger(
            project=config["project"],
            save_dir=config["stage_dir"],
            id=job_id,
            name=job_id,
            group=config.get("group"),
            resume=resume,
        )
        if wandb is not None and config.get("log_wandb", True)
        else CSVLogger(save_dir=config["stage_dir"])
    )

    filename_suffix = (
        str(logger.experiment.id)
        if (
            hasattr(logger, "experiment")
            and hasattr(logger.experiment, "id")
            and logger.experiment.id is not None
        )
        else ""
    )
    filename = "best-" + filename_suffix + "-{" + metric_to_monitor + ":5f}-{epoch}"
    accelerator = config.get("accelerator")
    devices = config.get("devices")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["stage_dir"], "artifacts"),
        filename=filename,
        monitor=metric_to_monitor,
        mode=metric_mode,
        save_top_k=config.get("save_top_k", 1),
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last-{filename_suffix}"

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
        precision=config.get("precision", 32),
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        default_root_dir=default_root_dir,
    )


def get_stage_module(
    config, stage_module_class, checkpoint_path=None, checkpoint_resume_dir=None
):
    # get a default_root_dr
    default_root_dir = get_default_root_dir()

    # if resume from a previous run that fails, allow to specify a checkpoint_resume_dir that must contain checkpoints from previous run
    # if checkpoint_resume_dir exists and contains a checkpoint, set as default_root_dir
    if checkpoint_resume_dir is not None:
        if not os.path.exists(checkpoint_resume_dir):
            raise Exception(
                f"Checkpoint resume directory {checkpoint_resume_dir} does not exist."
            )
        if not find_latest_checkpoint(checkpoint_resume_dir, "*.ckpt"):
            raise Exception(
                "No checkpoint found in checkpoint resume directory"
                f" {checkpoint_resume_dir}."
            )
        default_root_dir = checkpoint_resume_dir

    # if default_root_dir contains checkpoint, use latest checkpoint as starting point, ignore the input checkpoint_path
    if default_root_dir is not None and find_latest_checkpoint(
        default_root_dir, "*.ckpt"
    ):
        checkpoint_path = find_latest_checkpoint(default_root_dir, "*.ckpt")

    # Load a checkpoint if checkpoint_path is not None
    if checkpoint_path is not None:
        stage_module, config = load_module(checkpoint_path, stage_module_class)
    else:
        stage_module = stage_module_class(config)
    return stage_module, config, default_root_dir


def load_module(checkpoint_path, stage_module_class):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]
    stage_module = stage_module_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    return stage_module, config
