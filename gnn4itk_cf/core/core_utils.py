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

try:
    import wandb
except ImportError:
    wandb = None
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from gnn4itk_cf import stages
from gnn4itk_cf.stages import *  # noqa
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["stage_dir"], "artifacts"),
        filename="best",
        monitor=metric_to_monitor,
        mode=metric_mode,
        save_top_k=1,
        save_last=True,
    )

    job_id = (
        os.environ["SLURM_JOB_ID"]
        if "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
        else None
    )

    logger = (
        WandbLogger(project=config["project"], save_dir=config["stage_dir"], id=job_id)
        if wandb is not None and config.get("log_wandb", True)
        else CSVLogger(save_dir=config["stage_dir"])
    )

    gpus = config.get("gpus", 0)
    accelerator = "gpu" if gpus else None
    devices = gpus or None

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        default_root_dir=default_root_dir,
    )


def get_stage_module(config, stage_module_class, checkpoint_path=None):
    default_root_dir = get_default_root_dir()
    # First check if we need to load a checkpoint
    if checkpoint_path is not None:
        stage_module, config = load_module(checkpoint_path, stage_module_class)
    elif default_root_dir is not None and find_latest_checkpoint(
        default_root_dir, "*.ckpt"
    ):
        checkpoint_path = find_latest_checkpoint(default_root_dir, "*.ckpt")
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
