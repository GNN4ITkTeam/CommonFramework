import os
import logging
from pathlib import Path

import torch
import glob
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from gnn4itk_cf import stages
from gnn4itk_cf.stages import *
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.overrides import LightningDistributedModule

try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    logging.info("Wandb found, using WandbLogger")
except ImportError:
    wandb = None
    logging.info("Wandb not found, using CSVLogger")
    from pytorch_lightning.loggers import CSVLogger

def str_to_class(stage, model):
    """
    Convert a string to a class in the stages directory
    """
    
    return getattr(getattr(stages, stage), model)

def get_default_root_dir():
    if "SLURM_JOB_ID" in os.environ: 
        return os.path.join(".", os.environ["SLURM_JOB_ID"])
    else:
        return "."

def load_config_and_checkpoint(config_path, default_root_dir):
    # Check if there is a checkpoint to load
    checkpoint = find_latest_checkpoint(default_root_dir) if default_root_dir is not None else None
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        return torch.load(checkpoint, map_location=torch.device("cpu"))["hyper_parameters"], checkpoint
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
        checkpoint_paths = checkpoint_paths or [str(path) for path in Path(checkpoint_base).rglob(template)]
    return max(checkpoint_paths, key=os.path.getctime) if checkpoint_paths else None

def get_trainer(config, default_root_dir):

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

    job_id = (
        os.environ["SLURM_JOB_ID"]
        if "SLURM_JOB_ID" in os.environ and "SLURM_JOB_QOS" in os.environ and "interactive" not in os.environ["SLURM_JOB_QOS"]
        else None
    )
    logger = (
        WandbLogger(project=config["project"], save_dir=config["stage_dir"], id=job_id)
        if wandb is not None
        else CSVLogger(save_dir=config["stage_dir"])
    )

    accelerator = "gpu" if torch.cuda.is_available() else None
    devices = config["gpus"] if "gpus" in config else 1

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
        strategy=CustomDDPPlugin(find_unused_parameters=False),
        default_root_dir=default_root_dir
    )


class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()