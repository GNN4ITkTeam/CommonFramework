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
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)

from acorn import stages
from acorn.core.throughput import ThroughputCallback
from acorn.utils.ml_utils import PlateauWarmupCallback
from acorn.stages import *  # noqa
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger


class InferenceTimingCallback(Callback):
    """Measure and log the wall-clock time of the validation and test forward pass.

    Times only the model forward — excludes data loading, loss computation,
    and metric logging.  Results are logged as ``val_infer_ms_mean`` /
    ``test_infer_ms_mean`` and ``*_ms_per_edge`` via Lightning's standard
    ``log`` mechanism (visible in CSV / WandB).

    Activated by adding ``time_inference: true`` to the training config.

    Uses ``torch.cuda.Event`` on GPU for accurate kernel-level timing, and
    ``time.perf_counter`` on CPU.
    """

    def __init__(self):
        self._val_times   = []
        self._val_edges   = []
        self._test_times  = []
        self._test_edges  = []
        self._start_event = None
        self._end_event   = None
        self._t0          = None

    # ── shared helpers ────────────────────────────────────────────────────────

    def _start_timer(self, pl_module):
        if pl_module.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event   = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            import time
            self._t0 = time.perf_counter()

    def _stop_timer(self, pl_module, batch, times, edges):
        if pl_module.device.type == "cuda":
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            import time
            elapsed_ms = (time.perf_counter() - self._t0) * 1000
        times.append(elapsed_ms)
        if hasattr(batch, "edge_index"):
            edges.append(batch.edge_index.shape[1])

    def _log_and_reset(self, pl_module, times, edges, prefix):
        if not times:
            return
        mean_ms     = sum(times) / len(times)
        mean_edges  = sum(edges) / len(edges) if edges else 1.0
        ms_per_edge = mean_ms / mean_edges if mean_edges > 0 else 0.0
        pl_module.log(f"{prefix}_infer_ms_mean",     mean_ms,     on_epoch=True, sync_dist=False)
        pl_module.log(f"{prefix}_infer_ms_per_edge", ms_per_edge, on_epoch=True, sync_dist=False)
        print(f"[InferenceTiming/{prefix}] mean: {mean_ms:.1f} ms/graph  "
              f"({ms_per_edge*1e6:.2f} ns/edge,  {len(times)} graphs)")
        times.clear()
        edges.clear()

    # ── validation hooks ──────────────────────────────────────────────────────

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if trainer.global_rank != 0:
            return
        self._start_timer(pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.global_rank != 0:
            return
        self._stop_timer(pl_module, batch, self._val_times, self._val_edges)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._log_and_reset(pl_module, self._val_times, self._val_edges, "val")

    # ── test hooks ────────────────────────────────────────────────────────────

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if trainer.global_rank != 0:
            return
        self._start_timer(pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.global_rank != 0:
            return
        self._stop_timer(pl_module, batch, self._test_times, self._test_edges)

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._log_and_reset(pl_module, self._test_times, self._test_edges, "test")


class ModelExportCallback(Callback):
    """Export the best model to TorchScript and/or torch.export after each epoch.

    Triggered whenever ``ModelCheckpoint`` saves a new best checkpoint, so the
    exported files always reflect the best model seen so far.  Requires the
    Lightning module to implement ``_get_export_model``,
    ``_make_export_example_inputs``, ``_save_jit``, and
    ``_save_torch_export`` (provided by ``IGNN3``).

    Activated by adding an ``export`` dict to the training config:

    .. code-block:: yaml

        export:
          jit: true
          torch_export: true
          example_num_nodes: 50000
          example_num_edges: 500000
          dynamic_shapes: true
    """

    def __init__(self):
        self._last_best_path = None

    def on_validation_epoch_end(self, trainer, pl_module):
        export_cfg = getattr(pl_module, "hparams", {}).get("export") or {}
        if not export_cfg:
            return
        if trainer.global_rank != 0:
            return
        if not hasattr(pl_module, "_get_export_model"):
            return

        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return

        # Only export when ModelCheckpoint has written a new best file
        current_best = ckpt_callback.best_model_path
        if not current_best or current_best == self._last_best_path:
            return
        self._last_best_path = current_best

        save_dir = os.path.join(pl_module.hparams["stage_dir"], "artifacts")
        os.makedirs(save_dir, exist_ok=True)

        export_model = pl_module._get_export_model()
        example_inputs = pl_module._make_export_example_inputs(export_cfg)

        if export_cfg.get("jit", False):
            pl_module._save_jit(export_model, example_inputs, save_dir)
        if export_cfg.get("onnx", False):
            pl_module._save_onnx(export_model, example_inputs, save_dir)
        if export_cfg.get("torch_export", False):
            pl_module._save_torch_export(export_model, example_inputs, save_dir, export_cfg)


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
            torch.load(
                checkpoint, map_location=torch.device("cpu"), weights_only=False
            )["hyper_parameters"],
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

    throughput_callback = ThroughputCallback()

    lr_monitor_callback = LearningRateMonitor(logging_interval=None)
    callbacks = [checkpoint_callback, throughput_callback, lr_monitor_callback]
    if config.get("scheduler") == "ReduceLROnPlateau":
        callbacks.append(PlateauWarmupCallback())

    if config.get("export"):
        callbacks.append(ModelExportCallback())
    if config.get("time_inference"):
        callbacks.append(InferenceTimingCallback())

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        val_check_interval=config.get("val_check_interval", 1.0),
        callbacks=callbacks,
        logger=logger,
        precision=config.get("precision", 32),
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        default_root_dir=default_root_dir,
    )


def get_stage_module(
    config, stage_module_class, checkpoint_path=None, checkpoint_resume_dir=None, hparams_file=None
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
        stage_module, config = load_module(checkpoint_path, stage_module_class, hparams_file)
    else:
        stage_module = stage_module_class(config)
    return stage_module, config, default_root_dir, checkpoint_path


from pytorch_lightning.core.saving import load_hparams_from_yaml
def load_module(checkpoint_path, stage_module_class, hparams_file=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    config = load_hparams_from_yaml(hparams_file)
    hparams = checkpoint["hyper_parameters"]

    stage_module = stage_module_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams={**hparams, **config}
    )
    return stage_module, config
