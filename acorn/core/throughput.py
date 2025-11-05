import logging
import time
from typing import Any

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback


class ThroughputCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx):
        self.batch_start=time.perf_counter()

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ):
        batch_time = time.perf_counter() - self.batch_start
        pl_module.log(
            "train/forward_time_seconds",
            batch_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True  # no need to log for all devices
        )

        number_samples = batch.size(0)
        pl_module.log(
            "train/forward_time_seconds_per_sample",
            number_samples/batch_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True  # no need to log for all devices
        )

    def on_before_backward(self, trainer, pl_module, loss):
        self.backward_start = time.perf_counter()

    def on_after_backward(self, trainer, pl_module):
        backward_time = time.perf_counter() - self.backward_start
        pl_module.log(
            "train/backward_time_seconds",
            backward_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True  # no need to log for all devices
        )

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        self.step_start = time.perf_counter()

    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        # This will get called at the beginning of training to clear any gradients from tuning etc.
        # In those cases the step_start is not set so we do nothing.
        if not hasattr(self, "step_start"):
            return
        step_end = time.perf_counter() - self.step_start
        pl_module.log(
            "train/step_time_seconds",
            step_end,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True  # no need to log for all devices
        )