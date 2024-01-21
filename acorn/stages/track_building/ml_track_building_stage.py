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

# 3rd party imports
import warnings
import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader

# Local imports
from .track_building_stage import TrackBuildingStage
from acorn.stages.track_building.utils import PartialData
from acorn.utils import (
    load_datafiles_in_dir,
    get_optimizers,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLTrackBuildingStage(TrackBuildingStage, LightningModule):
    def __init__(self, hparams):
        TrackBuildingStage.__init__(self, hparams, get_logger=False)
        LightningModule.__init__(self)
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """        
        self.dataset_class = PartialGraphDataset
        
    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if stage in ["fit", "predict"]:
            self.load_data(stage, self.hparams["input_dir"])
        elif stage == "test":
            self.load_data(stage, self.hparams["stage_dir"])
        
    def train_dataloader(self):
        """
        Load the training set.
        """
        if self.trainset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
        return DataLoader(
            self.trainset, batch_size=1, num_workers=num_workers, shuffle=True, collate_fn = lambda lst: lst[0]
        )

    def val_dataloader(self):
        """
        Load the validation set.
        """
        if self.valset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[1]
        return DataLoader(self.valset, batch_size=1, num_workers=num_workers, collate_fn = lambda lst: lst[0])

    def test_dataloader(self):
        """
        Load the test set.
        """
        if self.testset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[2]
        return DataLoader(self.testset, batch_size=1, num_workers=num_workers, collate_fn = lambda lst: lst[0], shuffle=False)

    def predict_dataloader(self):
        """
        Load the prediction sets (which is a list of the three datasets)
        """
        return [self.train_dataloader(), self.val_dataloader(), self.test_dataloader()]
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, PartialData):
            batch.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch
    
    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(self.parameters(), self.hparams)
        return optimizer, scheduler

    def on_train_start(self):
        self.trainer.strategy.optimizers = [
            self.trainer.lr_scheduler_configs[0].scheduler.optimizer
        ]

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # after reaching minimum learning rate, stop LR decay
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], self.hparams.get("min_lr", 0))

        if self.hparams.get("debug") and self.trainer.current_epoch == 0:
            warnings.warn("DEBUG mode is on. Will print out gradient if encounter None")
            invalid_gradient = False
            for param in self.parameters():
                if param.grad is None:
                    warnings.warn(
                        "Some parameters get non-numerical gradient. Check model and"
                        " train settings"
                    )
                    invalid_gradient = True
                    break
            if invalid_gradient:
                print([param.grad for param in self.parameters()])
            self.hparams["debug"] = False
            
class PartialGraphDataset(Dataset):
    """
    The custom default HGNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage

        self.input_paths = load_datafiles_in_dir(
            self.input_dir, self.data_name, self.num_events
        )
        self.input_paths.sort()  # We sort here for reproducibility

    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        return PartialData(event, **self.hparams["data_config"])

