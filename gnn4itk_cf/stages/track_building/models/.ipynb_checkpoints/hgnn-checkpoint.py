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
import os
import logging
import torch
import math
from tqdm import tqdm
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch.utils.data import DataLoader
from class_resolver import ClassResolver


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local imports
from ..track_building_stage import TrackBuildingStage
from gnn4itk_cf.stages.track_building.utils import evaluate_tracking, get_statistics, PartialData
from gnn4itk_cf.stages.track_building.models.gnn_modules.hgnn_models import Pooling, InteractionGNNBlock, HierarchicalGNNBlock
from gnn4itk_cf.stages.track_building.models.gnn_modules.gnn_cells import find_neighbors
from gnn4itk_cf.utils import (
    load_datafiles_in_dir,
    get_optimizers,
)


class HierarchicalGNN(TrackBuildingStage, LightningModule):
    def __init__(self, hparams):
        TrackBuildingStage.__init__(self, hparams)
        LightningModule.__init__(self)
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        self.dataset_class = PartialGraphDataset
        
        model_config = hparams["model_config"]
        
        # Initialize Modules
        
        self.interaction_layer = InteractionGNNBlock(
            d_model = model_config["d_model"],
            n_node_features = len(hparams["node_features"]),
            n_node_layers = model_config["n_node_layers"],
            n_edge_layers = model_config["n_edge_layers"],
            n_iterations = model_config["n_interaction_iterations"],
            hidden_activation = model_config["hidden_activation"],
            output_activation = model_config["output_activation"],
            dropout = model_config["dropout"],
        )
        
        self.pooling_layer = Pooling(
            d_model = model_config["d_model"],
            emb_size = model_config["emb_size"],
            n_output_layers = model_config["n_output_layers"],
            hidden_activation = model_config["hidden_activation"],
            output_activation = model_config["output_activation"],
            dropout = model_config["dropout"],
            momentum = model_config["cut_momentum"],
            bsparsity = model_config["bsparsity"],
            ssparsity = model_config["ssparsity"],
            resolution = model_config["resolution"],
            min_size = model_config["min_size"], 
        )
        
        self.hgnn_layer = HierarchicalGNNBlock(
            d_model = model_config["d_model"],
            emb_size = model_config["emb_size"],
            n_node_layers = model_config["n_node_layers"],
            n_edge_layers = model_config["n_edge_layers"],
            n_output_layers = model_config["n_output_layers"],
            n_iterations = model_config["n_hierarchical_iterations"],
            hidden_activation = model_config["hidden_activation"],
            output_activation = model_config["output_activation"],
            dropout = model_config["dropout"],
        )
        
        self.save_hyperparameters(hparams)
        
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
        return DataLoader(self.testset, batch_size=1, num_workers=num_workers, collate_fn = lambda lst: lst[0])

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
    
    def forward(self, partial_event):
        node_attr = torch.stack(
            [
                partial_event[feature] / scale 
                for feature, scale in zip(self.hparams["node_features"], self.hparams["node_scales"])
            ], dim=-1
        ).float()
        graph = torch.cat([partial_event.edge_index, partial_event.edge_index.flip(0)], dim=1)
        nodes, edges = self.interaction_layer(node_attr, graph)
        emb, semb, bgraph, bweights, sgraph, sweights, emb_logits = self.pooling_layer(nodes, graph)
        logits = self.hgnn_layer(nodes, edges, semb, graph, bgraph, bweights, sgraph, sweights)
        
        return bgraph, logits, emb_logits, emb
    
    def loss_function(self, batch, bgraph, logits, emb_logits, emb, prefix="train"):
        
        loss_weight = math.sin(
            math.pi / 2 * min(1, max(0, self.trainer.global_step - self.hparams["auxiliary_emb_steps"]) / self.hparams["auxiliary_logit_steps"])
        )
        
        logits = logits + (1 - loss_weight) * emb_logits
        bgraph_y, weights, relavent_mask, truth_graph = batch.fetch_truth(bgraph, torch.sigmoid(logits))
        classification_loss = (
            weights * F.binary_cross_entropy_with_logits(logits[relavent_mask], bgraph_y.float())
        ).sum()
        
        hnm_graph_idxs = find_neighbors(emb, emb, r_max=self.hparams["margin"], k_max=50)
        positive_idxs = (hnm_graph_idxs >= 0)
        ind = torch.arange(hnm_graph_idxs.shape[0], device = self.device).unsqueeze(1).expand(hnm_graph_idxs.shape)
        hnm_graph = torch.stack([ind[positive_idxs], hnm_graph_idxs[positive_idxs]], dim = 0)
        hnm_graph = torch.cat([hnm_graph, batch.partial_event.edge_index], dim = 1)
        hnm_graph, hinge, emb_weights = batch.fetch_emb_truth(hnm_graph)
        dist = (emb[hnm_graph[0]] - emb[hnm_graph[1]]).square().sum(-1)
        dist = (dist + 1e-12).sqrt()
        embedding_loss = (F.hinge_embedding_loss(dist, hinge, margin=self.hparams["margin"], reduction = "none") / self.hparams["margin"]).square()
        embedding_loss = (emb_weights * embedding_loss).sum()
        
        loss = loss_weight * classification_loss + (1 - loss_weight) * embedding_loss
        
        return loss, {
            prefix + "_loss": loss,
            prefix + "_classification_loss": classification_loss, 
            prefix + "_embedding_loss": embedding_loss,
            prefix + "_graph_construction_efficiency": bgraph_y.sum() / batch.truth_info["nhits"].sum()        
        }
        

    def training_step(self, batch, batch_idx):
        bgraph, logits, emb_logits, emb, semb = self(batch.partial_event)
        loss, info = self.loss_function(batch, bgraph, logits, emb_logits, emb, prefix="train")
        
        info["num_cluster"] = (bgraph[1].max() + 1).float()
        info["score_cut"] = self.pooling_layer.score_cut.item()
        self.log_dict(
            info,
            batch_size=1,
            sync_dist=True
        )

        return loss

    def shared_evaluation(self, batch, batch_idx, stage = "val"):
        bgraph, logits, emb_logits, emb, semb = self(batch.partial_event)
        loss, info = self.loss_function(batch, bgraph, logits, emb_logits, emb, prefix=stage)
        
        # evaluation
        scores = torch.sigmoid(logits)
        bgraph = bgraph[:, scores > self.hparams["score_cut"]]
        
        matching_df, truth_df = evaluate_tracking(
            batch.full_event, 
            batch.get_tracks(bgraph),
            min_hits = self.hparams["min_track_length"],
            signal_selection = self.hparams["selection"],
            target_selection = {},
            matching_fraction = self.hparams["matching_fraction"],
            style=self.hparams["matching_style"]
        )
        
        stats = get_statistics(matching_df, truth_df)
        info.update({
            "tracking_efficiency": stats["reconstructed_signal"] / stats["total_signal"],
            "duplicate_rate": stats["duplicate_rate"],
            "fake_rate": stats["fake_rate"],
        })
        
        self.log_dict(
            info,
            batch_size=1,
            sync_dist=True
        )

        return info

    def validation_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx, stage = "val")

    def test_step(self, batch, batch_idx):
        return self.shared_evaluation(batch, batch_idx, stage = "test")
    
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This function handles the prediction of each graph. It is called in the `infer.py` script.
        It can be overwritted in your custom stage, but it should implement three simple steps:
        1. Run an edge-scoring model on the input graph
        2. Add the scored edges to the graph, as `scores` attribute
        3. Append the stage config to the `config` attribute of the graph
        """

        dataset = self.predict_dataloader()[dataloader_idx].dataset
        event_id = (
            batch.event_id[0] if isinstance(batch.event_id, list) else batch.event_id
        )
        output_dir = os.path.join(
            self.hparams["stage_dir"],
            dataset.data_name,
            f"event{event_id}.pyg",
        )
        self.build_track(batch, output_dir)
    
    def build_track(self, batch, output_dir):
        
        bgraph, logits, emb_logits, emb = self(batch.partial_event)
        loss, info = self.loss_function(batch, bgraph, logits, emb_logits, emb, prefix="val")
        
        # evaluation
        scores = torch.sigmoid(logits)
        bgraph = bgraph[:, scores > self.hparams["score_cut"]]
        batch.full_event.config.append(self.hparams)
        
        batch.full_event.bgraph = batch.get_tracks(bgraph)

        # TODO: Graph name file??
        torch.save(batch.full_event, output_dir)

            
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

