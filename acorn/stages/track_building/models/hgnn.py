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
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from time import process_time

# Local imports
from .ml_track_building import MLTrackBuildingStage
from acorn.stages.track_building.models.hgnn_matching_utils import (
    evaluate_tracking,
    get_statistics,
)
from acorn.stages.track_building.models.ml_modules.hgnn_models import (
    Pooling,
    InteractionGNNBlock,
    HierarchicalGNNBlock,
)
from acorn.stages.track_building.models.ml_modules.gnn_cells import find_neighbors
from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HierarchicalGNN(MLTrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        model_config = hparams["model_config"]

        # Initialize Modules

        self.interaction_layer = InteractionGNNBlock(
            d_model=model_config["d_model"],
            n_node_features=len(hparams["node_features"]),
            d_hidden=model_config["d_hidden"],
            n_iterations=model_config["n_interaction_iterations"],
            hidden_activation=model_config["hidden_activation"],
            output_activation=model_config["output_activation"],
            dropout=model_config["dropout"],
            checkpoint=hparams["checkpoint"],
        )

        self.pooling_layer = Pooling(
            d_model=model_config["d_model"],
            d_hidden=model_config["d_hidden"],
            emb_size=model_config["emb_size"],
            n_output_layers=model_config["n_output_layers"],
            hidden_activation=model_config["hidden_activation"],
            output_activation=model_config["output_activation"],
            dropout=model_config["dropout"],
            momentum=model_config["cut_momentum"],
            bsparsity=model_config["bsparsity"],
            ssparsity=model_config["ssparsity"],
            resolution=model_config["resolution"],
            min_size=model_config["min_size"],
            use_prebuilt=model_config["use_prebuilt"],
        )

        self.hgnn_layer = HierarchicalGNNBlock(
            d_model=model_config["d_model"],
            emb_size=model_config["emb_size"],
            d_hidden=model_config["d_hidden"],
            n_output_layers=model_config["n_output_layers"],
            n_iterations=model_config["n_hierarchical_iterations"],
            hidden_activation=model_config["hidden_activation"],
            output_activation=model_config["output_activation"],
            dropout=model_config["dropout"],
            checkpoint=hparams["checkpoint"],
        )

        self.save_hyperparameters(hparams)

    def forward(self, partial_event):
        node_attr = torch.stack(
            [
                partial_event[feature] / scale
                for feature, scale in zip(
                    self.hparams["node_features"], self.hparams["node_scales"]
                )
            ],
            dim=-1,
        ).float()
        graph = torch.cat(
            [partial_event.edge_index, partial_event.edge_index.flip(0)], dim=1
        )
        nodes, edges = self.interaction_layer(node_attr, graph)
        (
            emb,
            semb,
            bgraph,
            bweights,
            sgraph,
            sweights,
            emb_logits,
            mask,
        ) = self.pooling_layer(nodes, graph, partial_event.cluster)
        if self.hparams["model_config"].get("cut_edge", False):
            edges = edges[mask]
            graph = graph[:, mask]
        logits = self.hgnn_layer(
            nodes, edges, semb, graph, bgraph, bweights, sgraph, sweights
        )

        return bgraph, logits, emb, emb_logits

    def loss_function(self, batch, bgraph, logits, emb, emb_logits, prefix="train"):
        loss_weight = 1 - np.sin(
            np.clip(
                (self.trainer.global_step - self.hparams["pretraining_steps"])
                / self.hparams["auxiliary_decay"],
                0,
                1,
            )
            * np.pi
            / 2
        )

        bgraph_y, weights, relavent_mask = batch.fetch_truth(
            bgraph,
            torch.sigmoid(
                (1 - loss_weight) * logits + loss_weight * emb_logits
            ).detach(),
        )

        classification_loss = (
            weights
            * F.binary_cross_entropy_with_logits(
                logits[relavent_mask], bgraph_y.float()
            )
        ).sum()

        # compute auxaliary loss
        hnm_graph_idxs = find_neighbors(
            emb, emb, r_max=self.hparams["margin"], k_max=50
        )
        positive_idxs = hnm_graph_idxs >= 0
        ind = (
            torch.arange(hnm_graph_idxs.shape[0], device=self.device)
            .unsqueeze(1)
            .expand(hnm_graph_idxs.shape)
        )
        hnm_graph = torch.stack(
            [ind[positive_idxs], hnm_graph_idxs[positive_idxs]], dim=0
        )
        hnm_graph = torch.cat(
            [
                hnm_graph,
                batch.partial_event.edge_index,
                batch.partial_event.track_edges,
            ],
            dim=1,
        )
        hnm_graph, hinge, emb_weights = batch.fetch_emb_truth(hnm_graph)
        dist = (emb[hnm_graph[0]] - emb[hnm_graph[1]]).square().sum(-1)
        dist = (dist + 1e-12).sqrt()
        embedding_loss = (
            F.hinge_embedding_loss(
                dist, hinge, margin=self.hparams["margin"], reduction="none"
            )
            / self.hparams["margin"]
        ).square()
        embedding_loss = (emb_weights * embedding_loss).sum()

        loss = (
            1 - loss_weight
        ) * classification_loss + loss_weight * embedding_loss * self.hparams.get(
            "emb_ratio", 1
        )
        return loss, {
            prefix + "_loss": loss,
            prefix + "_classification_loss": classification_loss,
            prefix + "_embedding_loss": embedding_loss,
            prefix
            + "_graph_construction_efficiency": bgraph_y.sum()
            / batch.truth_info["nhits"].sum(),
        }

    def training_step(self, batch, batch_idx):
        bgraph, logits, emb, emb_logits = self(batch.partial_event)
        loss, info = self.loss_function(
            batch, bgraph, logits, emb, emb_logits, prefix="train"
        )

        info["num_cluster"] = (bgraph[1].max() + 1).float()
        if not self.hparams["model_config"]["use_prebuilt"]:
            info["score_cut"] = self.pooling_layer.score_cut.item()
        for key, value in info.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=False,
                batch_size=1,
                sync_dist=True,
            )

        return loss

    def shared_evaluation(self, batch, batch_idx, stage="val"):
        bgraph, logits, emb, emb_logits = self(batch.partial_event)
        loss, info = self.loss_function(
            batch, bgraph, logits, emb, emb_logits, prefix=stage
        )

        # evaluation
        scores = torch.sigmoid(logits)
        bgraph = bgraph[:, scores > self.hparams["score_cut"]]

        matching_df, truth_df = evaluate_tracking(
            batch.full_event,
            batch.get_tracks(bgraph),
            min_hits=self.hparams["min_track_length"],
            target_tracks=self.hparams["target_tracks"],
            matching_fraction=self.hparams["matching_fraction"],
            style=self.hparams["matching_style"],
        )

        stats = get_statistics(matching_df, truth_df)
        info.update(
            {
                "tracking_efficiency": stats["reconstructed_signal"]
                / stats["total_signal"],
                "duplicate_rate": stats["duplicate_rate"],
                "fake_rate": stats["fake_rate"],
            }
        )

        for key, value in info.items():
            self.log(
                key,
                value,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

        return info

    def validation_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.shared_evaluation(batch, batch_idx, stage="test")

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
            batch.full_event.event_id[0]
            if isinstance(batch.full_event.event_id, list)
            else batch.full_event.event_id
        )
        os.makedirs(
            os.path.join(self.hparams["stage_dir"], dataset.data_name), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.hparams["stage_dir"], f"{dataset.data_name}_tracks"),
            exist_ok=True,
        )
        output_dir = os.path.join(
            self.hparams["stage_dir"], dataset.data_name, f"event{event_id}.pyg"
        )
        track_dir = os.path.join(
            self.hparams["stage_dir"],
            f"{dataset.data_name}_tracks",
            f"event{event_id}.txt",
        )
        self.build_track(batch, output_dir, track_dir)

    def build_track(self, batch, output_dir, track_dir):
        start_time = process_time()

        bgraph, logits, _, _ = self(batch.partial_event)

        # evaluation
        scores = torch.sigmoid(logits)
        batch.full_event.config.append(self.hparams)

        batch.full_event.bgraph = batch.get_tracks(bgraph)
        batch.full_event.boutput = logits
        batch.full_event.bscores = scores

        # For backward compatibility build labels, should be removed in the future.
        mask = scores >= self.hparams.get("score_cut", 0)
        bgraph = (
            batch.get_tracks(
                bgraph[:, mask][:, torch.argsort(scores[mask], descending=True)]
            )
            .cpu()
            .numpy()
        )
        _, idx = np.unique(bgraph[0], return_index=True)
        filtered_bgraph = bgraph[:, idx]
        track_id_tensor = -torch.ones(batch.full_event.hit_x.shape[0], dtype=torch.long)
        track_id_tensor[filtered_bgraph[0]] = torch.as_tensor(filtered_bgraph[1]).long()
        batch.full_event.edge_track_labels = track_id_tensor

        batch.full_event.time_taken = process_time() - start_time + batch.time_taken

        if not self.hparams.get("variable_with_prefix"):
            batch.full_event = remove_variable_name_prefix_in_pyg(batch.full_event)
        torch.save(batch.full_event, output_dir)

        tracks = (
            pd.DataFrame(
                {
                    "hit_id": bgraph[0],
                    "track_id": bgraph[1],
                }
            )
            .groupby("track_id")["hit_id"]
            .apply(list)
        )

        with open(
            track_dir,
            "w",
        ) as f:
            f.write(
                "\n".join(
                    str(t).replace(",", "").replace("[", "").replace("]", "")
                    for t in tracks.values
                )
            )
