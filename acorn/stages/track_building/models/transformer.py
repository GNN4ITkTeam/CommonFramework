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
from torch import nn
import math
import torch.nn.functional as F
import pandas as pd
from torch_scatter import scatter_mean

# Local imports
from .ml_track_building import MLTrackBuildingStage
from acorn.stages.track_building.models.hgnn_matching_utils import (
    evaluate_tracking,
    get_statistics,
)
from acorn.utils.ml_utils import make_mlp
from acorn.stages.graph_construction.models.utils import graph_intersection
from acorn.stages.track_building.models.ml_modules.gnn_cells import (
    DynamicGraphConstruction,
)
from acorn.stages.track_building.models.ml_modules.transformer_utils import (
    BTransformerBlock,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(MLTrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        model_config = hparams["model_config"]

        # Initialize Modules
        self.encoder = make_mlp(
            input_size=len(hparams["node_features"]),
            sizes=[model_config["d_model"]] * 2,
            hidden_activation="GELU",
        )
        self.transformer_blocks = nn.ModuleList(
            [
                BTransformerBlock(
                    d_model=model_config["d_model"],
                    d_ff=model_config["d_ff"],
                    heads=model_config["heads"],
                    dropout=model_config["dropout"],
                )
                for _ in range(model_config["n_iter"])
            ]
        )
        self.output_hits = make_mlp(
            input_size=model_config["d_model"],
            sizes=[model_config["d_model"], model_config["d_emb"]],
            hidden_activation="GELU",
        )
        self.output_tracks = make_mlp(
            input_size=model_config["d_model"],
            sizes=[model_config["d_model"], model_config["d_emb"]],
            hidden_activation="GELU",
        )
        self.output_model = make_mlp(
            input_size=2 * model_config["d_model"],
            sizes=[model_config["d_model"], 1],
            hidden_activation="GELU",
        )
        self.graph_construction = DynamicGraphConstruction(
            k=model_config["sparsity"],
            symmetrize=False,
            return_weights=False,
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

        hits = self.encoder(node_attr)
        tracks = scatter_mean(
            hits[partial_event.cluster[0]], partial_event.cluster[1], dim=0
        )
        hits, tracks = hits[:, None, :], tracks[:, None, :]
        for layer in self.transformer_blocks:
            hits, tracks = layer(hits, tracks)

        emb = self.output_hits(hits.detach()).squeeze(1)
        semb = self.output_tracks(tracks.detach()).squeeze(1)

        bgraph = self.graph_construction(
            emb, semb, original_graph=partial_event.cluster
        )
        emb_logits = -torch.log(
            (emb[bgraph[0]] - semb[bgraph[1]]).square().sum(-1).clamp(min=1e-12)
        )
        logits = self.output_model(
            torch.cat([hits[bgraph[0]], tracks[bgraph[1]]], dim=-1).squeeze(1)
        )

        return bgraph, logits, emb, semb, emb_logits

    def embedding_loss(self, bgraph, y, emb, semb):
        hinge = 2 * y.float() - 1
        dist = (emb[bgraph[0]] - semb[bgraph[1]]).square().sum(-1)
        dist = (dist + 1e-12).sqrt()
        embedding_loss = (
            (
                F.hinge_embedding_loss(
                    dist, hinge, margin=self.hparams["margin"], reduction="none"
                )
                / self.hparams["margin"]
            )
            .square()
            .mean()
        )
        return embedding_loss

    def loss_function(
        self, batch, bgraph, logits, emb, semb, emb_logits, prefix="train"
    ):

        loss_weight = math.exp(
            -max(0, self.trainer.global_step - self.hparams["pretraining_steps"])
            / self.hparams["auxiliary_decay"]
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

        # compute embedding loss
        emb_bgraph, emb_y = graph_intersection(
            bgraph,
            batch.partial_event.cluster,
            return_pred_to_truth=False,
            return_truth_to_pred=False,
            unique_pred=False,
            unique_truth=True,
        )

        embedding_loss = (1 - loss_weight) * self.embedding_loss(
            emb_bgraph, emb_y
        ) + loss_weight * self.embedding_loss(bgraph, bgraph_y, emb, semb)

        loss = classification_loss + embedding_loss * self.hparams.get("emb_ratio", 1)
        return loss, {
            prefix + "_loss": loss,
            prefix + "_classification_loss": classification_loss,
            prefix + "_embedding_loss": embedding_loss,
            prefix
            + "_graph_construction_efficiency": bgraph_y.sum()
            / batch.truth_info["nhits"].sum(),
        }

    def training_step(self, batch, batch_idx):
        bgraph, logits, emb, semb, emb_logits = self(batch.partial_event)
        loss, info = self.loss_function(
            batch, bgraph, logits, emb, semb, emb_logits, prefix="train"
        )

        info["num_cluster"] = (bgraph[1].max() + 1).float()
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
        bgraph, logits, emb, semb, emb_logits = self(batch.partial_event)
        loss, info = self.loss_function(
            batch, bgraph, logits, emb, semb, emb_logits, prefix=stage
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
        output_dir = os.path.join(
            self.hparams["stage_dir"], dataset.data_name, f"event{event_id}.pyg"
        )
        self.build_track(batch, output_dir)

    def build_track(self, batch, output_dir):

        bgraph, logits, _, _, _ = self(batch.partial_event)

        # evaluation
        scores = torch.sigmoid(logits)
        batch.full_event.config.append(self.hparams)

        batch.full_event.bgraph = batch.get_tracks(bgraph)
        batch.full_event.boutput = logits
        batch.full_event.bscores = scores

        # For backward compatibility build labels, should be removed in the future.
        mask = scores >= self.hparams.get("score_cut", 0)
        filtered_bgraph = batch.get_tracks(
            bgraph[:, mask][:, torch.argsort(scores[mask])]
        ).cpu()
        track_df = pd.DataFrame(
            {"hit_id": filtered_bgraph[0], "track_id": filtered_bgraph[1]}
        )
        track_df = track_df.drop_duplicates(subset="hit_id", keep="last")
        hit_id_df = pd.DataFrame({"hit_id": batch.full_event.hit_id})
        hit_id_df = hit_id_df.merge(track_df, on="hit_id", how="left")
        hit_id_df.fillna(-1, inplace=True)
        track_id_tensor = torch.from_numpy(hit_id_df.track_id.values).long()
        batch.full_event.labels = track_id_tensor

        torch.save(batch.full_event, output_dir)

    def eval_preprocess_event(self, event, config):
        event.full_event.bgraph = event.full_event.bgraph[
            :, event.full_event.bscores > config.get("score_cut", 0)
        ]

        tracks = (
            pd.DataFrame(
                {
                    "hit_id": event.full_event.bgraph.cpu()[0],
                    "track_id": event.full_event.bgraph.cpu()[1],
                }
            )
            .groupby("track_id")["hit_id"]
            .apply(list)
        )

        os.makedirs(
            os.path.join(self.hparams["stage_dir"], "testset_tracks"),
            exist_ok=True,
        )
        with open(
            os.path.join(
                self.hparams["stage_dir"],
                "testset_tracks",
                f"event{event.full_event.event_id[0]}.txt",
            ),
            "w",
        ) as f:
            f.write(
                "\n".join(
                    str(t).replace(",", "").replace("[", "").replace("]", "")
                    for t in tracks.values
                )
            )

        return event.full_event
