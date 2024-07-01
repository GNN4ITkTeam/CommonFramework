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
import time

# 3rd party imports
import torch.nn.functional as F

import torch
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_add, scatter_mean
import torch.nn as nn
from cuml.neighbors import NearestNeighbors
import cupy

from torch.utils.checkpoint import checkpoint

# Local imports
from ..node_encoding_stage import NodeEncodingStage, PreGraphDataset
from ...edge_classifier.models.interaction_gnn import InteractionGNN2
from acorn.utils import (
    make_mlp,
    get_condition_lambda,
    get_optimizers,
)
from acorn.utils.version_utils import get_pyg_data_keys


class TransferedObjectCondensation(NodeEncodingStage):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.dataset_class = PreGraphDataset

        hparams["batchnorm"] = hparams.get("batchnorm", False)
        hparams["node_decoder_batch_norm"] = hparams.get("node_decoder_batch_norm", False)
        hparams["node_output_transform_final_batch_norm"] = hparams.get(
            "node_output_transform_final_batch_norm", False
        )
        hparams["track_running_stats"] = hparams.get("track_running_stats", False)

        # Define the dataset to be used, if not using the default
        self.save_hyperparameters(hparams)

        self.edge_classifier = InteractionGNN2.load_from_checkpoint(hparams["edge_classifier_path"])
        self.edge_classifier.eval()
        self.edge_classifier.freeze()

        # node decoder
        self.node_decoder = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"]] * hparams["n_node_decoder_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["node_decoder_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        # edge output transform layer
        self.node_output_transform = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"], hparams["node_pspace_dim"]],
            output_activation=hparams["node_output_transform_final_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["node_output_transform_final_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

    def forward(self, batch):
        x = torch.stack(
            [batch[feature] for feature in self.edge_classifier.hparams["node_features"]], dim=-1
        ).float()

        # Same features on the 3 channels in the STRIP ENDCAP TODO: Process it in previous stage
        mask = torch.logical_or(batch.hit_region == 2, batch.hit_region == 6).reshape(
            -1
        )
        x[mask] = torch.cat([x[mask, 0:4], x[mask, 0:4], x[mask, 0:4]], dim=1)
        # print(x[:, 8:12])

        if "edge_features" in self.edge_classifier.hparams and len(self.edge_classifier.hparams) != 0:
            edge_attr = torch.stack(
                [batch[feature] for feature in self.edge_classifier.hparams["edge_features"]], dim=-1
            ).float()
        else:
            edge_attr = None

        x.requires_grad = True
        if edge_attr is not None:
            edge_attr.requires_grad = True

        # Get src and dst
        src, dst = batch.edge_index

        with torch.no_grad():
            # Encode nodes and edges features into latent spaces
            if self.hparams["checkpointing"]:
                x = checkpoint(self.edge_classifier.node_encoder, x)
                if edge_attr is not None:
                    e = checkpoint(self.edge_classifier.edge_encoder, edge_attr)
                else:
                    e = checkpoint(self.edge_classifier.edge_encoder, torch.cat([x[src], x[dst]], dim=-1))
            else:
                x = self.edge_classifier.node_encoder(x)
                if edge_attr is not None:
                    e = self.edge_classifier.edge_encoder(edge_attr)
                else:
                    e = self.edge_classifier.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
            # Apply dropout
            # x = self.dropout(x)
            # e = self.dropout(e)

            # memorize initial encodings for concatenate in the gnn loop if request
            if self.edge_classifier.hparams["concat"]:
                input_x = x
                input_e = e
            # Initialize outputs
            # outputs = []
            # Loop over gnn layers
            for i in range(self.edge_classifier.hparams["n_graph_iters"]):
                if self.hparams["checkpointing"]:
                    if self.edge_classifier.hparams["concat"]:
                        x = checkpoint(self.edge_classifier.concat, x, input_x)
                        e = checkpoint(self.edge_classifier.concat, e, input_e)
                    if (
                        self.edge_classifier.hparams["node_net_recurrent"]
                        and self.edge_classifier.hparams["edge_net_recurrent"]
                    ):
                        x, e, out = checkpoint(self.edge_classifier.message_step, x, e, src, dst)
                    else:
                        x, e, out = checkpoint(self.edge_classifier.message_step, x, e, src, dst, i)
                else:
                    if self.edge_classifier.hparams["concat"]:
                        x = torch.cat([x, input_x], dim=-1)
                        e = torch.cat([e, input_e], dim=-1)
                    if (
                        self.edge_classifier.hparams["node_net_recurrent"]
                        and self.edge_classifier.hparams["edge_net_recurrent"]
                    ):
                        x, e, out = self.edge_classifier.message_step(x, e, src, dst)
                    else:
                        x, e, out = self.edge_classifier.message_step(x, e, src, dst, i)
                # outputs.append(out)
        return F.normalize(self.node_output_transform(self.node_decoder(x))).squeeze(-1)

    # def message_step(self, x, e, src, dst, i=None):
    #     edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?
    #     if self.hparams["edge_net_recurrent"]:
    #         e_updated = self.edge_network(edge_inputs)
    #     else:
    #         e_updated = self.edge_network[i](edge_inputs)
    #     # Update nodes
    #     edge_messages_from_src = self.aggr_function(
    #         e_updated, dst, dim=0, dim_size=x.shape[0]
    #     )
    #     edge_messages_from_dst = self.aggr_function(
    #         e_updated, src, dim=0, dim_size=x.shape[0]
    #     )
    #     if self.hparams["in_out_diff_agg"]:
    #         node_inputs = torch.cat(
    #             [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
    #         )  # to check : the order dst src  x ?
    #     else:
    #         # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
    #         edge_messages = edge_messages_from_src + edge_messages_from_dst
    #         node_inputs = torch.cat([edge_messages, x], dim=-1)
    #     # x_updated = self.dropout(self.node_network[i](node_inputs))
    #     if self.hparams["node_net_recurrent"]:
    #         x_updated = self.node_network(node_inputs)
    #     else:
    #         x_updated = self.node_network[i](node_inputs)

    #     return (
    #         x_updated,
    #         e_updated,
    #         F.normalize(self.node_output_transform(self.node_decoder(x_updated))),
    #     )

    # def concat(self, x, y):
    #     return torch.cat([x, y], dim=-1)

    def cu_knn_graph(self, x, k, loop=False, cosine=False, r=None):
        if not loop:
            k += 1
        with cupy.cuda.Device(self.device.index):
            x_cu = cupy.from_dlpack(x.detach())
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(x_cu)
            d, graph_idxs = knn.kneighbors(x_cu)
            graph_idxs = torch.from_dlpack(graph_idxs)
            if r:
                d = torch.from_dlpack(d)
        ind = (
            torch.arange(graph_idxs.shape[0], device=self.device)
            .unsqueeze(1)
            .expand(graph_idxs.shape)
        )
        graph = torch.stack([graph_idxs.flatten(), ind.flatten()], dim=0)
        if r:
            graph = graph[:, d.flatten() <= r]
        if not loop:
            return graph[:, graph[0] != graph[1]]
        else:
            return graph

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(self.parameters(), self.hparams)
        return optimizer, scheduler

    def signal_loss(self, batch):
        return self.hinge_loss(
            batch,
            batch.track_edges,
            y=torch.ones(batch.track_edges.shape[1], device=self.device),
        )

    def knn_loss(self, batch, k):
        if self.hparams.get("cu_knn"):
            edges = self.cu_knn_graph(
                batch.hit_embedding.detach(), k=k, cosine=False, loop=False
            )
        else:
            edges = knn_graph(
                batch.hit_embedding.detach(), k=k, cosine=False, loop=False
            )
        if self.hparams.get("node_filter"):
            edges = batch.filter_node_list[edges]
        y = self.get_target(batch, edges)
        w = self.get_weight(batch, edges, y)
        tp = torch.sum(y == 1)
        target_tp = torch.sum((y == 1) & (w > 0))
        return self.hinge_loss(batch, edges, y=y, w=w), tp, len(y), target_tp

    def random_loss(self, batch):
        edges = torch.randint(
            0,
            batch.hit_r.shape[0],
            (2, self.hparams["randomisation"]),
            device=self.device,
        )
        return self.hinge_loss(batch, edges)

    def hinge_loss(
        self,
        batch,
        edges,
        y=None,
        w=None,
    ):
        if y is None:
            y = self.get_target(batch, edges)

        if w is None:
            w = self.get_weight(batch, edges, y)

        d = self.get_distances(batch, edges)

        loss = torch.nn.functional.hinge_embedding_loss(
            d,
            y,
            margin=self.hparams["margin"],
            reduction="none",
        ).pow(2)
        return (loss * w).sum() / w.sum()

    def get_weight(self, batch, edges, y):
        w = torch.ones(edges.shape[1], device=self.device)
        if (
            "true_default" in self.hparams["weighting"]
            and self.hparams["weighting"]["true_default"] is not None
        ):
            w[y == 1] = self.hparams["weighting"]["true_default"]
        if (
            "fake_default" in self.hparams["weighting"]
            and self.hparams["weighting"]["fake_default"] is not None
        ):
            w[y == -1] = self.hparams["weighting"]["fake_default"]
        if (
            "conditional_weighting" in self.hparams["weighting"]
            and self.hparams["weighting"]["conditional_weighting"] is not None
        ):
            for weight_spec in self.hparams["weighting"]["conditional_weighting"]:
                graph_mask = y == 1

                for condition_key, condition_val in weight_spec["conditions"].items():
                    assert (
                        condition_key in get_pyg_data_keys(batch)
                    ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(batch)}"

                    condition_lambda = get_condition_lambda(
                        condition_key, condition_val
                    )
                    value_mask = condition_lambda(batch)
                    graph_mask = graph_mask & value_mask[edges[0]]

                w[graph_mask] = weight_spec["weight"]

        return w

    def get_distances(self, batch, edges):
        # batch.filter_node_list
        if self.hparams.get("node_filter"):
            res = torch.full((edges.shape[1],), 2., device=self.device)
            node_map = torch.full((batch.filter_node_list.max() + 1, ), -1, device=self.device)
            node_map[batch.filter_node_list] = torch.arange(len(batch.filter_node_list), device=self.device)
            edge_mask = torch.isin(edges, batch.filter_node_list).all(dim=0)
            # filter_edges = edges.T[edge_mask].T
            edges = node_map[edges.T[edge_mask].T]
        reference = batch.hit_embedding[edges[1]]
        neighbors = batch.hit_embedding[edges[0]]

        try:  # This can be resource intensive, so we chunk it if it fails
            d = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            d = [
                torch.sum((ref - nei) ** 2, dim=-1)
                for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))
            ]
            d = torch.cat(d)

        d = torch.sqrt(d + 1e-12)

        if self.hparams.get("node_filter"):
            res[edge_mask] = d
        else:
            res = d
        return res

    def training_step(self, batch, batch_idx):
        max_training_graph_size = self.hparams.get("max_training_graph_size", None)
        if (
            max_training_graph_size is not None
            and batch.edge_index.shape[1] > max_training_graph_size
        ):
            return None

        batch.hit_embedding = self(batch)

        signal_loss = self.signal_loss(batch)
        knn_loss, tp, n_edges, target_tp = self.knn_loss(
            batch, self.hparams["knn_loss"]
        )
        random_loss = self.random_loss(batch)
        loss = signal_loss + knn_loss + random_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_signal_loss": signal_loss,
                "train_knn_loss": knn_loss,
                "train_random_loss": random_loss,
            },
            batch_size=1,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        batch.hit_embedding = self(batch)

        signal_loss = self.signal_loss(batch)
        knn_loss, _tp, _n_edges, _target_tp = self.knn_loss(
            batch, self.hparams["knn_loss"]
        )
        _knn_loss, tp, n_edges, target_tp = self.knn_loss(
            batch, self.hparams["knn_val"]
        )
        random_loss = self.random_loss(batch)
        loss = signal_loss + knn_loss + random_loss
        eff = (
            tp
            / self.get_number_of_true_edges(
                batch, reduction="sum", upper_bound=self.hparams["knn_val"]
            )[1]
        )
        signal_eff = (
            target_tp
            / self.get_number_of_true_edges(
                batch,
                target="weight-based",
                reduction="sum",
                upper_bound=self.hparams["knn_val"],
            )[1]
        )
        pur = tp / n_edges
        # f1 = 2 * (eff * pur) / (eff + pur)

        current_lr = self.optimizers().param_groups[0]["lr"]

        # self.log("train_loss", loss, batch_size=1)
        self.log_dict(
            {
                "val_loss": loss,
                # "val_signal_loss": signal_loss,
                # "val_knn_loss": knn_loss,
                # "val_random_loss": random_loss,
                "lr": current_lr,
                "val_eff": eff,
                "val_signal_eff": signal_eff,
                "val_pur": pur,
                # "val_f1": f1,
            },
            batch_size=1,
        )

        return loss

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 0:
            return

        start = time.time()

        dataset = self.predict_dataloader()[dataloader_idx].dataset
        # data_name = ["trainset", "valset", "testset"][dataloader_idx]
        # dataset = getattr(self, data_name)
        if os.path.isfile(
            os.path.join(
                self.hparams["stage_dir"],
                dataset.data_name,
                f"event{batch.event_id[0]}.pyg",
            )
        ):
            return

        batch.hit_embedding = self(batch)

        dataset.unscale_features(batch)

        end = time.time()
        batch.inference_time = end - start

        self.save_graph(batch, dataset.data_name)

    def save_graph(self, event, data_name):
        event.config.append(self.hparams)
        os.makedirs(os.path.join(self.hparams["stage_dir"], data_name), exist_ok=True)
        torch.save(
            event.cpu(),
            os.path.join(
                self.hparams["stage_dir"], data_name, f"event{event.event_id[0]}.pyg"
            ),
        )
