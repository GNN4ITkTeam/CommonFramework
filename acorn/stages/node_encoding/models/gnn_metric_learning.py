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
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import torch.nn as nn
from cuml.neighbors import NearestNeighbors
import cupy
import pytorch_pfn_extras as ppe

from torch.utils.checkpoint import checkpoint

# Local imports
from ..node_encoding_stage import NodeEncodingStage
from acorn.utils import (
    make_mlp,
    get_condition_lambda,
    get_optimizers,
)


class GNNMetricLearning(NodeEncodingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        in_channels = len(hparams["node_features"])

        self.node_encoder = make_mlp(
            in_channels,
            [hparams["encoder_hidden"]] * (hparams["n_encoder_layers"] - 1)
            + [hparams["node_rep_dim"]],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        if hparams["n_iters"] > 0:
            self.edge_networks = nn.ModuleList(
                [
                    make_mlp(
                        (hparams["node_rep_dim"] * 2)
                        if i % hparams["n_gnns_per_iter"] == 0
                        else (hparams["node_rep_dim"] * 2 + hparams["edge_rep_dim"]),
                        [hparams["edge_hidden"]] * (hparams["n_edge_layers"] - 1)
                        + [hparams["edge_rep_dim"] + 1],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        hidden_activation=hparams["hidden_activation"],
                        output_activation=hparams["hidden_activation"],
                    )
                    for i in range(
                        (1 if hparams["recurrent"] else hparams["n_iters"])
                        * (
                            2
                            if hparams["recurrent_gnn"]
                            else hparams["n_gnns_per_iter"]
                        )
                    )
                ]
            )

        self.node_network_0 = make_mlp(
            hparams["node_rep_dim"],
            [hparams["node_0_hidden"]] * (hparams["n_node_0_layers"] - 1)
            + [hparams["node_rep_dim"]],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["hidden_activation"],
        )

        if hparams["n_iters"] > 0:
            self.node_networks = nn.ModuleList(
                [
                    make_mlp(
                        hparams["node_rep_dim"] + hparams["edge_rep_dim"],
                        [hparams["node_hidden"]] * (hparams["n_node_layers"] - 1)
                        + [hparams["node_rep_dim"]],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        hidden_activation=hparams["hidden_activation"],
                        output_activation=hparams["hidden_activation"],
                    )
                    for i in range(
                        (1 if hparams["recurrent"] else hparams["n_iters"])
                        * (
                            1
                            if hparams["recurrent_gnn"]
                            else hparams["n_gnns_per_iter"]
                        )
                    )
                ]
            )

        self.node_decoders = nn.ModuleList(
            [
                make_mlp(
                    hparams["node_rep_dim"],
                    [hparams["decoder_hiden"]] * (hparams["n_decoder_layers"] - 1)
                    + [hparams["node_pspace_dim"]],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    hidden_activation=hparams["hidden_activation"],
                    output_activation=hparams["output_activation"],
                )
                for i in range(1 if hparams["recurrent"] else (hparams["n_iters"] + 1))
            ]
        )

        if hparams.get("node_filter"):
            self.node_filters = nn.ModuleList(
                [
                    make_mlp(
                        hparams["node_rep_dim"],
                        [hparams["node_filter_hiden"]]
                        * (hparams["n_node_filter_layers"] - 1)
                        + [1],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        hidden_activation=hparams["hidden_activation"],
                        output_activation="Sigmoid",
                    )
                    for i in range(
                        1 if hparams["recurrent"] else (hparams["n_iters"] + 1)
                    )
                ]
            )

        ppe.cuda.use_torch_mempool_in_cupy()

    def cu_knn_graph(self, x, k, loop=False, cosine=False):
        if not loop:
            k += 1
        with cupy.cuda.Device(self.device.index):
            x_cu = cupy.from_dlpack(x.detach())
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(x_cu)
            _, graph_idxs = knn.kneighbors(x_cu)
            graph_idxs = torch.from_dlpack(graph_idxs)
        ind = (
            torch.arange(graph_idxs.shape[0], device=self.device)
            .unsqueeze(1)
            .expand(graph_idxs.shape)
        )
        graph = torch.stack([graph_idxs.flatten(), ind.flatten()], dim=0)
        if not loop:
            return graph[:, graph[0] != graph[1]]
        else:
            return graph

    def get_knn_edges(self, x, i):
        x = self.node_decoders[0 if self.hparams["recurrent"] else i](x).detach()
        if self.hparams["embedding_norm"]:
            x = F.normalize(x)
        if self.hparams.get("node_filter"):
            node_score = self.node_filters[0 if self.hparams["recurrent"] else i](x)

        k = (
            self.hparams["knn_train"]
            if type(self.hparams["knn_train"]) == int
            else self.hparams["knn_train"][i]
        )

        if self.hparams.get("cu_knn"):
            return self.cu_knn_graph(x, k=k, cosine=False, loop=False)
        else:
            return knn_graph(x, k=k, cosine=False, loop=False)

    def forward(self, batch, **kwargs):
        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        ).float()

        assert len(x) > 0, "Input node size == 0!!"

        if self.hparams.get("checkpoint", False):
            v = checkpoint(self.node_encoder, x, use_reentrant=False)
            x = checkpoint(self.node_network_0, v, use_reentrant=False)
        else:
            v = self.node_encoder(x)
            x = self.node_network_0(v)

        batch.knn_time = 0

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_iters"]):
            # KNN
            knn_start = time.time()
            if self.hparams.get("checkpoint", False):
                start, end = checkpoint(
                    self.get_knn_edges,
                    x,
                    i,
                    use_reentrant=False,
                )
            else:
                start, end = self.get_knn_edges(x, i)
            knn_end = time.time()
            # print("kNN time: ", knn_end - knn_start)
            batch.knn_time += knn_end - knn_start
            x = v
            # gat_start = time.time()
            if self.hparams.get("checkpoint", False):
                x = checkpoint(self.gat, x, start, end, i, use_reentrant=False)
            else:
                x = self.gat(x, start, end, i)
            # gat_end = time.time()
            # print("gat time: ", gat_end - gat_start)

        if self.hparams.get("checkpoint", False):
            x = checkpoint(self.node_decoders[-1], x, use_reentrant=False)
        else:
            x = self.node_decoders[-1](x)
        if self.hparams["embedding_norm"]:
            return F.normalize(x)
        else:
            x

    def gat(self, x, start, end, i):
        e = None
        for j in range(self.hparams["n_gnns_per_iter"]):
            x, e = self.message_passing(e, x, start, end, i, j)
        return x

    def message_passing(self, e, x, start, end, i, j):

        e = torch.cat([x[start], x[end]] if j == 0 else [x[start], x[end], e], dim=-1)

        e = self.edge_networks[
            (
                0
                if self.hparams["recurrent"]
                else (
                    i
                    * (
                        1
                        if self.hparams["recurrent_gnn"]
                        else self.hparams["n_gnns_per_iter"]
                    )
                )
            )
            + (min(1, j) if self.hparams["recurrent_gnn"] else j)
        ](e)
        w = e[:, -1:]
        w = softmax(w, end)
        e = e[:, :-1]

        # Node
        w = scatter_add(e * w, end, dim=0, dim_size=x.shape[0])
        # w = scatter_mean(e, end, dim=0, dim_size=x.shape[0])
        x = torch.cat([x, w], dim=1)
        x = self.node_networks[
            (
                0
                if self.hparams["recurrent"]
                else (
                    i
                    * (
                        1
                        if self.hparams["recurrent_gnn"]
                        else self.hparams["n_gnns_per_iter"]
                    )
                )
            )
            + (0 if self.hparams["recurrent_gnn"] else j)
        ](x)

        return x, e

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
        y = self.get_target(batch, edges)
        w = self.get_weight(batch, edges, y)
        tp = torch.sum(y == 1)
        target_tp = torch.sum((y == 1) & (w > 0))
        return self.hinge_loss(batch, edges, y=y, w=w), tp, len(y), target_tp

    def random_loss(self, batch):
        edges = torch.randint(
            0,
            batch.hit_embedding.shape[0],
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
                        condition_key in batch.keys
                    ), f"Condition key {condition_key} not found in event keys {batch.keys}"

                    condition_lambda = get_condition_lambda(
                        condition_key, condition_val
                    )
                    value_mask = condition_lambda(batch)
                    graph_mask = graph_mask & value_mask[edges[0]]

                w[graph_mask] = weight_spec["weight"]

        return w

    def get_distances(self, batch, edges):
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

        return torch.sqrt(d + 1e-12)

    def training_step(self, batch, batch_idx):

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

        embedding = self(batch)

        batch.hit_embedding = embedding

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
