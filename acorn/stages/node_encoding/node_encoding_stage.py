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

"""
This class represents the entire logic of the graph construction stage. In particular, it
1. Loads events from the Athena-dumped csv files
2. Processes them into PyG Data objects with the specificied structure (see docs)
3. Runs the training of the metric learning or module map
4. Can run inference to build graphs
5. Can run evaluation to plot/print the performance of the graph construction

TODO: Update structure with the latest Gravnet base class
"""

import sys
import time
import random
import math

sys.path.append("../")

from pytorch_lightning import LightningModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cuml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from acorn.utils import (
    get_condition_lambda,
    eval_utils,
    load_datafiles_in_dir,
    handle_hard_node_cuts,
)

from atlasify import atlasify
import atlasify as atl
import os
import numpy as np
from acorn.utils.plotting_utils import (
    get_ratio,
    plot_1d_histogram,
    # plot_eff_pur_region,
    # plot_efficiency_rz,
    # plot_score_histogram,
)


class NodeEncodingStage(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.save_hyperparameters(hparams)

        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = GraphDataset

    def setup(self, stage="fit"):
        if stage in ["fit", "predict"]:
            self.load_data(self.hparams["input_dir"], stage)
            torch.set_float32_matmul_precision("medium" if stage == "fit" else "high")
        elif stage == "test":
            self.load_data(self.hparams["stage_dir"], stage)
            torch.set_float32_matmul_precision("high")

    def load_data(self, input_dir, stage):
        for data_name, data_num in zip(
            ["trainset", "valset", "testset"], self.hparams["data_split"]
        ):
            if data_num > 0:
                dataset = self.dataset_class(
                    input_dir,
                    data_name,
                    data_num,
                    stage,
                    self.hparams,
                )
                setattr(self, data_name, dataset)

        print(
            f"Loaded {len(self.trainset) if self.trainset else 0} training events,"
            f" {len(self.valset) if self.valset else 0} validation events and {len(self.testset) if self.testset else 0} testing"
            " events"
        )

    def train_dataloader(self):
        if self.trainset is None:
            return None
        num_workers = (
            16
            if (
                "num_workers" not in self.hparams or self.hparams["num_workers"] is None
            )
            else self.hparams["num_workers"][0]
        )
        return DataLoader(self.trainset, batch_size=1, num_workers=num_workers)

    def val_dataloader(self):
        if self.valset is None:
            return None
        num_workers = (
            16
            if (
                "num_workers" not in self.hparams or self.hparams["num_workers"] is None
            )
            else self.hparams["num_workers"][1]
        )
        return DataLoader(self.valset, batch_size=1, num_workers=num_workers)

    def test_dataloader(self):
        if self.testset is None:
            return None
        num_workers = (
            16
            if (
                "num_workers" not in self.hparams or self.hparams["num_workers"] is None
            )
            else self.hparams["num_workers"][2]
        )
        return DataLoader(self.testset, batch_size=1, num_workers=num_workers)

    def predict_dataloader(self):
        """
        Load the prediction sets (which is a list of the three datasets)
        """
        dataloaders = [
            self.train_dataloader(),
            self.val_dataloader(),
            self.test_dataloader(),
        ]
        dataloaders = [
            dataloader for dataloader in dataloaders if dataloader is not None
        ]
        return dataloaders

    @classmethod
    def evaluate(cls, config, *args):
        """
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """

        # Load data from testset directory
        graph_constructor = cls(config).to(device)
        graph_constructor.setup(stage="test")

        all_plots = config["plots"]

        if config.get("trackML_label"):
            atl.ATLAS = "TrackML Dataset"

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(graph_constructor, plot_function):
                getattr(graph_constructor, plot_function)(plot_config, config)
            elif hasattr(eval_utils, plot_function):
                getattr(eval_utils, plot_function)(
                    graph_constructor, plot_config, config
                )
            else:
                print(f"Plot {plot_function} not implemented")

    def get_target(self, batch, edges):
        y = torch.ones(edges.shape[1], device=self.device) * (-1)
        y[
            (batch.hit_particle_id[edges[0]] == batch.hit_particle_id[edges[1]])
            & (batch.hit_particle_id[edges[0]] != 0)
        ] = 1
        return y

    def get_number_of_true_edges(
        self, batch, target=None, target_tracks=None, reduction=None, upper_bound=None
    ):
        if target is None:
            hit_t = batch.hit_particle_nhits[batch.hit_particle_nhits > 1] - 1
        elif target == "weight-based":
            if (
                "true_default" not in self.hparams["weighting"]
                or self.hparams["weighting"]["true_default"] != 0
            ):
                signal_mask = torch.ones_like(
                    batch.hit_particle_nhits, dtype=torch.bool
                )
            else:
                signal_mask = torch.zeros_like(
                    batch.hit_particle_nhits, dtype=torch.bool
                )

            if (
                "conditional_weighting" in self.hparams["weighting"]
                and self.hparams["weighting"]["conditional_weighting"]
            ):
                for weight_spec in self.hparams["weighting"]["conditional_weighting"]:
                    graph_mask = torch.ones_like(
                        batch.hit_particle_nhits, dtype=torch.bool
                    )

                    for condition_key, condition_val in weight_spec[
                        "conditions"
                    ].items():
                        assert (
                            condition_key in batch.keys
                        ), f"Condition key {condition_key} not found in event keys {batch.keys}"

                        condition_lambda = get_condition_lambda(
                            condition_key, condition_val
                        )
                        value_mask = condition_lambda(batch)
                        graph_mask = graph_mask * value_mask

                    if weight_spec["weight"] == 0:
                        signal_mask &= ~graph_mask
                    else:
                        signal_mask |= graph_mask
            hit_t = (
                batch.hit_particle_nhits[(batch.hit_particle_nhits > 1) & (signal_mask)]
                - 1
            )
        elif target == "mask-based":
            signal_mask = torch.ones_like(batch.hit_particle_nhits, dtype=torch.bool)
            if target_tracks:
                for condition_key, condition_val in target_tracks.items():
                    condition_lambda = get_condition_lambda(
                        condition_key, condition_val
                    )
                    value_mask = condition_lambda(batch)
                    signal_mask = signal_mask & value_mask
            hit_t = (
                batch.hit_particle_nhits[(batch.hit_particle_nhits > 1) & (signal_mask)]
                - 1
            )
        if upper_bound is not None:
            max_hit_t = torch.min(
                torch.stack(
                    [
                        hit_t,
                        torch.full(hit_t.shape, upper_bound, device=self.device),
                    ]
                ),
                dim=0,
            )[0]
        if reduction == "sum":
            hit_t = hit_t.sum()
            if upper_bound is not None:
                max_hit_t = max_hit_t.sum()
        if upper_bound is not None:
            return hit_t, max_hit_t
        else:
            return hit_t

    def get_edge_target_mask(self, event, edges, target_tracks=None, y=None):
        if y is None:
            graph_mask = torch.ones_like(edges[0], dtype=torch.bool)
        else:
            graph_mask = y == 1

        if target_tracks:
            for condition_key, condition_val in target_tracks.items():
                condition_lambda = get_condition_lambda(condition_key, condition_val)
                value_mask = condition_lambda(event)
                graph_mask = graph_mask & value_mask[edges[0]]

        return graph_mask

    def get_node_target_mask(self, event, target_tracks=None):
        graph_mask = event.hit_particle_id != 0

        if target_tracks:
            for condition_key, condition_val in target_tracks.items():
                condition_lambda = get_condition_lambda(condition_key, condition_val)
                value_mask = condition_lambda(event)
                graph_mask = graph_mask & value_mask

        return graph_mask

    def node_knn_eff_pur_fixed_k(self, plot_config, config):
        """
        Plot the graph construction efficiency vs. pT of the edge.
        """

        tp_pt_hist, tp_eta_hist = None, None
        target_tp_pt_hist, target_tp_eta_hist = None, None
        max_tp_pt_hist, max_tp_eta_hist = None, None
        max_target_tp_pt_hist, max_target_tp_eta_hist = None, None
        t_pt_hist, t_eta_hist = None, None
        p_pt_hist, p_eta_hist = None, None

        base_subtext = (
            (
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                r" $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
            )
            if not config.get("trackML_label")
            else r"$p_T > 1$GeV" + "\n"
        )

        if config.get("pT_unit", "MeV") == "MeV":
            pt_min, pt_max = 1000, 50000
        else:
            pt_min, pt_max = 1, 50
        pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)
        eta_bins = np.linspace(-4, 4)

        tp = 0
        target_tp = 0
        t = 0
        max_tp = 0
        max_target_tp = 0
        p = 0

        dataset_name = config["dataset"]
        dataset = getattr(self, dataset_name)

        for event in tqdm(dataset):
            event = event.to(self.device)
            event.edge_index = knn_graph(
                event.hit_embedding, k=plot_config["knn"], cosine=False, loop=False
            )
            event.edge_y = self.get_target(event, event.edge_index) == 1
            event.edge_target_mask = self.get_edge_target_mask(
                event,
                event.edge_index,
                config.get("target_tracks", None),
                y=event.edge_y,
            )
            event.hit_target_mask = self.get_node_target_mask(
                event, config.get("target_tracks", None)
            )
            event.hit_target_t, event.max_hit_target_t = self.get_number_of_true_edges(
                event,
                target="mask-based",
                target_tracks=config.get("target_tracks", None),
                upper_bound=plot_config["knn"],
            )
            event.hit_t, event.max_hit_t = self.get_number_of_true_edges(
                event,
                upper_bound=plot_config["knn"],
            )

            event = event.cpu()

            if tp_pt_hist is None:
                tp_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[event.edge_index[1, event.edge_y]].numpy(),
                    bins=pt_bins,
                )
                target_tp_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[
                        event.edge_index[1, event.edge_target_mask]
                    ].numpy(),
                    bins=pt_bins,
                )
                max_tp_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[event.hit_particle_nhits > 1].numpy(),
                    bins=pt_bins,
                    weights=event.max_hit_t.numpy(),
                )
                max_target_tp_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.max_hit_target_t.numpy(),
                )
                t_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.hit_target_t.numpy(),
                )
                p_pt_hist = (
                    np.histogram(event.hit_particle_pt.numpy(), bins=pt_bins)[0]
                    * plot_config["knn"]
                )
                if config.get("plot_eta", True):
                    tp_eta_hist, _ = np.histogram(
                        event.hit_particle_eta[
                            event.edge_index[1, event.edge_y]
                        ].numpy(),
                        bins=eta_bins,
                    )
                    target_tp_eta_hist, _ = np.histogram(
                        event.hit_particle_eta[
                            event.edge_index[1, event.edge_target_mask]
                        ].numpy(),
                        bins=eta_bins,
                    )
                    max_tp_eta_hist, _ = np.histogram(
                        event.hit_particle_eta[event.hit_particle_nhits > 1].numpy(),
                        bins=eta_bins,
                        weights=event.max_hit_t.numpy(),
                    )
                    max_target_tp_eta_hist, _ = np.histogram(
                        event.hit_particle_eta[event.hit_target_mask].numpy(),
                        bins=eta_bins,
                        weights=event.max_hit_target_t.numpy(),
                    )
                    t_eta_hist, _ = np.histogram(
                        event.hit_particle_eta[event.hit_target_mask].numpy(),
                        bins=eta_bins,
                        weights=event.hit_target_t.numpy(),
                    )
                    p_eta_hist = (
                        np.histogram(event.hit_particle_eta.numpy(), bins=eta_bins)[0]
                        * plot_config["knn"]
                    )
            else:
                tp_pt_hist += np.histogram(
                    event.hit_particle_pt[event.edge_index[1, event.edge_y]].numpy(),
                    bins=pt_bins,
                )[0]
                target_tp_pt_hist += np.histogram(
                    event.hit_particle_pt[
                        event.edge_index[1, event.edge_target_mask]
                    ].numpy(),
                    bins=pt_bins,
                )[0]
                max_tp_pt_hist += np.histogram(
                    event.hit_particle_pt[event.hit_particle_nhits > 1].numpy(),
                    bins=pt_bins,
                    weights=event.max_hit_t.numpy(),
                )[0]
                max_target_tp_pt_hist += np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.max_hit_target_t.numpy(),
                )[0]
                t_pt_hist += np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.hit_target_t.numpy(),
                )[0]
                p_pt_hist += (
                    np.histogram(event.hit_particle_pt.numpy(), bins=pt_bins)[0]
                    * plot_config["knn"]
                )
                if config.get("plot_eta", True):
                    tp_eta_hist += np.histogram(
                        event.hit_particle_eta[
                            event.edge_index[1, event.edge_y]
                        ].numpy(),
                        bins=eta_bins,
                    )[0]
                    target_tp_eta_hist += np.histogram(
                        event.hit_particle_eta[
                            event.edge_index[1, event.edge_target_mask]
                        ].numpy(),
                        bins=eta_bins,
                    )[0]
                    max_tp_eta_hist += np.histogram(
                        event.hit_particle_eta[event.hit_particle_nhits > 1].numpy(),
                        bins=eta_bins,
                        weights=event.max_hit_t.numpy(),
                    )[0]
                    max_target_tp_eta_hist += np.histogram(
                        event.hit_particle_eta[event.hit_target_mask].numpy(),
                        bins=eta_bins,
                        weights=event.max_hit_target_t.numpy(),
                    )[0]
                    t_eta_hist += np.histogram(
                        event.hit_particle_eta[event.hit_target_mask].numpy(),
                        bins=eta_bins,
                        weights=event.hit_target_t.numpy(),
                    )[0]
                    p_eta_hist += (
                        np.histogram(event.hit_particle_eta.numpy(), bins=eta_bins)[0]
                        * plot_config["knn"]
                    )
            tp += event.edge_y.sum().item()
            target_tp += event.edge_target_mask.sum().item()
            t += event.hit_target_t.sum().item()
            max_tp += event.max_hit_t.sum().item()
            max_target_tp += event.max_hit_target_t.sum().item()
            p += len(event.hit_particle_pt) * plot_config["knn"]

        for (
            target_tp_hist,
            max_target_tp_hist,
            t_hist,
            bins,
            xlabel,
            logx,
            filename,
        ) in zip(
            [target_tp_pt_hist, target_tp_eta_hist],
            [max_target_tp_pt_hist, max_target_tp_eta_hist],
            [t_pt_hist, t_eta_hist],
            [pt_bins, eta_bins],
            [f"$p_T$ [{config.get('pT_unit', 'MeV')}]", r"$\eta$"],
            [True, False],
            ["edgewise_efficiency_pt.png", "edgewise_efficiency_eta.png"],
        ):
            if target_tp_hist is None:
                continue
            hist, err = get_ratio(target_tp_hist, t_hist)
            hist_up, _ = get_ratio(max_target_tp_hist, t_hist)
            if "filename_template" in plot_config:
                filename = config["filename_template"] + "_" + filename

            fig, ax = plot_1d_histogram(
                hist,
                bins,
                err,
                xlabel,
                plot_config["efficiency_title"],
                plot_config.get("ylim", [0.6, 1.04]),
                "Efficiency",
                logx=logx,
                color="red",
            )

            fig, ax = plot_1d_histogram(
                hist_up,
                bins,
                np.zeros_like(hist_up),
                xlabel,
                plot_config["efficiency_title"],
                plot_config.get("ylim", [0.6, 1.04]),
                "Eff. Upper Bound",
                canvas=(fig, ax),
                logx=logx,
                color="black",
            )

            # Save the plot
            atlasify(
                atlas=True if config.get("trackML_label") else "Internal",
                subtext=base_subtext + f"kNN graph (k={plot_config['knn']})" + "\n"
                f"Global efficiency: {target_tp / t :.4f}"
                + "\n"
                + f"Efficiency upper bound: {max_target_tp / t :.4f}",
            )
            fig.savefig(os.path.join(config["stage_dir"], filename))

            print(
                "Finish plotting. Find the plot at"
                f' {os.path.join(config["stage_dir"], filename)}'
            )

        for tp_hist, max_tp_hist, p_hist, bins, xlabel, logx, filename in zip(
            [tp_pt_hist, tp_eta_hist],
            [max_tp_pt_hist, max_tp_eta_hist],
            [p_pt_hist, p_eta_hist],
            [pt_bins, eta_bins],
            [f"$p_T$ [{config.get('pT_unit', 'MeV')}]", r"$\eta$"],
            [True, False],
            ["edgewise_purity_pt.png", "edgewise_purity_eta.png"],
        ):
            if tp_hist is None:
                continue
            hist, err = get_ratio(tp_hist, p_hist)
            hist_up, _ = get_ratio(max_tp_hist, p_hist)
            if "filename_template" in plot_config:
                filename = config["filename_template"] + "_" + filename

            fig, ax = plot_1d_histogram(
                hist,
                bins,
                err,
                xlabel,
                plot_config["purity_title"],
                plot_config.get("ylim", [0.6, 1.04]),
                "Purity",
                logx=logx,
                color="red",
            )

            fig, ax = plot_1d_histogram(
                hist_up,
                bins,
                np.zeros_like(hist_up),
                xlabel,
                plot_config["purity_title"],
                plot_config.get("ylim", [0.6, 1.04]),
                "Pur. Upper Bound",
                canvas=(fig, ax),
                logx=logx,
                color="black",
            )

            # Save the plot
            atlasify(
                atlas=True if config.get("trackML_label") else "Internal",
                subtext=base_subtext + f"kNN graph (k={plot_config['knn']})" + "\n"
                f"Global efficiency: {target_tp / t :.4f}"
                + "\n"
                + f"Efficiency upper bound: {max_target_tp / t :.4f}",
            )
            fig.savefig(os.path.join(config["stage_dir"], filename))

            print(
                "Finish plotting. Find the plot at"
                f' {os.path.join(config["stage_dir"], filename)}'
            )

    def node_knn_eff_pur_vs_k(self, plot_config, config):
        # knn = range(1, 21)
        knn = range(4, 81, 4)
        tp = [0] * 20
        target_tp = [0] * 20
        t = [0] * 20
        max_tp = [0] * 20
        max_target_tp = [0] * 20
        p = [0] * 20

        base_subtext = (
            (
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                r" $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
            )
            if not config.get("trackML_label")
            else r"$p_T > 1$GeV" + "\n"
        )

        dataset_name = config["dataset"]
        dataset = getattr(self, dataset_name)

        for event in tqdm(dataset):
            event = event.to(self.device)
            for i, k in enumerate(knn):
                assert (
                    len(event.hit_embedding) > k
                ), f"number of nodes ({len(event.hit_embedding)}) < k ({k})!!"
                edge_index = knn_graph(
                    event.hit_embedding, k=k, cosine=False, loop=False
                )
                edge_y = self.get_target(event, edge_index) == 1
                edge_target_mask = self.get_edge_target_mask(
                    event,
                    edge_index,
                    config.get("target_tracks", None),
                    y=edge_y,
                )
                hit_target_t, max_hit_target_t = self.get_number_of_true_edges(
                    event,
                    target="mask-based",
                    target_tracks=config.get("target_tracks", None),
                    upper_bound=k,
                )
                hit_t, max_hit_t = self.get_number_of_true_edges(
                    event,
                    upper_bound=k,
                )

                tp[i] += edge_y.sum().item()
                target_tp[i] += edge_target_mask.sum().item()
                t[i] += hit_target_t.sum().item()
                max_tp[i] += max_hit_t.sum().item()
                max_target_tp[i] += max_hit_target_t.sum().item()
                p[i] += len(event.hit_particle_pt) * k

        knn = np.array(knn)
        tp = np.array(tp)
        target_tp = np.array(target_tp)
        t = np.array(t)
        max_tp = np.array(max_tp)
        max_target_tp = np.array(max_target_tp)
        p = np.array(p)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            knn,
            target_tp / t,
            color="red",
            marker="o",
            linestyle=":",
            label="Efficiency",
        )
        ax.plot(knn, tp / p, color="blue", marker="o", linestyle="-.", label="Purity")
        ax.plot(
            knn,
            max_target_tp / t,
            color="grey",
            markersize=0,
            linestyle=":",
            label="Eff. Upper Bound",
        )
        ax.plot(
            knn,
            max_tp / p,
            color="grey",
            markersize=0,
            linestyle="-.",
            label="Pur. Upper Bound",
        )
        ax.set_xlabel("k", ha="right", x=0.95, fontsize=14)
        ax.set_ylabel("Efficiency (Purity)", ha="right", y=0.95, fontsize=14)
        ax.legend(loc="upper right", fontsize=14)
        # ax.set_ylim(ylim)
        plt.tight_layout()

        # Save the plot
        atlasify(
            atlas=True if config.get("trackML_label") else "Internal",
            subtext=base_subtext + "kNN graph",
        )
        fig.savefig(os.path.join(config["stage_dir"], "knn_eff_pur_vs_k.png"))

        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"], "knn_eff_pur_vs_k.png")}'
        )

    def cluster(self, event, eps, min_samples):
        clusterer = cuml.cluster.DBSCAN(eps=eps, min_samples=min_samples)
        # clusterer = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=3, allow_single_cluster=True, cluster_selection_epsilon=0)
        hit_label = clusterer.fit_predict(event.hit_embedding)
        event.hit_label = torch.as_tensor(hit_label, device=self.device)

    def dbscan_eff_pur(self, plot_config, config):

        dataset_name = config["dataset"]
        dataset = getattr(self, dataset_name)

        eps = plot_config["eps"]

        if config.get("pT_unit", "MeV") == "MeV":
            pt_min, pt_max = 1000, 50000
        else:
            pt_min, pt_max = 1, 50
        pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)
        eta_bins = np.linspace(-4, 4)

        base_subtext = (
            (
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                r" $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
            )
            if not config.get("trackML_label")
            else r"$p_T > 1$GeV" + "\n"
        )

        particles_pt_hist, particles_eta_hist = None, None
        matched_target_particles_pt_hist, matched_target_particles_eta_hist = None, None

        n_particles = 0
        n_matched_particles = 0
        n_matched_tracks = 0
        n_matched_target_particles = 0
        n_matched_target_tracks = 0
        n_tracks = 0

        for event in tqdm(dataset):
            event = event.to(self.device)

            event.hit_target_mask = self.get_node_target_mask(
                event, config.get("target_tracks", None)
            )
            particles = torch.unique(
                torch.stack(
                    [
                        event.hit_particle_id,
                        event.hit_particle_pt,
                    ]
                    + (
                        [event.hit_particle_eta] if config.get("plot_eta", True) else []
                    ),
                    dim=0,
                )[:, event.hit_target_mask],
                dim=1,
            )

            self.cluster(event, eps, 3)
            uni_labels, inv_idx, count = torch.unique(
                event.hit_label, return_counts=True, return_inverse=True
            )
            event.hit_track_length = count[inv_idx]
            hit_track_info = torch.stack(
                [
                    event.hit_label,
                    event.hit_particle_id,
                    event.hit_track_length,
                    event.hit_particle_pt,
                ]
                + ([event.hit_particle_eta] if config.get("plot_eta", True) else []),
                dim=0,
            )
            uni_track_info, inv_idx, n_matched_hits = torch.unique(
                hit_track_info, dim=1, return_counts=True, return_inverse=True
            )
            matched_track_particle_id = uni_track_info[1][
                (uni_track_info[0] >= 0) & (n_matched_hits / uni_track_info[2] > 0.5)
            ]
            hit_target_track_info = hit_track_info[:, event.hit_target_mask]
            uni_target_track_info, inv_idx, n_matched_target_hits = torch.unique(
                hit_target_track_info, dim=1, return_counts=True, return_inverse=True
            )
            matched_target_tracks = uni_target_track_info[
                [1, 3] + ([4] if config.get("plot_eta", True) else [])
            ][
                :,
                (uni_target_track_info[0] >= 0)
                & (n_matched_target_hits / uni_target_track_info[2] > 0.5),
            ]
            matched_target_particles = torch.unique(matched_target_tracks, dim=1)

            n_particles += len(particles[0])
            n_matched_particles += len(torch.unique(matched_track_particle_id))
            n_matched_tracks += len(matched_track_particle_id)
            n_matched_target_particles += len(matched_target_particles[0])
            n_matched_target_tracks += len(matched_target_tracks[0])
            n_tracks += len(uni_labels[uni_labels >= 0])

            if particles_pt_hist is None:
                particles_pt_hist = np.histogram(
                    particles[1].cpu().numpy(), bins=pt_bins
                )[0]
                matched_target_particles_pt_hist = np.histogram(
                    matched_target_particles[1].cpu().numpy(), bins=pt_bins
                )[0]
                if config.get("plot_eta", True):
                    particles_eta_hist = np.histogram(
                        particles[2].cpu().numpy(), bins=eta_bins
                    )[0]
                    matched_target_particles_eta_hist = np.histogram(
                        matched_target_particles[2].cpu().numpy(), bins=eta_bins
                    )[0]
            else:
                particles_pt_hist += np.histogram(
                    particles[1].cpu().numpy(), bins=pt_bins
                )[0]
                matched_target_particles_pt_hist += np.histogram(
                    matched_target_particles[1].cpu().numpy(), bins=pt_bins
                )[0]
                if config.get("plot_eta", True):
                    particles_eta_hist += np.histogram(
                        particles[2].cpu().numpy(), bins=eta_bins
                    )[0]
                    matched_target_particles_eta_hist += np.histogram(
                        matched_target_particles[2].cpu().numpy(), bins=eta_bins
                    )[0]

        eff = n_matched_target_particles / n_particles
        dup = (
            n_matched_target_tracks - n_matched_target_particles
        ) / n_matched_target_particles
        fak = (n_tracks - n_matched_tracks) / n_matched_particles

        for (
            matched_target_particles_hist,
            particles_hist,
            bins,
            xlabel,
            logx,
            filename,
        ) in zip(
            [matched_target_particles_pt_hist, matched_target_particles_eta_hist],
            [particles_pt_hist, particles_eta_hist],
            [pt_bins, eta_bins],
            [f"$p_T$ [{config.get('pT_unit', 'MeV')}]", r"$\eta$"],
            [True, False],
            ["track_efficiency_pt.png", "track_efficiency_eta.png"],
        ):
            if matched_target_particles_hist is None:
                continue
            hist, err = get_ratio(matched_target_particles_hist, particles_hist)
            if "filename_template" in plot_config:
                filename = config["filename_template"] + "_" + filename

            fig, ax = plot_1d_histogram(
                hist,
                bins,
                err,
                xlabel,
                plot_config["title"],
                plot_config.get("ylim", [0.7, 1.04]),
                "Efficiency",
                logx=logx,
                color="black",
            )

            # Save the plot
            atlasify(
                atlas=True if config.get("trackML_label") else "Internal",
                subtext=base_subtext
                + r"DBSCAN ($\epsilon$"
                + f"={plot_config['eps']}, min_samples=3)"
                + "\n"
                f"Efficiency: {eff :.4f}" + "\n"
                f"Duplication rate: {dup :.4f}" + "\n"
                f"Fake rate: {fak :.4f}" + "\n",
            )
            fig.savefig(os.path.join(config["stage_dir"], filename))

            print(
                "Finish plotting. Find the plot at"
                f' {os.path.join(config["stage_dir"], filename)}'
            )

    def dbscan_vs_eps(self, plot_config, config):

        dataset_name = config["dataset"]
        dataset = getattr(self, dataset_name)

        base_subtext = (
            (
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                r" $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
            )
            if not config.get("trackML_label")
            else r"$p_T > 1$GeV" + "\n"
        )

        epss = np.linspace(0.05, 0.5, 10)
        n_particles = [0] * len(epss)
        n_matched_particles = [0] * len(epss)
        n_matched_tracks = [0] * len(epss)
        n_matched_target_particles = [0] * len(epss)
        n_matched_target_tracks = [0] * len(epss)
        n_tracks = [0] * len(epss)

        for event in tqdm(dataset):
            event = event.to(self.device)

            event.hit_target_mask = self.get_node_target_mask(
                event, config.get("target_tracks", None)
            )
            event_n_particles = len(
                torch.unique(event.hit_particle_id[event.hit_target_mask])
            )
            for i, eps in enumerate(epss):
                n_particles[i] += event_n_particles

                self.cluster(event, eps, 3)
                uni_labels, inv_idx, count = torch.unique(
                    event.hit_label, return_counts=True, return_inverse=True
                )
                event.hit_track_length = count[inv_idx]
                hit_track_info = torch.stack(
                    [event.hit_label, event.hit_particle_id, event.hit_track_length],
                    dim=0,
                )
                uni_track_info, inv_idx, n_matched_hits = torch.unique(
                    hit_track_info, dim=1, return_counts=True, return_inverse=True
                )
                matched_track_particle_id = uni_track_info[1][
                    (uni_track_info[0] >= 0)
                    & (n_matched_hits / uni_track_info[2] > 0.5)
                ]
                hit_target_track_info = torch.stack(
                    [
                        event.hit_label[event.hit_target_mask],
                        event.hit_particle_id[event.hit_target_mask],
                        event.hit_track_length[event.hit_target_mask],
                    ],
                    dim=0,
                )
                uni_target_track_info, inv_idx, n_matched_target_hits = torch.unique(
                    hit_target_track_info,
                    dim=1,
                    return_counts=True,
                    return_inverse=True,
                )
                matched_target_track_particle_id = uni_target_track_info[1][
                    (uni_target_track_info[0] >= 0)
                    & (n_matched_target_hits / uni_target_track_info[2] > 0.5)
                ]

                n_matched_particles[i] += len(torch.unique(matched_track_particle_id))
                n_matched_tracks[i] += len(matched_track_particle_id)
                n_matched_target_particles[i] += len(
                    torch.unique(matched_target_track_particle_id)
                )
                n_matched_target_tracks[i] += len(matched_target_track_particle_id)
                n_tracks[i] += len(uni_labels[uni_labels >= 0])

        n_particles = np.array(n_particles)
        n_matched_particles = np.array(n_matched_particles)
        n_matched_tracks = np.array(n_matched_tracks)
        n_matched_target_particles = np.array(n_matched_target_particles)
        n_matched_target_tracks = np.array(n_matched_target_tracks)
        n_tracks = np.array(n_tracks)

        eff = n_matched_target_particles / n_particles
        dup = (
            n_matched_target_tracks - n_matched_target_particles
        ) / n_matched_target_particles
        fak = (n_tracks - n_matched_tracks) / n_matched_particles

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epss, eff, color="black", marker="o", linestyle=":", label="Efficiency")
        ax.plot(
            epss, dup, color="red", marker="o", linestyle="-.", label="Duplication rate"
        )
        ax.plot(epss, fak, color="blue", marker="o", linestyle="--", label="Fake rate")
        ax.set_xlabel(r"$\epsilon$", ha="right", x=0.95, fontsize=14)
        ax.set_ylabel("Efficiency (Rate)", ha="right", y=0.95, fontsize=14)
        ax.set_ylim([0, 1])
        ax.legend(loc="upper right", fontsize=14)
        plt.tight_layout()

        # Save the plot
        atlasify(
            atlas=True if config.get("trackML_label") else "Internal",
            subtext=base_subtext + "DBSCAN (min_samples = 3)",
        )
        fig.savefig(os.path.join(config["stage_dir"], "traack_eff_dbscan_vs_eps.png"))

        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"], "traack_eff_dbscan_vs_eps.png")}'
        )

    def plot_inference_time(self, plot_config, config):

        eps = plot_config["eps"]

        dataset_name = config["dataset"]
        dataset = getattr(self, dataset_name)

        base_subtext = (
            (
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                r" $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
            )
            if not config.get("trackML_label")
            else r"$p_T > 1$GeV" + "\n"
        )

        ns = []
        ts = []
        knn_ts = []
        dbscan_ts = []
        for event in tqdm(dataset):
            event = event.to(self.device)

            n_spacepoints = len(event.hit_r)
            t = event.inference_time
            knn_t = event.knn_time
            ns.append(n_spacepoints)
            ts.append(t)
            knn_ts.append(knn_t)

            start = time.time()
            self.cluster(event, eps, 3)
            end = time.time()
            dbscan_ts.append(end - start)

        ns = np.array(ns)
        ts = np.array(ts)
        dbscan_ts = np.array(dbscan_ts)
        knn_ts = np.array(knn_ts)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ns, ts + dbscan_ts, "o", label="Total")
        ax.plot(ns, ts - knn_ts, "o", label="Graph attention")
        ax.plot(ns, knn_ts, "o", label="KNN")
        ax.plot(ns, dbscan_ts, "o", label="DBScan")
        ax.set_xlabel("Number of spacepoints", ha="right", x=0.95, fontsize=14)
        ax.set_ylabel("Inference time per event [s]", ha="right", y=0.95, fontsize=14)
        ax.set_ylim([0, 1.8])
        plt.tight_layout()

        # Save the plot
        atlasify(
            atlas=True if config.get("trackML_label") else "Internal",
            subtext=base_subtext,
        )
        fig.savefig(os.path.join(config["stage_dir"], "inference_time.png"))

        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"], "inference_time.png")}'
        )


class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
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
        **kwargs,
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
        self.preprocess_event(event)

        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """

        self.cleaning_and_tests(event)
        self.apply_hard_cuts(event)
        # self.remove_split_cluster_truth(event) TODO: Should handle this at some point
        if self.stage != "test":
            self.scale_features(event)
        if "hit_module_index" in event:
            event["hit_module_id"] = event.pop("hit_module_index")

    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if self.hparams.get("hard_cuts") or (
            self.hparams.get("phi_segmented") and self.data_name == "trainset"
        ):
            hard_cut_finished = False
            while not hard_cut_finished:
                hard_cuts = self.hparams.get("hard_cuts", {})
                if self.hparams.get("phi_segmented") and self.data_name == "trainset":
                    graph_fraction = self.hparams.get("graph_fraction", 0.1)
                    phi_low = math.pi * (2 * random.random() - 1)
                    phi_high = phi_low + math.pi * 2 * graph_fraction
                    phi_high %= math.pi * 2
                    if phi_high > phi_low:
                        hard_cuts["hit_phi"] = [phi_low, phi_high]
                    else:
                        hard_cuts["hit_phi"] = ["not_within", [phi_high, phi_low]]
                hard_cut_finished = handle_hard_node_cuts(
                    event,
                    hard_cuts,
                    self.hparams.get("min_nodes", 20),
                    self.hparams.get("max_nodes"),
                )

            uni, inv_idx, count = torch.unique(
                event.hit_particle_id, return_counts=True, return_inverse=True
            )
            event.hit_particle_nhits = count[inv_idx]

    def cleaning_and_tests(self, event):
        """
        Ensure that data is clean and has the correct shape
        """

        if not hasattr(event, "num_nodes") or event.num_nodes is None:
            assert "hit_x" in event.keys, "No node features found in event"
            event.num_nodes = event.hit_x.shape[0]

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]

    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]

    def handle_edge_list(self, event):
        """
        TODO
        """
        pass
