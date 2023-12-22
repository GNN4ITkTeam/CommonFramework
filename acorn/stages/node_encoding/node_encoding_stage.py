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

sys.path.append("../")

from pytorch_lightning import LightningModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from acorn.utils import (
    get_condition_lambda,
    eval_utils,
    load_datafiles_in_dir,
    handle_hard_node_cuts,
)

from atlasify import atlasify
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
        elif stage == "test":
            self.load_data(self.hparams["stage_dir"], stage)

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
        dataloaders = []
        for i, (data_name, data_num) in enumerate(
            zip(["trainset", "valset", "testset"], self.hparams["data_split"])
        ):
            if data_num > 0:
                dataset = getattr(self, data_name)
                num_workers = (
                    16
                    if (
                        "num_workers" not in self.hparams
                        or self.hparams["num_workers"] is None
                    )
                    else self.hparams["num_workers"][i]
                )
                dataloaders.append(
                    DataLoader(dataset, batch_size=1, num_workers=num_workers)
                )
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
                        torch.full(
                            hit_t.shape, self.hparams["knn_val"], device=self.device
                        ),
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

    def node_knn_eff(self, plot_config, config):
        """
        Plot the graph construction efficiency vs. pT of the edge.
        """

        tp_pt_hist, tp_eta_hist = None, None
        max_tp_pt_hist, max_tp_eta_hist = None, None
        t_pt_hist, t_eta_hist = None, None
        p_pt_hist, p_eta_hist = None, None

        pt_min, pt_max = 1000, 50000
        pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)
        eta_bins = np.linspace(-4, 4)

        tp = 0
        t = 0
        max_tp = 0
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
            event.hit_t, event.max_hit_t = self.get_number_of_true_edges(
                event,
                target="mask-based",
                target_tracks=config.get("target_tracks", None),
                upper_bound=plot_config["knn"],
            )

            event = event.cpu()

            if tp_pt_hist is None:
                tp_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[
                        event.edge_index[1, event.edge_target_mask]
                    ].numpy(),
                    bins=pt_bins,
                )
                tp_eta_hist, _ = np.histogram(
                    event.hit_particle_eta[
                        event.edge_index[1, event.edge_target_mask]
                    ].numpy(),
                    bins=eta_bins,
                )
                max_tp_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.max_hit_t.numpy(),
                )
                max_tp_eta_hist, _ = np.histogram(
                    event.hit_particle_eta[event.hit_target_mask].numpy(),
                    bins=eta_bins,
                    weights=event.max_hit_t.numpy(),
                )
                t_pt_hist, _ = np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.hit_t.numpy(),
                )
                t_eta_hist, _ = np.histogram(
                    event.hit_particle_eta[event.hit_target_mask].numpy(),
                    bins=eta_bins,
                    weights=event.hit_t.numpy(),
                )
                p_pt_hist = (
                    np.histogram(event.hit_particle_pt.numpy(), bins=pt_bins)[0]
                    * plot_config["knn"]
                )
                p_eta_hist = (
                    np.histogram(event.hit_particle_eta.numpy(), bins=eta_bins)[0]
                    * plot_config["knn"]
                )
            else:
                tp_pt_hist += np.histogram(
                    event.hit_particle_pt[
                        event.edge_index[1, event.edge_target_mask]
                    ].numpy(),
                    bins=pt_bins,
                )[0]
                tp_eta_hist += np.histogram(
                    event.hit_particle_eta[
                        event.edge_index[1, event.edge_target_mask]
                    ].numpy(),
                    bins=eta_bins,
                )[0]
                max_tp_pt_hist += np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.max_hit_t.numpy(),
                )[0]
                max_tp_eta_hist += np.histogram(
                    event.hit_particle_eta[event.hit_target_mask].numpy(),
                    bins=eta_bins,
                    weights=event.max_hit_t.numpy(),
                )[0]
                t_pt_hist += np.histogram(
                    event.hit_particle_pt[event.hit_target_mask].numpy(),
                    bins=pt_bins,
                    weights=event.hit_t.numpy(),
                )[0]
                t_eta_hist += np.histogram(
                    event.hit_particle_eta[event.hit_target_mask].numpy(),
                    bins=eta_bins,
                    weights=event.hit_t.numpy(),
                )[0]
                p_pt_hist += (
                    np.histogram(event.hit_particle_pt.numpy(), bins=pt_bins)[0]
                    * plot_config["knn"]
                )
                p_eta_hist += (
                    np.histogram(event.hit_particle_eta.numpy(), bins=eta_bins)[0]
                    * plot_config["knn"]
                )
            tp += event.edge_target_mask.sum().item()
            t += event.hit_t.sum().item()
            max_tp += event.max_hit_t.sum().item()
            p += len(event.hit_particle_pt) * plot_config["knn"]

        for tp_hist, max_tp_hist, t_hist, bins, xlabel, logx, filename in zip(
            [tp_pt_hist, tp_eta_hist],
            [max_tp_pt_hist, max_tp_eta_hist],
            [t_pt_hist, t_eta_hist],
            [pt_bins, eta_bins],
            ["$p_T [MeV]$", r"$\eta$"],
            [True, False],
            ["edgewise_efficiency_pt.png", "edgewise_efficiency_eta.png"],
        ):
            hist, err = get_ratio(tp_hist, t_hist)
            hist_up, _ = get_ratio(max_tp_hist, t_hist)
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
                atlas="Internal",
                subtext=(
                    r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                    r" $t \bar{t}$ and soft interactions) "
                )
                + "\n"
                r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
                f"kNN graph (k={plot_config['knn']})" + "\n"
                f"Global efficiency: {tp / t :.4f}"
                + "\n"
                + f"Efficiency upper bound: {max_tp / t :.4f}",
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
            ["$p_T [MeV]$", r"$\eta$"],
            [True, False],
            ["edgewise_purity_pt.png", "edgewise_purity_eta.png"],
        ):
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
                atlas="Internal",
                subtext=(
                    r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                    r" $t \bar{t}$ and soft interactions) "
                )
                + "\n"
                r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
                f"kNN graph (k={plot_config['knn']})" + "\n"
                f"Global purity: {tp / p :.4f}"
                + "\n"
                + f"Purity upper bound: {max_tp / p :.4f}",
            )
            fig.savefig(os.path.join(config["stage_dir"], filename))

            print(
                "Finish plotting. Find the plot at"
                f' {os.path.join(config["stage_dir"], filename)}'
            )

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

    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if (
            self.hparams is not None
            and "hard_cuts" in self.hparams.keys()
            and self.hparams["hard_cuts"]
        ):
            assert isinstance(
                self.hparams["hard_cuts"], dict
            ), "Hard cuts must be a dictionary"
            handle_hard_node_cuts(event, self.hparams["hard_cuts"])

            uni, inv_idx, count = torch.unique(
                event.hit_particle_id, return_counts=True, return_inverse=True
            )
            event.hit_particle_nhits = count[inv_idx]

    def cleaning_and_tests(self, event):
        """
        Ensure that data is clean and has the correct shape
        """

        if not hasattr(event, "num_nodes"):
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
