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
from __future__ import annotations
import warnings
import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import (
    connected_components,
    min_weight_full_bipartite_matching,
)
from time import process_time


# Local imports
from ..track_building_stage import TrackBuildingStage
from .hgnn_matching_utils import build_truth_bgraph, build_pred_bgraph
from .cc_and_walk_utils import remove_cycles
from acorn.stages.graph_construction.models.utils import graph_intersection
from acorn.utils import (
    load_datafiles_in_dir,
    get_optimizers,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLTrackBuildingStage(TrackBuildingStage, LightningModule):
    def __init__(self, hparams):
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        TrackBuildingStage.__init__(self, hparams, get_logger=False)
        LightningModule.__init__(self)

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if stage in ["fit", "predict"]:
            self.dataset_class = PartialGraphDataset
            self.load_data(stage, self.hparams["input_dir"])
        elif stage == "test":
            torch.manual_seed(0)
            self.load_data(stage, self.hparams["stage_dir"])

    def train_dataloader(self):
        """
        Load the training set.
        """
        if self.trainset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
        return DataLoader(
            self.trainset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=lambda lst: lst[0],
        )

    def val_dataloader(self):
        """
        Load the validation set.
        """
        if self.valset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[1]
        return DataLoader(
            self.valset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=lambda lst: lst[0],
        )

    def test_dataloader(self):
        """
        Load the test set.
        """
        if self.testset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[2]
        return DataLoader(
            self.testset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=lambda lst: lst[0],
            shuffle=False,
        )

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


# ------------- HGNN UTILS ----------------


class PartialData(object):
    """
    A data object that removes isolated hits and select the good tracks
    """

    def __init__(
        self,
        event: Data,
        edge_cut: float = 1e-3,
        clustering_cut: float = 0.2,
        loose_cut: float = 0.01,
        tight_cut: Optional[float] = 0.5,
        min_hits: int = 9,
        clustering_min_hit: int = 3,
        random_drop: float = 0.0,
        signal_weight: float = 2,
        signal_selection: Optional[Dict[str, Tuple[float, float]]] = {
            "pt": [1000, float("inf")],
            "nhits": [3, float("inf")],
        },
        target_selection: Optional[Dict[str, Tuple[float, float]]] = {
            "pt": [500, float("inf")],
            "nhits": [3, float("inf")],
        },
    ):
        """
        Arguments:
            event: a `pytorch_geometric` event record
            edge_cut: the cut that is applied edge-wise
            clustering_cut: the cut to perform clustering
            loose_cut: any hit that is isolated by the cut will be removed
            tight_cut: any connected component that is longer than `min_hits`
                under this cut will be selected and disregarded.
            min_hits: see `tight_cut`
            clustering_min_hit: the minimum number of hits required in a cluster.
            random_drop: randonly drop some edge for data augmentation
            signal_weight: the weight to weigh signals over other positives.
            signal_selection: a dictionary map the name of the feature to the signal selection,
            which is a tuple that is one of the following:
                ("range", (vmin, vmax)): interval of that feature. Use `None` for unbounded.
                ("isin", [allowed values]): is in a set of values
                ("notin", [disallowed values]): is not in the set of values
            target_selection: the same idea as selection but this will REMOVE the tracks from consideration.
        """
        self.device = "cpu"

        self.full_event = event
        self.edge_cut = edge_cut
        self.clustering_cut = clustering_cut
        self.loose_cut = loose_cut
        self.tight_cut = tight_cut
        self.min_hits = min_hits
        self.clustering_min_hit = clustering_min_hit
        self.random_drop = random_drop
        self.signal_weight = signal_weight
        self.signal_selection = signal_selection
        self.target_selection = target_selection
        self.time_taken = process_time()

        self._partial_event = self._preprocess_event()
        self.time_taken = process_time() - self.time_taken
        self.truth_bgraph, self._truth_info = self._build_truth()

    def _preprocess_event(self) -> Data:
        """
        everything is done on cpu then moved to the device!
        """
        event = self.full_event.cpu()
        event = remove_cycles(event)

        # Initialize the array to track which hit to keep
        to_keep = torch.zeros_like(event.hit_id, dtype=torch.bool)

        # Filter out noise hit with loose cut
        edge_mask = torch.rand(event.edge_index.shape[1]) >= self.random_drop
        loose_edges = event.edge_index[:, (event.scores > self.loose_cut) & edge_mask]
        to_keep[loose_edges.unique()] = True

        # Select good tracks with at least `min_hits` hits
        if self.tight_cut:
            # run cc
            tight_edges = event.edge_index[
                :, (event.scores > self.tight_cut) & edge_mask
            ]
            graph = to_scipy_sparse_matrix(
                tight_edges, num_nodes=event.x.shape[0]
            ).tocsr()
            _, track_id = connected_components(graph, directed=False)

            # remove short tracks
            _, inverse, nhits = np.unique(
                track_id, return_counts=True, return_inverse=True
            )
            track_id[nhits[inverse] < self.min_hits] = -1

            # remove any not simple tracks
            out_hit_id, out_degree = np.unique(
                tight_edges[0], return_counts=True, return_inverse=False
            )
            in_hit_id, in_degree = np.unique(
                tight_edges[1], return_counts=True, return_inverse=False
            )
            track_id[np.isin(track_id, track_id[in_hit_id[in_degree > 1]])] = -1
            track_id[np.isin(track_id, track_id[out_hit_id[out_degree > 1]])] = -1

            # store the good tracks and update tracker
            track_id = torch.as_tensor(track_id, device=self.device)
            self.cc_tracks = torch.stack(
                [
                    torch.arange(track_id.shape[0])[track_id >= 0],
                    track_id[track_id >= 0],
                ]
            ).to(self.device)
            to_keep[track_id >= 0] = False

        # perform clustering
        if self.clustering_cut:
            # Apply the score cut
            cluster_edges = event.edge_index[
                :, (event.scores > self.clustering_cut) & edge_mask
            ]

            # Compute in and out degrees
            out_hit_id, out_degree = torch.unique(
                cluster_edges[0], return_counts=True, return_inverse=False
            )
            in_hit_id, in_degree = torch.unique(
                cluster_edges[1], return_counts=True, return_inverse=False
            )

            # Masking out the junctions
            mask = torch.isin(
                cluster_edges[0], out_hit_id[out_degree <= 1]
            ) | torch.isin(cluster_edges[1], out_hit_id[out_degree <= 1])
            cluster_edges = cluster_edges[:, mask]

            # build csr graph and run cc
            graph = to_scipy_sparse_matrix(cluster_edges, num_nodes=event.x.shape[0])
            _, labels = connected_components(graph, directed=False)
            labels = torch.as_tensor(labels, dtype=torch.long, device=event.x.device)
            _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
            valid_mask = counts[inverse] >= self.clustering_min_hit
            idxs = labels[valid_mask].unique(return_inverse=True)[1]

            # build the cluster
            cluster = torch.stack(
                [torch.arange(valid_mask.shape[0])[valid_mask], idxs],
                dim=0,
            )

        # prepare to mask out the noise hits
        masked_idx = torch.cumsum(to_keep, dim=0) - 1
        masked_idx[~to_keep] = -1

        # build new `Data` instance
        edge_mask = (
            to_keep[event.edge_index].all(0)
            & edge_mask
            & (event.scores > self.edge_cut)
        )
        track_edge_mask = to_keep[event.track_edges].all(0)
        partial_event = Data(
            x=event.x[to_keep],
            hit_id=event.hit_id[to_keep],
            z=event.z[to_keep],
            r=event.r[to_keep],
            phi=event.phi[to_keep],
            eta=event.eta[to_keep],
            edge_index=masked_idx[event.edge_index[:, edge_mask]],
            y=event.y[edge_mask],
            track_edges=masked_idx[event.track_edges[:, track_edge_mask]],
            eta_particle=event.eta_particle[track_edge_mask],
            radius=event.radius[track_edge_mask],
            nhits=event.nhits[track_edge_mask],
            particle_id=event.particle_id[track_edge_mask],
            pt=event.pt[track_edge_mask],
            pdgId=event.pdgId[track_edge_mask],
            primary=event.primary[track_edge_mask],
        )

        if self.clustering_cut:
            cluster = cluster[:, to_keep[cluster[0]]]
            cluster[0] = masked_idx[cluster[0]]
            partial_event.cluster = cluster
        else:
            partial_event.cluster = None

        return partial_event

    def _build_truth(self):
        """
        build truth informaiton
        """
        truth_bgraph, truth_info = build_truth_bgraph(
            self._partial_event,
            signal_selection=self.signal_selection,
            target_selection=self.target_selection,
        )

        pid_truth_graph = (truth_bgraph @ truth_bgraph.T).tocoo()
        self.pid_truth_graph = torch.stack(
            [
                torch.as_tensor(pid_truth_graph.row),
                torch.as_tensor(pid_truth_graph.col),
            ],
            dim=0,
        )

        return truth_bgraph, truth_info

    def fetch_emb_truth(self, edges):
        edges, y = graph_intersection(
            edges,
            self.pid_truth_graph,
            return_pred_to_truth=False,
            return_truth_to_pred=False,
            unique_pred=False,
            unique_truth=True,
        )
        emb_weights = torch.ones(y.shape, device=y.device)
        emb_weights[y] /= y.sum() * 5
        emb_weights[~y] /= (~y).sum() * 5 / 4
        hinge = 2 * y.float() - 1
        return edges, hinge, emb_weights

    def fetch_truth(
        self,
        tracks: torch.Tensor,
        scores: torch.Tensor,
    ):
        """
        Everything should be done on cpu.
        """
        tracks, scores = tracks.cpu(), scores.cpu()
        pred_bgraph, pred_info = build_pred_bgraph(
            self._partial_event, tracks, 0, scores=scores.numpy()
        )
        graph = (self.truth_bgraph.T @ pred_bgraph).tocoo()
        graph.eliminate_zeros()
        graph.data = (
            np.square(graph.data)
            / self._truth_info["nhits"][graph.row].numpy()
            / pred_info["track_size"][graph.col].numpy()
        )
        num_particles, num_tracks = graph.shape
        graph = coo_matrix(
            (
                np.concatenate([graph.data, 1e-12 * np.ones(num_particles)]),
                (
                    np.concatenate([graph.row, np.arange(num_particles)]),
                    np.concatenate(
                        [graph.col, np.arange(num_particles) + num_tracks]
                    ),  # include virtual tracks for each particle
                ),
            ),
            shape=(num_particles, num_particles + num_tracks),
        )
        row, col = min_weight_full_bipartite_matching(graph, maximize=True)
        row, col = row[col < num_tracks], col[col < num_tracks]
        track_to_particle = -torch.ones(
            num_tracks, dtype=torch.long, device=self.device
        )
        track_to_particle[col] = torch.tensor(row, dtype=torch.long, device=self.device)

        input_pred_graph = torch.stack(
            [tracks[0].to(self.device), track_to_particle[pred_info["track_id"]]], dim=0
        )

        relavent_mask = input_pred_graph[1] >= 0

        y = torch.zeros(relavent_mask.sum(), dtype=torch.bool, device=self.device)

        input_truth_graph = self.truth_bgraph.tocoo()
        input_truth_graph = torch.stack(
            [
                torch.tensor(input_truth_graph.row, device=self.device),
                torch.tensor(input_truth_graph.col, device=self.device),
            ],
            dim=0,
        )

        y = graph_intersection(
            input_pred_graph[:, relavent_mask],
            input_truth_graph,
            return_pred_to_truth=False,
            return_truth_to_pred=False,
            unique_pred=True,
            unique_truth=True,
        )

        weights = torch.ones(y.shape[0], dtype=torch.float, device=self.device)
        weights[
            self.truth_info["is_signal"][input_pred_graph[1, relavent_mask]]
        ] *= self.signal_weight
        weights[~y] /= weights[~y].sum() * 2
        weights[y] /= weights[y].sum() * 2

        return y, weights, relavent_mask

    def get_tracks(self, tracks: torch.Tensor) -> torch.Tensor:
        tracks = tracks.to(self.device)
        if self.tight_cut and self.cc_tracks.numel() > 0:
            tracks = torch.stack(
                [
                    self.partial_event.hit_id[tracks[0]],
                    tracks[1] + self.cc_tracks[1].max() + 1,
                ],
                dim=0,
            )
            tracks = torch.cat([self.cc_tracks, tracks], dim=1)
        else:
            tracks = tracks.clone()
            tracks[0] = self.partial_event.hit_id[tracks[0]]
        return tracks

    @property
    def partial_event(self):
        return self._partial_event.to(self.device)

    @property
    def truth_info(self):
        return {key: value.to(self.device) for key, value in self._truth_info.items()}

    def to(self, device: torch.device) -> PartialData:
        if self.tight_cut:
            self.cc_tracks = self.cc_tracks.to(device)
        self.pid_truth_graph = self.pid_truth_graph.to(device)
        self.device = device
        return self
