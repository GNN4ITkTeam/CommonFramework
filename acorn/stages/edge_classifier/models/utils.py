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
import random

import torch

from tqdm import tqdm
from torch_geometric.data import Dataset
from acorn.utils.version_utils import get_pyg_data_keys

# ---------------------------- Dataset Processing -------------------------


def load_dataset(
    input_subdir="",
    num_events=10,
    pt_background_cut=0,
    pt_signal_cut=0,
    noise=False,
    triplets=False,
    input_cut=None,
    **kwargs,
):
    if input_subdir is not None:
        all_events = os.listdir(input_subdir)
        if "sorted_events" in kwargs.keys() and kwargs["sorted_events"]:
            all_events = sorted(all_events)
        else:
            random.shuffle(all_events)

        all_events = [os.path.join(input_subdir, event) for event in all_events]
        print(f"Loading events from {input_subdir}")

        loaded_events = []
        for event in tqdm(all_events[:num_events]):
            loaded_events.append(torch.load(event, map_location=torch.device("cpu")))

        print("Events loaded!")

        loaded_events = process_data(
            loaded_events, pt_background_cut, pt_signal_cut, noise, triplets, input_cut
        )

        print("Events processed!")
        return loaded_events
    else:
        return None


def process_data(events, pt_background_cut, pt_signal_cut, noise, triplets, input_cut):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if pt_background_cut > 0 or not noise:
        for i, event in tqdm(enumerate(events)):
            if triplets:  # Keep all event data for posterity!
                # event = convert_triplet_graph(event)
                raise NotImplementedError

            else:
                event = background_cut_event(event, pt_background_cut, pt_signal_cut)

    for i, event in tqdm(enumerate(events)):
        # Ensure PID definition is correct
        event.y_pid = (
            event.pid[event.edge_index[0]] == event.pid[event.edge_index[1]]
        ) & event.pid[event.edge_index[0]].bool()
        event.pid_signal = (
            torch.isin(event.edge_index, event.signal_true_edges).all(0) & event.y_pid
        )

        if (input_cut is not None) and "scores" in get_pyg_data_keys(event):
            score_mask = event.scores > input_cut
            for edge_attr in ["edge_index", "y", "y_pid", "pid_signal", "scores"]:
                event[edge_attr] = event[edge_attr][..., score_mask]

    return events


def background_cut_event(event, pt_background_cut=0, pt_signal_cut=0):
    edge_mask = (
        (event.pt[event.edge_index] > pt_background_cut)
        & (event.pid[event.edge_index] == event.pid[event.edge_index])
        & (event.pid[event.edge_index] != 0)
    ).any(0)
    event.edge_index = event.edge_index[:, edge_mask]
    event.y = event.y[edge_mask]

    if "y_pid" in event.__dict__.keys():
        event.y_pid = event.y_pid[edge_mask]

    if "weights" in event.__dict__.keys():
        if event.weights.shape[0] == edge_mask.shape[0]:
            event.weights = event.weights[edge_mask]

    if (
        "signal_true_edges" in event.__dict__.keys()
        and event.signal_true_edges is not None
    ):
        signal_mask = (event.pt[event.signal_true_edges] > pt_signal_cut).all(0)
        event.signal_true_edges = event.signal_true_edges[:, signal_mask]

    return event


class LargeDataset(Dataset):
    def __init__(
        self,
        root,
        subdir,
        num_events,
        hparams,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.subdir = subdir
        self.hparams = hparams

        self.input_paths = os.listdir(os.path.join(root, subdir))
        if "sorted_events" in hparams.keys() and hparams["sorted_events"]:
            self.input_paths = sorted(self.input_paths)
        else:
            random.shuffle(self.input_paths)

        self.input_paths = [
            os.path.join(root, subdir, event) for event in self.input_paths
        ][:num_events]

    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event = torch.load(self.input_paths[idx], map_location=torch.device("cpu"))

        # Process event with pt cuts
        if self.hparams["pt_background_cut"] > 0:
            event = background_cut_event(
                event, self.hparams["pt_background_cut"], self.hparams["pt_signal_cut"]
            )

        # Ensure PID definition is correct
        event.y_pid = (
            event.pid[event.edge_index[0]] == event.pid[event.edge_index[1]]
        ) & event.pid[event.edge_index[0]].bool()
        event.pid_signal = (
            torch.isin(event.edge_index, event.signal_true_edges).all(0) & event.y_pid
        )

        # if ("delta_eta" in self.hparams.keys()) and ((self.subdir == "train") or (self.subdir == "val" and self.hparams["n_graph_iters"] == 0)):
        if "delta_eta" in self.hparams.keys():
            # eta_mask = hard_eta_edge_slice(self.hparams["delta_eta"], event)
            eta_mask = None
            for edge_attr in ["edge_index", "y", "y_pid", "pid_signal", "scores"]:
                if edge_attr in get_pyg_data_keys(event):
                    event[edge_attr] = event[edge_attr][..., eta_mask]

        if (
            ("input_cut" in self.hparams.keys())
            and (self.hparams["input_cut"] is not None)
            and "scores" in get_pyg_data_keys(event)
        ):
            score_mask = event.scores > self.hparams["input_cut"]
            for edge_attr in ["edge_index", "y", "y_pid", "pid_signal", "scores"]:
                if edge_attr in get_pyg_data_keys(event):
                    event[edge_attr] = event[edge_attr][..., eta_mask]

        return event
