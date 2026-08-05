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
import csv
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

# Local imports
from ..track_building_stage import TrackBuildingStage
from ..timing import DeviceTimer
from . import fast_walkthrough_utils, cc_and_walk_utils
from . import dwalk_utils


class DWALK(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the DWALK
        """
        self.hparams = hparams
        self.gpu_available = torch.cuda.is_available()
        self.cc_only = self.hparams.get("cc_only", False)

    def move_graph_to_compute_device(self, graph):
        if self.hparams.get("use_gpu", False) and self.gpu_available:
            return graph.to("cuda")
        return graph

    def _build_tracks_one_evt(self, graph, output_dir):
        """
        Build tracks for one event from connected components + walkthrough
        """
        os.sched_setaffinity(0, range(1000))
        working_graph = self.move_graph_to_compute_device(graph)
        total_timer = DeviceTimer(working_graph.edge_index.device)
        total_timer.start()

        all_trks = dict()

        if self.hparams.get("on_true_graph", False):
            score_name = "edge_y"
            threshold = 0
        else:
            score_name = "edge_scores"
            threshold = self.hparams["score_cut_cc"]

        filtered_graph = fast_walkthrough_utils.filter_graph(
            working_graph, score_name, threshold
        )

        filtered_graph = cc_and_walk_utils.remove_cycles(filtered_graph)

        all_trks["cc"], filtered_graph = fast_walkthrough_utils.get_simple_path(
            filtered_graph,
            use_gpu=self.hparams.get("use_gpu", False),
            use_cudf=self.hparams.get("use_cudf", False),
            cc_backend=self.hparams.get("cc_backend", "auto"),
        )

        if not self.cc_only:
            # filtered_graph = self.move_graph_to_compute_device(filtered_graph)
            all_trks["walk"] = dwalk_utils.walk_through(
                filtered_graph,
                score_name,
                self.hparams["score_cut_walk"]["min"],
                self.hparams["score_cut_walk"]["add"],
                self.hparams.get("path_metrics", "length"),
                self.hparams.get("use_gpu", False),
                self.hparams.get("use_cudf", False),
                self.hparams.get("cc_backend", "auto"),
            )
        else:
            if hasattr(filtered_graph, "cached_component_labels"):
                residual_tracks = fast_walkthrough_utils.labels_to_lists(
                    filtered_graph.cached_component_labels,
                    filtered_graph.hit_id,
                    use_cudf=self.hparams.get("use_cudf", False),
                )
                all_trks["walk"] = residual_tracks
        graph.time_taken = total_timer.stop()

        if self.hparams.get("save_graph", True):
            cc_and_walk_utils.add_track_labels(graph, all_trks)

        tracks = cc_and_walk_utils.join_track_lists(all_trks)

        if self.hparams.get("save_tracks", True):
            self.save_tracks(graph, tracks, output_dir)

        if self.hparams.get("save_graph", True):
            graph = self.save_graph(graph, output_dir)

        return graph

    def build_tracks(self, dataset, data_name):
        """
        Explain here the algorithm
        """

        self.log.info("Using FastWalkthrough method to reconstruct the tracks")

        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        self.log.info(f"Saving tracks to {output_dir}")

        max_workers = (
            self.hparams["max_workers"] if "max_workers" in self.hparams else None
        )
        if max_workers != 1:
            process_map(
                partial(self._build_tracks_one_evt, output_dir=output_dir),
                dataset,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Reconstructing tracks for {data_name} data",
            )
        else:
            for event in tqdm(
                dataset, desc=f"Reconstructing tracks for {data_name} data"
            ):
                self._build_tracks_one_evt(event, output_dir=output_dir)
