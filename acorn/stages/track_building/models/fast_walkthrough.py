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
from time import process_time

# Local imports
from ..track_building_stage import TrackBuildingStage
from . import fast_walkthrough_utils, cc_and_walk_utils
from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg


class FastWalkthrough(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the FastWalkthrough
        """
        self.hparams = hparams
        self.gpu_available = torch.cuda.is_available()
        self.cc_only = self.hparams.get("cc_only", False)

    def _build_tracks_one_evt(self, graph, output_dir):
        """
        Build tracks for one event from connected components + walkthrough
        """
        os.sched_setaffinity(0, range(1000))
        start_time = process_time()
        all_trks = dict()

        if self.hparams.get("on_true_graph", False):
            score_name = "edge_y"
            threshold = 0
        else:
            score_name = "edge_scores"
            threshold = self.hparams["score_cut_cc"]

        filtered_graph = fast_walkthrough_utils.filter_graph(
            graph, score_name, threshold
        )

        filtered_graph = cc_and_walk_utils.remove_cycles(filtered_graph)

        all_trks["cc"], filtered_graph = fast_walkthrough_utils.get_simple_path(
            filtered_graph
        )

        if not self.cc_only:
            all_trks["walk"] = fast_walkthrough_utils.walk_through(
                filtered_graph,
                score_name,
                self.hparams["score_cut_walk"]["min"],
                self.hparams["score_cut_walk"]["add"],
                self.hparams.get("reuse_hits", False),
            )

        if self.hparams.get("save_graph", True):
            cc_and_walk_utils.add_track_labels(graph, all_trks)

        tracks = cc_and_walk_utils.join_track_lists(all_trks)

        if self.hparams.get("resolve_ambiguities", False) and self.hparams.get(
            "reuse_hits", False
        ):
            tracks = fast_walkthrough_utils.resolve_ambiguities(
                tracks, self.hparams.get("max_ambi_hits", 2)
            )
        graph.time_taken = process_time() - start_time
        if self.hparams.get("save_tracks", True):
            tracks_dir = os.path.join(
                self.hparams["stage_dir"], f"{os.path.basename(output_dir)}_tracks"
            )
            os.makedirs(tracks_dir, exist_ok=True)
            output_file = os.path.join(tracks_dir, f"event{graph.event_id[0]}.txt")
            with open(output_file, "w", newline="") as f:
                csv.writer(f, delimiter=" ").writerows(tracks)

        if self.hparams.get("save_graph", True):
            if not self.hparams.get("variable_with_prefix"):
                graph = remove_variable_name_prefix_in_pyg(graph)
            torch.save(graph, os.path.join(output_dir, f"event{graph.event_id[0]}.pyg"))

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
