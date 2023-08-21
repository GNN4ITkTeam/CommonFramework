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
import numpy as np
from torch.utils.data import random_split
import trackml.dataset

from ..data_reading_stage import EventReader
from . import trackml_utils


class TrackMLReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        """
        Here we initialize and load any attributes that are needed for the _build_single_csv function.
        """

        input_dir = self.config["input_dir"]
        self.raw_events = self.get_file_names(
            input_dir, filename_terms=["truth", "hits", "particles", "cells"]
        )

        # Very opinionated: We split the data by 80/10/10: train/val/test
        self.trainset, self.valset, self.testset = random_split(
            self.raw_events,
            [
                int(len(self.raw_events) * 0.8),
                int(len(self.raw_events) * 0.1),
                len(self.raw_events)
                - int(len(self.raw_events) * 0.8)
                - int(len(self.raw_events) * 0.1),
            ],
        )
        self.get_detector()
        self.get_module_lookup()

    def get_detector(self):
        # Let's get the detector
        self.detector, self.detector_dims = trackml_utils.load_detector(
            self.config["detector_path"]
        )

    def get_module_lookup(self):
        # Let's get the module lookup
        self.module_lookup = self.detector.reset_index()[
            ["index"] + self.config["module_columns"]
        ].rename(columns={"index": "module_index"})

    def _build_single_csv(self, event, output_dir=None):
        truth_file = event["truth"]
        # hits_file = event["hits"]
        # particles_file = event["particles"]
        # cells_file = event["cells"]
        event_id = event["event_id"]

        # Check if file already exists
        if os.path.exists(
            os.path.join(output_dir, "event{:09}-particles.csv".format(int(event_id)))
        ) and os.path.exists(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event_id)))
        ):
            print(f"File {event_id} already exists, skipping...")
            return

        # Read all dataframes
        path, filename = os.path.split(truth_file)
        event_path = os.path.join(path, filename.split("-")[0])
        particles, hits, cells, truth = trackml.dataset.load_event(
            event_path, parts=["particles", "hits", "cells", "truth"]
        )

        # Merge hits and truth, and add module_id, region_id and cell information
        truth = self._process_truth(truth, hits, cells)
        particles = self._process_particles(particles)

        # Save to CSV
        truth.to_csv(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event_id))),
            index=False,
        )
        particles.to_csv(
            os.path.join(output_dir, "event{:09}-particles.csv".format(int(event_id))),
            index=False,
        )

    def _process_truth(self, truth, hits, cells):
        # Merge hits and truth
        truth = truth.merge(hits, on="hit_id", how="left")

        # Add module_id and region_id
        truth = truth.merge(
            self.module_lookup, on=self.config["module_columns"], how="left"
        )
        truth = trackml_utils.add_region_labels(truth, self.config["region_labels"])

        # Add cell information
        truth = trackml_utils.add_cell_info(truth, cells, self.detector_dims)

        # Remap truth hit_id from [1, ...] to [0, ...]
        truth["hit_id"] = truth["hit_id"] - 1

        return truth

    def _process_particles(self, particles):
        """
        Calculate the radius and pT of each particle
        """

        # Calculate the radius of the particle
        particles["radius"] = np.sqrt(particles["vx"] ** 2 + particles["vy"] ** 2)

        # Calculate the pT of the particle
        particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)

        return particles
