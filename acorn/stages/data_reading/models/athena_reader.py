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
from torch.utils.data import random_split
import torch
import pandas as pd
import logging

from ..data_reading_stage import EventReader
from . import athena_utils
from .athena_datatypes import SPACEPOINTS_DATATYPES, PARTICLES_DATATYPES


class AthenaReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        """
        Here we initialize and load any attributes that are needed for the _build_single_csv function.
        """

        self.log.info("Using AthenaReader to read events")

        input_dir = self.config["input_dir"]
        self.raw_events = self.get_file_names(
            input_dir, filename_terms=["clusters", "particles", "spacepoints"]
        )

        # Very opinionated: We split the data by 80/10/10: train/val/test
        torch.manual_seed(42)  # We want the same split every time for convenience
        data_split = self.config.get("data_split", [0.8, 0.1, 0.1])
        self.trainset, self.valset, self.testset = random_split(
            self.raw_events,
            [
                int(len(self.raw_events) * data_split[0]),
                int(len(self.raw_events) * data_split[1]),
                int(len(self.raw_events) * data_split[2]),
            ],
        )
        self.module_lookup = self.get_module_lookup()

    def get_module_lookup(self):
        # Let's get the module lookup
        names = [
            "hardware",
            "barrel_endcap",
            "layer_disk",
            "eta_module",
            "phi_module",
            "centerMod_z",
            "centerMod_x",
            "centerMod_y",
            "ID",
            "side",
        ]
        module_lookup = pd.read_csv(
            self.config["module_lookup_path"], sep=" ", names=names, header=None
        )
        module_lookup = module_lookup.drop_duplicates()
        return module_lookup[module_lookup.side == 0]  # .copy() ??

    def _build_single_csv(self, event, output_dir=None):
        os.sched_setaffinity(0, range(1000))
        clusters_file = event["clusters"]
        particles_file = event["particles"]
        spacepoints_file = event["spacepoints"]
        event_id = event["event_id"]

        # Check if file already exists
        if os.path.exists(
            os.path.join(output_dir, "event{:09}-particles.csv".format(int(event_id)))
        ) and os.path.exists(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event_id)))
        ):
            print(f"File {event_id} already exists, skipping...")
            return

        # Read particles
        particles = athena_utils.read_particles(particles_file)
        particles = athena_utils.convert_barcodes(particles)
        particles = particles.astype(
            {k: v for k, v in PARTICLES_DATATYPES.items() if k in particles.columns}
        )

        # Read spacepoints
        spacepoints = athena_utils.read_spacepoints(spacepoints_file)
        if self.log.getEffectiveLevel() == logging.DEBUG:
            print("\nSpace points\n")
            print(spacepoints)
            print(spacepoints.dtypes)

        # Read clusters
        clusters, self.shape_list = athena_utils.read_clusters(
            clusters_file, particles, self.config["column_lookup"]
        )
        if self.log.getEffectiveLevel() == logging.DEBUG:
            print("\nClusters\n")
            print(clusters)
            print(clusters.dtypes)

        # Get detectable particles
        detectable_particles = athena_utils.get_detectable_particles(
            particles, clusters
        )

        # Get truth spacepoints
        truth = athena_utils.get_truth_spacepoints(
            spacepoints, clusters, SPACEPOINTS_DATATYPES
        )
        truth = athena_utils.remove_undetectable_particles(truth, detectable_particles)
        truth = athena_utils.add_region_labels(truth, self.config["region_labels"])
        truth = athena_utils.add_module_id(truth, self.module_lookup)

        # Save to CSV
        truth.to_csv(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event_id))),
            index=False,
        )
        detectable_particles.to_csv(
            os.path.join(output_dir, "event{:09}-particles.csv".format(int(event_id))),
            index=False,
        )

        if self.log.getEffectiveLevel() == logging.DEBUG:
            print("\n*** Truth ***\n")
            print(truth)
            print(truth.dtypes)

            print("\n*** Particles ***\n")
            print(detectable_particles)
            print(detectable_particles.dtypes)
