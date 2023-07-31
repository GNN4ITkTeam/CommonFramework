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
import uproot
import logging

from ..data_reading_stage import EventReader
from . import athena_utils
from . import athena_root_utils
from .athena_datatypes import SPACEPOINTS_DATATYPES, PARTICLES_DATATYPES


class AthenaRootReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        """
        Here we initialize and load any attributes that are needed for the _build_single_csv function.
        """

        self.log.info("Using AthenaRootReader to read events")

        self.tree_name = "GNN4ITk"
        self.setnames = ["train", "valid", "test"]

        # Get list of all root files in input_dir (sorted)
        input_sets = {
            dataset_name: self.config["input_sets"][f"{dataset_name}"]
            for dataset_name in self.setnames
        }

        self.root_files = {dataset_name: [] for dataset_name in self.setnames}

        # Make the map of all Athena Event Numbers, TTree entry and file names where we can find them
        # from the input event list txt files
        self.evtsmap = {}

        for dataset_name, evt_list_fname in input_sets.items():
            with open(evt_list_fname) as evt_list_file:
                for line in evt_list_file:
                    # we ignore the run number
                    items = line.split()
                    evt = int(items[1])
                    entry = int(items[2])
                    root_fname = str(items[3])
                    self.evtsmap[evt] = {
                        "fname": root_fname,
                        "entry": entry,
                        "dataset_name": dataset_name,
                    }

        # Sanity checks on the sample splitting
        nEvts = len(self.evtsmap)
        print(f"Total number of events : {nEvts}")

        trainset = [e for e, v in self.evtsmap.items() if v["dataset_name"] == "train"]
        validset = [e for e, v in self.evtsmap.items() if v["dataset_name"] == "valid"]
        testset = [e for e, v in self.evtsmap.items() if v["dataset_name"] == "test"]

        self.log.info(
            "Training events   : {0:>7} -> {1:>7} ({2} evts)".format(
                trainset[0], trainset[-1], len(trainset)
            )
        )
        self.log.info(
            "Validation events : {0:>7} -> {1:>7} ({2} evts)".format(
                validset[0], validset[-1], len(validset)
            )
        )
        self.log.info(
            "Test events       : {0:>7} -> {1:>7} ({2} evts)".format(
                testset[0], testset[-1], len(testset)
            )
        )

        if len(trainset) + len(validset) + len(testset) < nEvts:
            raise ValueError("Error in data splitting, we are not using all events!")

        if len(trainset) + len(validset) + len(testset) > nEvts:
            raise ValueError(
                "Error in data splitting, we are trying to use more events than we can!"
            )

        test1 = list(set(trainset) & set(validset))
        test2 = list(set(trainset) & set(testset))
        test3 = list(set(testset) & set(validset))

        if len(test1) != 0 or len(test2) != 0 or len(test3) != 0:
            raise ValueError(
                "Error in data splitting, train/valid/test sets are not independent!"
            )

        # The sets are good, we can use them (list of event numbers)
        self.trainset = trainset
        self.valset = validset
        self.testset = testset
        self.module_lookup = None

    def _build_single_csv(self, event, output_dir=None):
        # Trick to make all workers are using separate CPUs
        # https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
        os.sched_setaffinity(0, range(1000))

        # Check if file already exists
        if os.path.exists(
            os.path.join(output_dir, "event{:09}-particles.csv".format(event))
        ) and os.path.exists(
            os.path.join(output_dir, "event{:09}-truth.csv".format(event))
        ):
            print(f"File for event number {event} already exists, skipping...")
            return

        # Determine which root file and wich TTree entry to read given the event number to be processed
        filename = self.config["input_dir"] + "/" + self.evtsmap[event]["fname"]
        entry = self.evtsmap[event]["entry"]

        # From the TTree extract numpy arrays of interesting TBranches, only for the desired event number
        with uproot.open(
            filename + ":" + self.tree_name,
            filter_name=athena_root_utils.all_branches,
            library="np",
        ) as tree:
            self.log.debug(
                f"Opening file {filename} to read entry {entry} corresponding to event"
                f" number {event}"
            )

            # Get the dict of np arrays corresponding the the wished TBranches, only for the desired event number
            part_branches = tree.arrays(
                athena_root_utils.particle_branch_names,
                entry_start=entry,
                entry_stop=(entry + 1),
                library="np",
            )
            self.log.debug("Particles branches read")
            sp_branches = tree.arrays(
                athena_root_utils.spacepoint_branch_names,
                entry_start=entry,
                entry_stop=(entry + 1),
                library="np",
            )
            self.log.debug("Space points branches read")
            cl_branches = tree.arrays(
                athena_root_utils.cluster_branch_names,
                entry_start=entry,
                entry_stop=(entry + 1),
                library="np",
            )
            self.log.debug("Clusters branches read")

            # Read particles
            particles = athena_root_utils.read_particles(part_branches)
            particles = athena_utils.convert_barcodes(particles)
            particles = particles.astype(
                {k: v for k, v in PARTICLES_DATATYPES.items() if k in particles.columns}
            )
            self.log.debug("Particles data frame made")

            # Read spacepoints
            spacepoints = athena_root_utils.read_spacepoints(sp_branches)
            self.log.debug("Space points data frame made")
            if self.log.getEffectiveLevel() == logging.DEBUG:
                print("\nSpace points\n")
                print(spacepoints)
                print(spacepoints.dtypes)

            # Read clusters
            clusters = athena_root_utils.read_clusters(cl_branches, particles)
            self.log.debug("Clusters data frame made")
            if self.log.getEffectiveLevel() == logging.DEBUG:
                print("\nClusters\n")
                print(clusters)
                print(clusters.dtypes)

            # Get detectable particles
            detectable_particles = athena_utils.get_detectable_particles(
                particles, clusters
            )
            self.log.debug("Detectable particles data frame made")

            # Get truth spacepoints
            truth = athena_utils.get_truth_spacepoints(
                spacepoints, clusters, SPACEPOINTS_DATATYPES
            )
            truth = athena_utils.remove_undetectable_particles(
                truth, detectable_particles
            )
            truth = athena_utils.add_region_labels(truth, self.config["region_labels"])
            truth = athena_utils.add_module_id(truth, self.module_lookup)

            # To ease validation when comapring to txt reading, re-order to get same columns ordering
            truth = truth[athena_root_utils.truth_col_order]
            detectable_particles = detectable_particles[
                athena_root_utils.particles_col_order
            ]

            # Save to CSV
            truth.to_csv(
                os.path.join(output_dir, "event{:09}-truth.csv".format(int(event))),
                index=False,
            )
            detectable_particles.to_csv(
                os.path.join(output_dir, "event{:09}-particles.csv".format(int(event))),
                index=False,
            )
            self.log.debug(f"truth.csv and particles.csv made for event {event}")

            if self.log.getEffectiveLevel() == logging.DEBUG:
                print("\n*** Truth ***\n")
                print(truth)
                print(truth.dtypes)

                print("\n*** Particles ***\n")
                print(detectable_particles)
                print(detectable_particles.dtypes)
