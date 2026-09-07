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
import warnings
import numpy as np

from ..data_reading_stage import EventReader
from . import athena_utils
from . import athena_root_utils
from .athena_datatypes import SPACEPOINTS_DATATYPES, PARTICLES_DATATYPES
from acorn.utils.loading_utils import pyg_exists


class AthenaRootReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        """
        Here we initialize and load any attributes that are needed for the _build_single_csv function.
        """

        self.log.info("Using AthenaRootReader to read events")

        self.tree_name = "GNN4ITk"
        self.setnames = ["train", "valid", "test"]
        self.fix_index_mismatch = self.config.get(
            "fix_index_mismatch", False
        )  # fix bug only for old dump version

        self.config.setdefault("skip_csv_conversion", True)

        # Get list of all root files in input_dir (sorted)
        input_sets = {
            dataset_name: self.config["input_sets"][f"{dataset_name}"]
            for dataset_name in self.setnames
            if dataset_name in self.config["input_sets"]
        }

        self.root_files = {dataset_name: [] for dataset_name in self.setnames}

        # Make the map of all Athena Event Numbers, TTree entry and file names where we can find them
        # from the input event list txt files
        self.evtsmap = {}

        for dataset_name, evt_list_fname in input_sets.items():
            self.log.info(
                f"Using events listed in {evt_list_fname} for {dataset_name} sample"
            )
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

        if trainset:
            self.log.info(
                "Training events   : {0:>7} -> {1:>7} ({2} evts)".format(
                    trainset[0], trainset[-1], len(trainset)
                )
            )
        if validset:
            self.log.info(
                "Validation events : {0:>7} -> {1:>7} ({2} evts)".format(
                    validset[0], validset[-1], len(validset)
                )
            )
        if testset:
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

        if self.config.get("overlap_sp_cut"):
            self.log.info(
                f"Selecting overlap space points with flag < {self.config.get('overlap_sp_cut')}"
            )

    def _read_event_dataframes(self, event):
        """Read one event from ROOT and build the (detectable_particles, truth)
        dataframes. Shared by the CSV and direct-PyG paths; returns None if the
        event should be skipped."""
        filename = self._resolve_filename(event)
        entry = self.evtsmap[event]["entry"]

        # From the TTree extract numpy arrays of interesting TBranches, only for the desired event number
        with uproot.open(
            filename + ":" + self.tree_name,
            filter_name=athena_root_utils.all_branches,
            library="ak",
        ) as tree:
            self.log.debug(
                f"Opening file {filename} to read entry {entry} corresponding to event"
                f" number {event}"
            )

            # Starting from v5 of dump, we have the branch 'SPisOverlap', together with the fix
            # of SP->cluster indices (not more +1 shift to be corrected)
            if "SPisOverlap" not in tree.keys():
                self.fix_index_mismatch = True
                self.log.info(
                    "You are running on file older than v5, the +1 shift in cluster indices will be corrected"
                )
                # Also remove SPisOverlap from spacepoint_branch_names and truth_col_order
                if "SPisOverlap" in athena_root_utils.spacepoint_branch_names:
                    athena_root_utils.spacepoint_branch_names.remove("SPisOverlap")
                if "SPisOverlap" in athena_root_utils.truth_col_order:
                    athena_root_utils.truth_col_order.remove("SPisOverlap")

            if "SPisOverlap" not in tree.keys() and self.config.get("overlap_sp_cut"):
                raise ValueError(
                    "Error, we will try to cut on the overlap space points flag while it is absent in the TTree!"
                )

            # Get the dict of np arrays corresponding the the wished TBranches, only for the desired event number
            part_branches = tree.arrays(
                athena_root_utils.particle_branch_names,
                entry_start=entry,
                entry_stop=(entry + 1),
                library="ak",
            )
            part_branches = {k: part_branches[k] for k in part_branches.fields}
            self.log.debug("Particles branches read")
            sp_branches = tree.arrays(
                athena_root_utils.spacepoint_branch_names,
                entry_start=entry,
                entry_stop=(entry + 1),
                library="ak",
            )
            sp_branches = {k: sp_branches[k] for k in sp_branches.fields}
            self.log.debug("Space points branches read")
            cl_branches = tree.arrays(
                athena_root_utils.cluster_branch_names,
                entry_start=entry,
                entry_stop=(entry + 1),
                library="ak",
            )
            cl_branches = {k: cl_branches[k] for k in cl_branches.fields}
            self.log.debug("Clusters branches read")

            # Read particles
            particles = athena_root_utils.read_particles(part_branches)
            if particles is None or len(particles) == 0:
                warnings.warn(f"No particles found in event number {event}")
                return None
            particles = athena_utils.convert_barcodes(particles)
            particles = particles.astype(
                {k: v for k, v in PARTICLES_DATATYPES.items() if k in particles.columns}
            )
            self.log.debug("Particles data frame made")

            # Read spacepoints
            spacepoints = athena_root_utils.read_spacepoints(
                sp_branches, self.config.get("overlap_sp_cut", 999)
            )
            # At least 2 spacepoints are needed to build at least one edge
            if len(spacepoints) < 2:
                self.log.warn(
                    f"Not enough spacepoints ({len(spacepoints)}) found in event number {event}"
                )
                return None

            self.log.debug("Space points data frame made")
            if self.log.getEffectiveLevel() == logging.DEBUG:
                print("\nSpace points\n")
                print(spacepoints)
                print(spacepoints.dtypes)

            # Read clusters
            clusters = athena_root_utils.read_clusters(
                cl_branches, particles, self.fix_index_mismatch
            )
            self.log.debug("Clusters data frame made")
            if self.log.getEffectiveLevel() == logging.DEBUG:
                print("\nClusters\n")
                print(clusters)
                print(clusters.dtypes)

            # Get detectable particles
            detectable_particles = athena_utils.get_detectable_particles(
                particles, clusters
            )
            if len(detectable_particles) == 0:
                self.log.warn(f"No detectable particles found in event {event}")
                return None
            self.log.debug("Detectable particles data frame made")

            # Get truth spacepoints
            truth = athena_utils.get_truth_spacepoints(
                spacepoints,
                clusters,
                SPACEPOINTS_DATATYPES,
                self.config.get("phi_overlap_sp_only_same_petal", True),
            )
            truth = athena_utils.remove_undetectable_particles(
                truth, detectable_particles
            )
            truth = athena_utils.add_region_labels(truth, self.config["region_labels"])
            truth = athena_utils.add_module_id(truth, self.module_lookup)
            
            # Force module id to be int64, as it uint64 cannot be serialized on torch 2.4.0
            truth["module_id"] = truth["module_id"].astype(np.int64)

            # To ease validation when comapring to txt reading, re-order to get same columns ordering
            truth = truth[athena_root_utils.truth_col_order]
            detectable_particles = detectable_particles[
                athena_root_utils.particles_col_order
            ]
            if "track_particle_phi" in self.config["feature_sets"]["track_features"]:
                print("Calculating track_particle_phi for detectable particles")
                detectable_particles["phi"] = np.arctan2(
                    detectable_particles["py"], detectable_particles["px"]
                )

            if "track_particle_d0" in self.config["feature_sets"]["track_features"]:
                print("Calculating track_particle_d0 for detectable particles")
                vx = detectable_particles["vx"]
                vy = detectable_particles["vy"]
                px = detectable_particles["px"]
                py = detectable_particles["py"]
                pt = detectable_particles["pt"]
                q = detectable_particles["charge"]
                B = 2.0  # Tesla

                cx = vx + np.sign(q) * py / (0.3 * B * np.abs(q))
                cy = vy - np.sign(q) * px / (0.3 * B * np.abs(q))
                R = pt / (B * 0.3 * np.abs(q))

                detectable_particles["d0"] = np.sqrt((cx) ** 2 + (cy) ** 2) - R

            if "track_particle_z0" in self.config["feature_sets"]["track_features"]:
                print("Calculating track_particle_z0 for detectable particles")
                vx = detectable_particles["vx"]
                vy = detectable_particles["vy"]
                vz = detectable_particles["vz"]
                px = detectable_particles["px"]
                py = detectable_particles["py"]
                pz = detectable_particles["pz"]
                pt = detectable_particles["pt"]
                q = detectable_particles["charge"]
                B = 2.0  # Tesla

                cx = vx + np.sign(q) * py / (0.3 * B * np.abs(q))
                cy = vy - np.sign(q) * px / (0.3 * B * np.abs(q))
                R = pt / (B * 0.3 * np.abs(q))

                phi_v = np.arctan2(py, px)
                phi_c = np.sign(q) * np.pi / 2 + np.arctan2(cy, cx)

                # wrap phi_c to (-pi, pi]
                phi_c = (phi_c + np.pi) % (2 * np.pi) - np.pi

                # align branches of phi_c relative to phi_v
                mask1 = (phi_c > 0) & (phi_c > 1) & (phi_v < 0)
                mask2 = (phi_c < 0) & (phi_c < -1) & (phi_v > 0)

                phi_c = phi_c.copy()
                phi_c[mask1] -= 2 * np.pi
                phi_c[mask2] += 2 * np.pi

                detectable_particles["z0"] = vz - R * np.abs(phi_c - phi_v) * pz / pt

            if self.log.getEffectiveLevel() == logging.DEBUG:
                print("\n*** Truth ***\n")
                print(truth)
                print(truth.dtypes)

                print("\n*** Particles ***\n")
                print(detectable_particles)
                print(detectable_particles.dtypes)

            return detectable_particles, truth

    def _build_single_csv(self, event, output_dir=None):
        # Trick to make all workers are using separate CPUs
        # https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
        os.sched_setaffinity(0, range(1000))

        # Check if file already exists
        if os.path.exists(
            os.path.join(
                output_dir, f"{self.event_prefix}event{event:09}-particles.csv"
            )
        ) and os.path.exists(
            os.path.join(output_dir, f"{self.event_prefix}event{event:09}-truth.csv")
        ):
            print(f"File for event number {event} already exists, skipping...")
            return

        result = self._read_event_dataframes(event)
        if result is None:
            return
        detectable_particles, truth = result

        truth.to_csv(
            os.path.join(
                output_dir, f"{self.event_prefix}event{int(event):09}-truth.csv"
            ),
            index=False,
        )
        detectable_particles.to_csv(
            os.path.join(
                output_dir, f"{self.event_prefix}event{int(event):09}-particles.csv"
            ),
            index=False,
        )
        self.log.debug(f"truth.csv and particles.csv made for event {event}")

    def _build_all_pyg(self, dataset_name):
        # CSVs were requested and written: build graphs from them, as the base class does.
        if not self.config.get("skip_csv_conversion"):
            return super()._build_all_pyg(dataset_name)

        if dataset_name == "trainset":
            dataset = self.trainset
        elif dataset_name == "valset":
            dataset = self.valset
        elif dataset_name == "testset":
            dataset = self.testset
        else:
            self.log.warning(f"Unknown dataset name {dataset_name}")
            return

        if not dataset:
            self.log.warning(f"No dataset available for {dataset_name}")
            return

        stage_dir = os.path.join(self.config["stage_dir"], dataset_name)
        os.makedirs(stage_dir, exist_ok=True)

        groups = self._group_events(dataset)
        self._dispatch_groups(
            groups,
            self._build_single_pyg_event_from_root,
            stage_dir,
            desc=f"Building {dataset_name} graphs",
            max_workers=self.config.get("max_workers", 1),
        )

    def _build_single_pyg_event_from_root(self, event_id, output_dir=None):
        os.sched_setaffinity(0, range(1000))

        event_id_str = f"{int(event_id):09}"

        graph_path = os.path.join(
            output_dir, f"{self.event_prefix}event{event_id_str}-graph.pyg"
        )
        if pyg_exists(graph_path):
            self.log.info(f"Graph {event_id} already exists, skipping...")
            return

        result = self._read_event_dataframes(event_id)
        if result is None:
            return
        detectable_particles, truth = result

        self._build_single_pyg_from_df(
            detectable_particles, truth, event_id_str, output_dir
        )

    def _resolve_filename(self, event):
        # Determine which root file to read given the event number to be processed
        # In case we use files on grid local disk (with xrootd), we provide the full file name (base name otherwise)
        if self.config["input_dir"] == "XROOTD":
            return self.evtsmap[event]["fname"]
        fname = os.path.basename(self.evtsmap[event]["fname"])
        return os.path.join(self.config["input_dir"], fname)

    def _resolve_group_key(self, event):
        # When CSVs were converted first, _build_all_pyg falls back to the
        # base implementation, which groups CSV row dicts (from
        # get_file_names) rather than the raw event numbers in evtsmap --
        # those aren't ours to resolve a filename for.
        if isinstance(event, dict):
            return super()._resolve_group_key(event)
        return self._resolve_filename(event)

    def _group_events(self, dataset):
        """Group events by resolved ROOT file, ordered by file name, so a
        worker that receives a group reads all of its events from the same
        file, regardless of chunksize/worker count. This improves locality on
        network filesystems even without any array-level caching.

        When CSVs were converted first, _build_all_pyg falls back to the base
        implementation and passes CSV row dicts here instead -- those aren't
        grouped by ROOT file, so just keep the base class's grouping order.
        """
        groups = super()._group_events(dataset)
        if dataset and isinstance(dataset[0], dict):
            return groups
        return sorted(groups, key=lambda events: self._resolve_group_key(events[0]))
