import glob
import os
import random
import re
from functools import partial
from itertools import chain
from multiprocessing import get_context
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from ..data_reading_stage import EventReader
from .colliderml_utils import (
    load_explode_parquet,
    validate_file_schema,
    get_tracker_hits_schema,
    get_particles_schema,
)
from acorn.utils.loading_utils import pyg_exists


class ColliderMLReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        """
        Initialize ColliderMLReader for parquet files containing tracker hits and particles data.
        """

        os.makedirs(self.config["stage_dir"], exist_ok=True)

        # Set random seed if provided in config
        if "random_seed" in self.config:
            seed = self.config["random_seed"]
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.log.info(f"Set random seed to {seed}")

        # skip CSV conversion
        self.config["skip_csv_conversion"] = True

        do_discovery = True
        if self.config.get("event_mapping"):
            self.log.info("Event mapping provided. Looking for event map ...")
            event_map_file = os.path.join(
                self.config["stage_dir"], self.config["event_mapping"]
            )
            if os.path.isfile(event_map_file):
                self.event_to_file_map = pd.read_csv(event_map_file)
                self.log.info(f"Found {len(self.event_to_file_map)} events")
                do_discovery = False
            else:
                self.log.info(
                    "Event mapping provided but not found. Will save event mapping after file discovery ..."
                )

        # Explicit directories are required
        hits_dir = self.config["tracker_hits_dir"]
        particles_dir = self.config["particles_dir"]

        tracker_pattern = os.path.join(hits_dir, "**", "*.parquet")
        tracker_files = sorted(glob.glob(tracker_pattern, recursive=True))
        particles_pattern = os.path.join(particles_dir, "**", "*.parquet")
        particles_files = sorted(
            f
            for f in glob.glob(particles_pattern, recursive=True)
            if "tracker_hits" not in f
        )

        self.files = {
            "tracker_hits": sorted(tracker_files),
            "particles": sorted(particles_files),
        }

        if do_discovery:
            self.event_to_file_map = self._build_event_file_mapping(self.files)

        # Build raw events list from available event IDs
        all_event_ids = sorted(self.event_to_file_map["event_id"].values)

        # max_events caps the pool before splitting
        if "max_events" in self.config:
            all_event_ids = all_event_ids[: self.config["max_events"]]
            self.log.info(
                f"max_events={self.config['max_events']}: using {len(all_event_ids)} events"
            )

        # Split events into train/val/test
        data_split = self.config.get("data_split", [0.8, 0.1, 0.1])
        assert isinstance(data_split, list) and len(data_split) == 3

        num_events = sum(data_split)
        if all(0 < s < 1 for s in data_split):
            assert (
                sum(data_split) == 1
            ), f"If ratios are given as data_split, they must sum to 1. Got {sum(data_split)}"
            num_events = len(all_event_ids)
            data_split = np.floor(
                np.array([ratio * num_events for ratio in data_split])
            ).astype(np.int64)
            num_events = np.sum(data_split)
            self.log.info(
                f"Requested {data_split[0]} train events, {data_split[1]} val events, and {data_split[2]} test events. Totalling {num_events} events"
            )

        if num_events > len(all_event_ids):
            self.log.warning(
                f"Requested {num_events} events but only {len(all_event_ids)} available"
            )
            num_events = len(all_event_ids)
        self.trainset, self.valset, self.testset = random_split(
            all_event_ids[:num_events], data_split
        )

        split_df = (
            pd.DataFrame(
                {
                    "event_id": list(self.trainset)
                    + list(self.valset)
                    + list(self.testset),
                    "dataset": (
                        ["trainset"] * len(self.trainset)
                        + ["valset"] * len(self.valset)
                        + ["testset"] * len(self.testset)
                    ),
                }
            )
            .sort_values("event_id")
            .reset_index(drop=True)
        )

        split_csv = os.path.join(self.config["stage_dir"], "dataset_split.csv")
        if os.path.isfile(split_csv):
            prev_split_df = (
                pd.read_csv(split_csv).sort_values("event_id").reset_index(drop=True)
            )
            if not split_df.equals(prev_split_df):
                raise RuntimeError(
                    f"Split has changed from the previous run. "
                    f"Delete {self.config['stage_dir']}/trainset, valset, and testset "
                    f"(and {split_csv}) before re-running with a different split."
                )

        split_df.to_csv(split_csv, index=False)

        if "module_columns" not in self.config:
            self.config["module_columns"] = []

        # When use_true_positions is set, guarantee hit_x/y/z are stored
        if self.config.get("use_true_positions"):
            hit_features = self.config["feature_sets"]["hit_features"]
            for coord in ["hit_x", "hit_y", "hit_z"]:
                if coord not in hit_features:
                    hit_features.append(coord)
            self.log.info(
                "use_true_positions=True: hit_x/y/z will be populated from true_x/y/z"
            )

    def get_file_names(self, inputdir, filename_terms: Union[str, list] = None):
        """
        Takes a list of filename terms and searches for all files containing those terms AND a number. Returns the files and numbers.
        For the list of numbers, search for each of the matching terms and files containing that number AND ONLY THAT NUMBER.
        """
        self.log.info("Getting input file names")
        if isinstance(filename_terms, str):
            filename_terms = [filename_terms]
        elif filename_terms is None:
            filename_terms = ["*"]

        all_files_in_template = [
            glob.glob(os.path.join(inputdir, f"*{template}*"))
            for template in filename_terms
        ]
        all_files_in_template = list(chain.from_iterable(all_files_in_template))
        all_event_ids = sorted(
            list({re.findall("[0-9]+", file)[-1] for file in all_files_in_template})
        )

        self.log.debug("Loop on all events ids")
        all_events = []
        for event_id in all_event_ids:
            event = {"event_id": event_id}
            for term in filename_terms:
                if template_file := [
                    file
                    for file in all_files_in_template
                    if term in os.path.basename(file)
                    and re.findall("[0-9]+", file)[-1] == event_id
                ]:
                    event[term] = template_file[0]
                else:
                    print(
                        f"Could not find file for term {term} and event id {event_id}"
                    )
                    break
            else:
                all_events.append(event)

        return all_events

    def _build_all_pyg(self, dataset_name):
        """
        Build PyG graphs directly from parquet without CSVs.

        This overrides the base implementation to iterate over event IDs and
        construct graphs one event at a time.
        """
        stage_dir = os.path.join(self.config["stage_dir"], dataset_name)
        os.makedirs(stage_dir, exist_ok=True)

        if dataset_name == "trainset":
            dataset = self.trainset
        elif dataset_name == "valset":
            dataset = self.valset
        elif dataset_name == "testset":
            dataset = self.testset
        else:
            self.log.warning(f"Unknown dataset name {dataset_name}")
            return

        if dataset is None:
            self.log.warning(f"No dataset available for {dataset_name}")
            return

        # Sort by file so consecutive events share EOS-cached parquet files.
        file_for_event = dict(
            zip(
                self.event_to_file_map["event_id"],
                self.event_to_file_map["tracker_hits"],
            )
        )
        dataset = sorted(dataset, key=lambda e: file_for_event[e])

        max_workers = self.config.get("max_workers", 1)
        if max_workers != 1:
            with get_context("spawn").Pool(max_workers) as pool:
                pool.map(
                    partial(self._build_single_pyg_event, output_dir=stage_dir),
                    tqdm(dataset, desc=f"Building {dataset_name} graphs (parquet)"),
                    chunksize=1,
                )
        else:
            for event in tqdm(
                dataset, desc=f"Building {dataset_name} graphs (parquet)"
            ):
                self._build_single_pyg_event(event, output_dir=stage_dir)

    def _build_single_pyg_event(self, event_id, output_dir=None):
        """
        Load a single event from parquet and construct a PyG graph.

        NOTE: Physics-specific feature selection/engineering should be added by the user.
        This scaffolding loads hits and particles, applies minimal transformations,
        and leverages the existing graph-building utilities from the base class.
        """
        # Ensure separate CPU affinity for multiprocessing
        os.sched_setaffinity(0, range(1000))

        graph_path = os.path.join(
            output_dir, f"{self.event_prefix}event{event_id}-graph.pyg"
        )
        if pyg_exists(graph_path):
            if not self.config.get("overwrite"):
                self.log.info(f"Graph {event_id} already exists, skipping...")
                return
            else:
                self.log.info(f"Graph {event_id} already exists, overwriting...")

        # Load per-event hits and particles from files
        file_info = self.event_to_file_map.query(f"event_id=={event_id}")
        if file_info is None:
            self.log.warning(f"No files mapped for event {event_id}")
            return

        tracker_hits_schema = get_tracker_hits_schema()
        particles_schema = get_particles_schema()

        validate_file_schema(
            file_info["tracker_hits"].values[0], tracker_hits_schema, event_id
        )
        validate_file_schema(
            file_info["particles"].values[0], particles_schema, event_id
        )

        hits_df = load_explode_parquet(file_info["tracker_hits"].values[0], event_id)
        particles_df = load_explode_parquet(file_info["particles"].values[0], event_id)

        if hits_df is None or len(hits_df) == 0:
            self.log.warning(f"Empty hits for event {event_id}, skipping")
            return
        if particles_df is None or len(particles_df) == 0:
            self.log.warning(f"Empty particles for event {event_id}, skipping")
            return

        if "hit_id" not in hits_df.columns:
            hits_df["hit_id"] = np.arange(len(hits_df), dtype=np.int64)

        # Use MC-truth positions instead of digitised ones when requested.
        if self.config.get("use_true_positions"):
            hits_df["x"] = hits_df["true_x"]
            hits_df["y"] = hits_df["true_y"]
            hits_df["z"] = hits_df["true_z"]

        particles_df = particles_df[
            particles_df["particle_id"].isin(hits_df["particle_id"])
        ].copy()

        if len(particles_df) == 0:
            self.log.warning(f"No particles with hits for event {event_id}, skipping")
            return

        # Data quality check: hits referencing a particle_id absent from the particles
        # file indicate a ColliderML data inconsistency. Drop them rather than letting
        # NaN particle features propagate and break downstream assertions.
        unmatched_mask = ~hits_df["particle_id"].isin(particles_df["particle_id"]) & (
            hits_df["particle_id"] != 0
        )
        if unmatched_mask.any():
            bad_pids = hits_df.loc[unmatched_mask, "particle_id"].unique()
            self.log.warning(
                f"Data quality issue in event {event_id}: {unmatched_mask.sum()} hit(s) "
                f"reference particle_id(s) {bad_pids.tolist()} absent from the "
                f"particles file. Dropping these hits."
            )
            hits_df = hits_df[~unmatched_mask].copy()

        hits_df = self._select_hit_features(hits_df)
        particles_df = self._select_particle_features(particles_df)

        success = self._build_single_pyg_from_df(
            particles_df, hits_df, event_id, output_dir
        )
        if not success:
            self.log.warning("Found issue in building true tracks... skipping event")
            return

    def _select_hit_features(self, hits_df: pd.DataFrame) -> pd.DataFrame:
        # unsigned int columns cannot be serialized by torch.save — cast to int32
        for col in hits_df.select_dtypes(include=["uint8", "uint16", "uint32"]).columns:
            hits_df[col] = hits_df[col].astype("int32")

        # Map ColliderML detector code (0-8) to hardware type, barrel/endcap, and region.
        # Detector codes: Pixel=0,1,2 | Short strip=3,4,5 | Long strip=6,7,8
        # Barrel: 1, 4, 7 | Endcap: 0, 2, 3, 5, 6, 8
        # Short strips are merged into PIXEL (no stereo modules)
        # ColliderML does not distinguish ±z endcaps, so both sides map to barrel_endcap=2.
        # Region integers follow the TrackML/Athena convention:
        #   1=pixel endcap, 2=strip endcap, 3=pixel barrel, 4=strip barrel
        hits_df["hardware"] = "PIXEL"  # pixel + short strip
        hits_df.loc[
            hits_df["detector"].isin([6, 7, 8]), "hardware"
        ] = "STRIP"  # long strip only

        hits_df["barrel_endcap"] = 0  # 0 = barrel
        hits_df.loc[hits_df["detector"].isin([0, 2, 3, 5, 6, 8]), "barrel_endcap"] = 2

        hw, be = hits_df["hardware"], hits_df["barrel_endcap"]
        hits_df["region"] = 0
        hits_df.loc[(hw == "PIXEL") & (be != 0), "region"] = 1
        hits_df.loc[(hw == "STRIP") & (be != 0), "region"] = 2
        hits_df.loc[(hw == "PIXEL") & (be == 0), "region"] = 3
        hits_df.loc[(hw == "STRIP") & (be == 0), "region"] = 4

        # Compute ACTS-style geometry ID: (volume_id << 48) | (layer_id << 32) | surface_id
        # This matches the module IDs used by the module map generator (MMG).
        vol = hits_df["volume_id"].values.astype(np.int64)
        lay = hits_df["layer_id"].values.astype(np.int64)
        surf = hits_df["surface_id"].values.astype(np.int64)
        hits_df["module_id"] = (vol << 48) | (lay << 32) | surf

        return hits_df

    def _select_particle_features(self, particles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select particle features from the particles dataframe.
        """

        # Add pt to particles
        particles_df["pt"] = np.sqrt(particles_df["px"] ** 2 + particles_df["py"] ** 2)

        # in case pt is really small, can cause eta to blow up. Add a small component to pT to make it numerically non-zero
        epsilon = 1e-4
        particles_df["pt"] = np.where(
            particles_df["pt"] > epsilon,
            particles_df["pt"],
            np.ones_like(particles_df["pt"]) * epsilon,
        )

        # Add eta to particles — clip cos(theta) away from ±1 to avoid arctanh(±1)=±inf
        cos_theta = particles_df["pz"] / np.sqrt(
            particles_df["pt"] ** 2 + particles_df["pz"] ** 2
        )
        particles_df["eta"] = np.arctanh(np.clip(cos_theta, -1 + 1e-7, 1 - 1e-7))

        # Use the pre-computed perigee parameters; fill NaN (secondaries without
        # proper track params) with 0 so the per-edge consistency check passes.
        particles_df["d0"] = particles_df["perigee_d0"].fillna(0.0)
        particles_df["z0"] = particles_df["perigee_z0"].fillna(0.0)

        return particles_df

    def _discover_parquet_files(self, input_dir: str) -> Dict[str, List[str]]:
        """
        Discover parquet files in the input directory and subdirectories.

        Expected naming patterns:
        - *.tracker_hits.events{start}-{end}.parquet
        - *.particles.events{start}-{end}.parquet

        Returns:
            Dictionary with 'tracker_hits' and 'particles' keys containing file lists
        """
        files = {"tracker_hits": [], "particles": []}

        # Match by top-level directory name since the parquet filenames themselves
        # are generic (e.g. train-00000-of-01000.parquet).
        tracker_pattern = os.path.join(input_dir, "*tracker_hits*", "**", "*.parquet")
        tracker_files = glob.glob(tracker_pattern, recursive=True)
        files["tracker_hits"] = sorted(tracker_files)

        particles_pattern = os.path.join(input_dir, "*particles*", "**", "*.parquet")
        # Exclude any tracker_hits directories that also match "*particles*"
        particles_files = [
            f
            for f in glob.glob(particles_pattern, recursive=True)
            if "tracker_hits" not in f
        ]
        files["particles"] = sorted(particles_files)

        self.log.info(
            f"Found {len(tracker_files)} tracker hits files and {len(particles_files)} particles files"
        )

        if not tracker_files:
            raise ValueError(
                f"No tracker hits parquet files found in {input_dir} or subdirectories"
            )
        if not particles_files:
            raise ValueError(
                f"No particles parquet files found in {input_dir} or subdirectories"
            )

        return files

    def _build_event_file_mapping(self, files: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Build mapping from event ID to containing parquet files.

        Args:
            files: Dictionary of parquet file lists

        Returns:
            Dictionary mapping event_id -> {'tracker_hits': filepath, 'particles': filepath}
        """
        event_map = {}

        self.log.info("Building event map...")

        def parse_event_range(filename):
            return (
                pl.read_parquet(filename, columns=["event_id"])
                .to_numpy()
                .flatten()
                .tolist()
            )

        # Map tracker hits files
        for tracker_file in tqdm(files["tracker_hits"]):
            for event_id in parse_event_range(tracker_file):
                if event_id not in event_map:
                    event_map[event_id] = {}
                event_map[event_id]["tracker_hits"] = tracker_file

        # Map particles files
        for particles_file in tqdm(files["particles"]):
            for event_id in parse_event_range(particles_file):
                if event_id not in event_map:
                    event_map[event_id] = {}
                event_map[event_id]["particles"] = particles_file

        # Filter to events that have both tracker hits and particles
        complete_events = {
            event_id: files
            for event_id, files in event_map.items()
            if "tracker_hits" in files and "particles" in files
        }

        complete_events = [
            {"event_id": key, **files} for key, files in complete_events.items()
        ]

        complete_events = pd.DataFrame(complete_events)

        if self.config.get("event_mapping"):
            stage_dir = self.config["stage_dir"]
            os.makedirs(stage_dir, exist_ok=True)
            event_map_file = os.path.join(stage_dir, self.config["event_mapping"])
            complete_events.to_csv(event_map_file, index=False)
            self.log.info(f"Saved event mapping to {event_map_file}")

        self.log.info(
            f"Found {len(complete_events)} events with both tracker hits and particles data"
        )
        return complete_events
