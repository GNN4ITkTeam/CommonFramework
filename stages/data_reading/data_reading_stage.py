import os

import numpy as np
import pandas as pd
import yaml
from typing import Union
import glob
import re
from itertools import chain, product
from torch_geometric.data import Data
import torch

class EventReader:
    """ 
    A general class for reading simulated particle collision events from a set of files. Several convenience utilities are built in,
    and conversion to CSV and Pytorch Geometric data objects is enforced. However the reading of the input files is left up to the user.
    It is expected that general usage is, e.g.
    AthenaReader(path/to/files), which consists of:
    1. Raw files -> CSV
    2. CSV -> PyG data objects    
    """
    def __init__(self, config):
        self.files = None
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            self.config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            raise NotImplementedError

    def infer(self):
        """ 
        The gateway for the inference stage. This class method is called from the infer_stage.py script.
        It assumes a set of basic steps. These are:
        1. Convert to CSV - This should be implemented by the user
        2. Convert to PyG
        """
        
        self.convert_to_csv()
        self._test_csv_conversion()
        self._convert_to_pyg()

    def convert_to_csv(self):
        raise NotImplementedError

    def _convert_to_pyg(self):

        self.csv_events = self.get_file_names(self.config["output_dir"], filename_terms = ["particles", "truth"])

        for event in self.csv_events:
            particles = pd.read_csv(event["particles"])
            hits = pd.read_csv(event["truth"])
            hits, particles = self._select_hits(hits, particles)
            hits = self._add_all_features(hits)
            hits = self._clean_noise_duplicates(hits)

            tracks, track_features, hits = self._build_true_tracks(hits, particles)
            hits, particles, tracks = self._custom_processing(hits, particles, tracks)
            graph = self._build_graph(hits, tracks, track_features)
            self._save_pyg_data(graph, event["event_id"])

    def _build_graph(self, hits, tracks, track_features):
        """
        Builds a PyG data object from the hits, particles and tracks.
        """
        
        graph = Data()
        for feature in self.config["feature_sets"]["hit_features"]:
            graph[feature] = torch.from_numpy(hits[feature].values)
        
        graph.track_edges = torch.from_numpy(tracks)
        for feature in self.config["feature_sets"]["track_features"]:
            graph[feature] = torch.from_numpy(track_features[feature])

        return graph

    def _save_pyg_data(self, graph, event_id):
        """
        Save the PyG constructed graph
        """
        torch.save(graph, os.path.join(self.config["output_dir"], f"event{event_id}-graph.pyg"))


    @staticmethod
    def calc_eta(r, z):
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))  

    def _select_hits(self, hits, particles):

        """ 
        Takes a set of hits and particles and applies hard cuts to the list of hits and particles. These should be defined
        in a `hard_cuts` key in the config file. E.g.
        hard_cuts: 
            pt: [1000, inf]
            barcode: [0, 200000]
        """

        particles = particles.assign(primary=(particles.barcode < 200000).astype(int))

        if "hard_cuts" in self.config and self.config["hard_cuts"] is not None:
            raise NotImplementedError("Hard cuts not implemented yet")
            for cut in self.config["hard_cuts"]:
                pass
            hits = hits.merge(
                particles[["particle_id", "pt", "vx", "vy", "vz", "primary"]],
                on="particle_id",
            )

        else:
            hits = hits.merge(
                particles[["particle_id", "pt", "vx", "vy", "vz", "primary", "pdgId", "radius"]],
                on="particle_id",
                how="left",
            )

        hits["nhits"] = hits.groupby("particle_id")["particle_id"].transform("count")
        hits["particle_id"] = hits["particle_id"].fillna(0).astype(int)
        hits.loc[hits.particle_id == 0, "nhits"] = -1

        return hits, particles

    @staticmethod
    def _clean_noise_duplicates(hits):
        """
        This handles the case where a hit is assigned to both a particle and noise (e.g. in a ghost spacepoint). 
        This is not sensible, so we remove those duplicated noise hits.
        """

        noise_hits = hits[hits.particle_id == 0].drop_duplicates(subset="hit_id")
        signal_hits = hits[hits.particle_id != 0]

        non_duplicate_noise_hits = noise_hits[~noise_hits.hit_id.isin(signal_hits.hit_id)]
        hits = pd.concat([signal_hits, non_duplicate_noise_hits], ignore_index=True)
        # Sort hits by hit_id for ease of processing
        hits = hits.sort_values("hit_id").reset_index(drop=True)

        return hits

    def _add_all_features(self, hits):
        assert all([col in hits.columns for col in ["x","y","z"]]), "Need to add (x,y,z) features"
        
        r = np.sqrt(hits.x**2 + hits.y**2)
        phi = np.arctan2(hits.y, hits.x)
        eta = self.calc_eta(r, hits.z)
        hits = hits.assign(r=r, phi=phi, eta=eta)

        return hits

    def _build_true_tracks(self, hits, particles):

        assert all([col in hits.columns for col in ["particle_id", "hit_id", "x","y","z","vx","vy","vz"]]), "Need to add (particle_id, hit_id), (x,y,z) and (vx,vy,vz) features to hits dataframe in custom EventReader class"

        signal = hits[(hits.particle_id != 0)]

        # Sort by increasing distance from production
        signal = signal.assign(
            R=np.sqrt(
                (signal.x - signal.vx) ** 2
                + (signal.y - signal.vy) ** 2
                + (signal.z - signal.vz) ** 2
            )
        )

        signal = signal.sort_values("R").reset_index(drop=False)

        # Group by particle ID
        if "module_columns" not in self.config or self.config["module_columns"] is None:
            module_columns = ["barrel_endcap", "hardware", "layer_disk", "eta_module", "phi_module"]
        
        signal_index_list = (signal.groupby(
                ["particle_id"] + module_columns,
                sort=False,
            )["index"]
            .agg(lambda x: list(x))
            .groupby(level=0)
            .agg(lambda x: list(x)))

        track_index_edges = []
        for row in signal_index_list.values:
            for i, j in zip(row[:-1], row[1:]):
                track_index_edges.extend(list(product(i, j)))

        track_index_edges = np.array(track_index_edges).T
        track_edges = hits.hit_id.values[track_index_edges]

        assert (hits[hits.hit_id.isin(track_edges.flatten())].particle_id == 0).sum() == 0, "There are hits in the track edges that are noise"

        track_features = self._get_track_features(hits, track_index_edges)

        # Remap
        track_edges, hits = self.remap_edges(track_edges, track_features, hits)

        return track_edges, track_features, hits

    def _get_track_features(self, hits, track_index_edges):
        track_features = {}
        for track_feature in self.config["feature_sets"]["track_features"]:
            assert (hits[track_feature].values[track_index_edges][0] == hits[track_feature].values[track_index_edges][1]).all()
            track_features[track_feature] = hits[track_feature].values[track_index_edges[0]]
    
        return track_features

    @staticmethod
    def remap_edges(track_edges, track_features, hits):

        unique_hid = np.unique(hits.hit_id)
        hid_mapping = np.zeros(unique_hid.max() + 1).astype(int)
        hid_mapping[unique_hid] = np.arange(len(unique_hid))

        hits = hits.drop_duplicates(subset="hit_id").sort_values("hit_id")
        assert (hits.hit_id == unique_hid).all(), "If hit IDs are not sequential, this will mess up graph structure!"

        track_edges = hid_mapping[track_edges]

        # This test imposes a limit to how we simplify the graph: We don't allow shared EDGES (i.e. two different particles can share a hit, but not an edge between the same two hits). We want to ensure these are in a tiny minority
        assert ((hits.particle_id.values[track_edges[0]] != track_features["particle_id"]) & (hits.particle_id.values[track_edges[1]] != track_features["particle_id"])).sum() < 50, "The number of shared EDGES is unusually high!"

        return track_edges, hits

    def get_file_names(self, inputdir, filename_terms : Union[str, list] = None):
        """
        Takes a list of filename terms and searches for all files containing those terms AND a number. Returns the files and numbers.
        For the list of numbers, search for each of the matching terms and files containing that number AND ONLY THAT NUMBER.
        """

        if isinstance(filename_terms, str):
            filename_terms = [filename_terms]
        elif filename_terms is None:
            filename_terms = ["*"]

        all_files_in_template = [ glob.glob(os.path.join(inputdir, f"*{template}*")) for template in filename_terms ]
        all_files_in_template = list(chain.from_iterable(all_files_in_template))
        all_event_ids = sorted(list(set([re.findall("[0-9]+", file)[-1] for file in all_files_in_template])))

        all_events = []
        for event_id in all_event_ids:
            event = {"event_id": event_id}
            for term in filename_terms:
                # Search for a file containing the term and EXACTLY the event id (i.e. no other numbers)
                template_file = [file for file in all_files_in_template if term in os.path.basename(file) and re.findall("[0-9]+", file)[-1] == event_id]
                if len(template_file) == 0:
                    print(f"Could not find file for term {term} and event id {event_id}")
                    break
                else:
                    event[term] = template_file[0]
            else:
                all_events.append(event)

        return all_events

    def _custom_processing(self, hits, particles, tracks):
        """
        This is called after the base class has finished processing the hits, particles and tracks.
        """
        pass

    def _test_csv_conversion(self):
        
        self.csv_events = self.get_file_names(self.config["output_dir"], filename_terms=["truth", "particles"])
        assert len(self.csv_events) > 0, "No CSV files found in output directory matching the formats (event[eventID]-truth.csv, event[eventID]-particles.csv). Please check that the conversion to CSV was successful."

        # Load the first event
        event = self.csv_events[0]
        truth = pd.read_csv(event["truth"])
        particles = pd.read_csv(event["particles"])
        assert len(truth) > 0, "No truth spacepoints found in CSV file. Please check that the conversion to CSV was successful."
        assert len(particles) > 0, "No particles found in CSV file. Please check that the conversion to CSV was successful."

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.files[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]