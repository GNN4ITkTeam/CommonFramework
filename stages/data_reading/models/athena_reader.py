import os
import numpy as np
from torch.utils.data import random_split

from ..data_reading_stage import EventReader
from .athena_utils import read_particles, read_spacepoints, read_clusters, convert_barcodes, get_truth_spacepoints, get_detectable_particles


class AthenaReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        
    def _custom_processing(self, hits, particles, tracks):
        """
        This is called after the base class has finished processing the hits, particles and tracks.
        In Athena, we will use it for some fine-tuning of the final outputs, including adding region labels to the hits (for heteroGNN stage).
        """
        # Add region labels to hits
        hits = self._add_region_labels(hits)
        
        return hits, particles, tracks
        
    def _add_region_labels(self, hits):
        """
        Label the 6 detector regions (forward-endcap pixel, forward-endcap strip, etc.)
        """
        
        for region_label, conditions in self.config["region_labels"].items():
            condition_mask = np.logical_and.reduce([hits[condition_column] == condition for condition_column, condition in conditions.items()])
            hits.loc[condition_mask, "region"] = region_label

        assert (hits.region.isna()).sum() == 0, "There are hits that do not belong to any region!"

        return hits

    def convert_to_csv(self):
        """
        Convert the full set of Athena events to CSV. This produces files in /trainset, /valset and /testset to ensure no overlaps.
        By default, we split 80/10/10 the three datasets.
        """
        
        input_dir = self.config["input_dir"]
        self.raw_events = self.get_file_names(input_dir, filename_terms = ["clusters", "particles", "spacepoints"])
        self.trainset, self.valset, self.testset = random_split(self.raw_events, [int(len(self.raw_events)*0.8), int(len(self.raw_events)*0.1), int(len(self.raw_events)*0.1)])

        for dataset, dataset_name in zip([self.trainset, self.valset, self.testset], ["trainset", "valset", "testset"]):
            self.build_csv_dataset(dataset, dataset_name)

    def build_csv_dataset(self, dataset, data_name):

        output_dir = os.path.join(self.config["output_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)

        for event in dataset:

            clusters_file = event["clusters"]
            particles_file = event["particles"]
            spacepoints_file = event["spacepoints"]
            event_id = event["event_id"]

            # Check if file already exists
            if os.path.exists(os.path.join(output_dir, "event{:09}-particles.csv".format(int(event_id)))) and os.path.exists(os.path.join(self.config["output_dir"], "event{:09}-truth.csv".format(int(event_id)))):
                print("File already exists, skipping...")
                continue

            # Read particles
            particles = read_particles(particles_file)
            particles = convert_barcodes(particles)
            particles = particles.astype(self.config["particles_datatypes"])

            # Read spacepoints
            pixel_spacepoints, strip_spacepoints = read_spacepoints(spacepoints_file)

            # Read clusters
            clusters = read_clusters(clusters_file, particles, self.config["column_lookup"])

            # Get truth spacepoints
            truth = get_truth_spacepoints(pixel_spacepoints, strip_spacepoints, clusters, self.config["spacepoints_datatypes"])

            # Get detectable particles
            detectable_particles = get_detectable_particles(particles, clusters)

            # Save to CSV
            truth.to_csv(os.path.join(output_dir, "event{:09}-truth.csv".format(int(event_id))), index=False)
            detectable_particles.to_csv(os.path.join(output_dir, "event{:09}-particles.csv".format(int(event_id))), index=False)
