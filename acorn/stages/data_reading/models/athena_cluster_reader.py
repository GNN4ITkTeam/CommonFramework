import os
from torch.utils.data import random_split
import pandas as pd

from ..data_reading_stage import EventReader
from . import athena_utils


class AthenaClusterReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        """
        Here we initialize and load any attributes that are needed for the _build_single_csv function.
        """

        input_dir = self.config["input_dir"]
        self.raw_events = self.get_file_names(
            input_dir, filename_terms=["clusters", "particles", "spacepoints"]
        )

        # Very opinionated: We split the data by 80/10/10: train/val/test
        self.trainset, self.valset, self.testset = random_split(
            self.raw_events,
            [
                int(len(self.raw_events) * 0.8),
                int(len(self.raw_events) * 0.1),
                int(len(self.raw_events) * 0.1),
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
        clusters_file = event["clusters"]
        particles_file = event["particles"]
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
        particles = particles.astype(self.config["particles_datatypes"])

        # Read clusters
        clusters, self.shape_list = athena_utils.read_clusters(
            clusters_file, particles, self.config["column_lookup"]
        )

        # Get truth spacepoints
        truth = athena_utils.get_truth_clusters(
            clusters, self.config["spacepoints_datatypes"]
        )
        truth = athena_utils.add_region_labels(truth, self.config["region_labels"])
        truth = athena_utils.add_module_id(truth, self.module_lookup)

        # Get detectable particles
        detectable_particles = athena_utils.get_detectable_particles(
            particles, clusters
        )

        # Save to CSV
        truth.to_csv(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event_id))),
            index=False,
        )
        detectable_particles.to_csv(
            os.path.join(output_dir, "event{:09}-particles.csv".format(int(event_id))),
            index=False,
        )
