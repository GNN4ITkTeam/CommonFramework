"""
This class represents the entire logic of the graph construction stage. In particular, it
1. Loads events from the Athena-dumped csv files
2. Processes them into PyG Data objects with the specificied structure (see docs)
3. Runs the training of the metric learning or module map
4. Can run inference to build graphs
5. Can run evaluation to plot/print the performance of the graph construction

TODO: Update structure with the latest Gravnet base class
"""

import sys
sys.path.append("../")
import os
import warnings

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
import torch

from .utils import str_to_class, load_datafiles_in_dir, run_data_tests, construct_event_truth

class GraphConstructionStage(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Assign hyperparameters
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = CsvEventDataset

        # Load in the model to be used
        self.model = str_to_class(self.hparams["model_name"])(self.hparams)


    def forward(self, batch):

        return self.model(batch)

    def setup(self, stage):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        self.load_data()
        self.test_data()

    def load_data(self):
        """
        Load in the data for training, validation and testing.
        """

        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = self.dataset_class(self.hparams["input_dir"], data_name, data_num, self.hparams)
                setattr(self, data_name, dataset)

    def test_data(self):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "edge_index", "truth_graph"]
        optional_features = ["pid", "n_hits", "primary", "pdg_id", "ghost", "shared", "module_id", "region_id", "hit_id"]

        run_data_tests(self.trainset, self.valset, self.testset, required_features, optional_features)

        assert self.trainset[0].x.shape[1] == self.hparams["spatial_channels"], "Input dimension does not match the data"

    def build_infer_data(self):

        pass

    def produce_plots(self):

        pass

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=4
            )  
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=0
            )  
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=0
            )  
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def get_input_data(self, batch):

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], axis=-1
            )
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0

        return input_data

    def get_query_points(self, batch, spatial):

        if "query_all_points" in self.hparams["regime"]:
            query_indices = torch.arange(len(spatial)).to(spatial.device)
        elif "query_noise_points" in self.hparams["regime"]:
            query_indices = torch.cat(
                [torch.where(batch.pid == 0)[0], batch.signal_true_edges.unique()]
            )
        else:
            query_indices = batch.signal_true_edges.unique()

        query_indices = query_indices[torch.randperm(len(query_indices))][
            : self.hparams["points_per_batch"]
        ]
        query = spatial[query_indices]

        return query_indices, query

    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial, r_train=None, knn=None):
        if r_train is None:
            r_train = self.hparams["r_train"]
        if knn is None:
            knn = self.hparams["knn"]

        if "low_purity" in self.hparams["regime"]:
            knn_edges = build_edges(
                query, spatial, query_indices, r_train, 500
            )
            knn_edges = knn_edges[
                :,
                torch.randperm(knn_edges.shape[1])[
                    : int(r_train * len(query))
                ],
            ]

        else:
            knn_edges = build_edges(
                query,
                spatial,
                query_indices,
                r_train,
                knn,
            )

        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            axis=-1,
        )

        return e_spatial

    def append_random_pairs(self, e_spatial, query_indices, spatial):
        n_random = int(self.hparams["randomisation"] * len(query_indices))
        indices_src = torch.randint(
            0, len(query_indices), (n_random,), device=self.device
        )
        indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])

        e_spatial = torch.cat(
            [e_spatial, random_pairs],
            axis=-1,
        )
        return e_spatial

    def get_true_pairs(self, e_spatial, y_cluster, e_bidir, new_weights=None):
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            axis=-1,
        )
        y_cluster = torch.cat([y_cluster.int(), torch.ones(e_bidir.shape[1])])
        if new_weights is not None:
            new_weights = torch.cat(
                [new_weights, torch.ones(e_bidir.shape[1], device=self.device)]
            )
        return e_spatial, y_cluster, new_weights

    def get_hinge_distance(self, spatial, e_spatial, y_cluster):

        logging.info(f"Memory at hinge start: {torch.cuda.max_memory_allocated() / 1024**3} Gb")

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        try:
            d = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            # Need to perform this step in chunks?
            d = []
            for ref, nei in zip(reference.chunk(10), neighbors.chunk(10)):
                d.append(torch.sum((ref - nei) ** 2, dim=-1))
            d = torch.cat(d)

        logging.info(f"Memory at hinge end: {torch.cuda.max_memory_allocated() / 1024**3} Gb")
        return hinge, d

    def get_truth(self, batch, e_spatial, e_bidir):

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        return e_spatial, y_cluster

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        logging.info(f"Memory at train start: {torch.cuda.max_memory_allocated() / 1024**3} Gb")

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        with torch.no_grad():
            spatial = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial)
        
        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        if "module_veto" in self.hparams["regime"]:
            e_spatial = e_spatial[:, batch.modules[e_spatial][0] != batch.modules[e_spatial][1]]
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device)

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, e_bidir
        )

        included_hits = e_spatial.unique()
        spatial[included_hits] = self(input_data[included_hits])

        hinge, d = self.get_hinge_distance(spatial, e_spatial, y_cluster)

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + self.hparams["weight"] * positive_loss

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        input_data = self.get_input_data(batch)
        spatial = self(input_data)

        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        # Build whole KNN graph
        e_spatial = build_edges(
            spatial, spatial, indices=None, r_max=knn_radius, k_max=knn_num
        )

        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        hinge, d = self.get_hinge_distance(
            spatial, e_spatial.to(self.device), y_cluster
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        eff = cluster_true_positive / cluster_true
        pur = cluster_true_positive / cluster_positive
        if "module_veto" in self.hparams["regime"]:
            module_veto_pur = cluster_true_positive / (batch.modules[e_spatial[0]] != batch.modules[e_spatial[1]]).sum()
        else:
            module_veto_pur = 0
        
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "module_veto_pur": module_veto_pur, "current_lr": current_lr}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))
        logging.info(batch.event_file)

        return {
            "loss": loss,
            "distances": d,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        knn_val = 500 if "knn_val" not in self.hparams else self.hparams["knn_val"]
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], knn_val, log=True
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 1000, log=False
        )

        return outputs

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        logging.info("Optimizer step for batch {}".format(batch_idx))
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        logging.info("Optimizer step done for batch {}".format(batch_idx))

class CsvEventDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name, num_events, hparams=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        
        self.input_paths = load_datafiles_in_dir(self.input_dir, self.data_name, self.num_events)
        
    def len(self):
        return len(self.input_paths)

    def get(self, idx):

        event = torch.load(self.input_paths[idx], map_location=torch.device("cpu"))
        event = self.construct_truth(event)
                
        return event

    def construct_truth(self, event):
        """
        Construct the truth and weighting labels for the model training.
        """
        
        return construct_event_truth(event, self.hparams)