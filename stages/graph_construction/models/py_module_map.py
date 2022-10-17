# 3rd party imports
import logging
import pandas as pd
import torch
try:
    import cudf
except ImportError:
    logging.warning("cuDF not found, using pandas instead")

# Local imports
from ..graph_construction_stage import GraphConstructionStage

class PyModuleMap(GraphConstructionStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        self.hparams = hparams
        self.gpu_available = torch.cuda.is_available()
        self.load_module_map()

    def load_module_map(self):
        """
        Load in the module map dataframe. Should be in CSV format.
        """

        names = ["mid_1","mid_2","mid_3","occurence","z0max_12","z0min_12","dphimax_12","dphimin_12","phiSlopemax_12","phiSlopemin_12","detamax_12","detamin_12","z0max_23","z0min_23","dphimax_23","dphimin_23","phiSlopemax_23","phiSlopemin_23","detamax_23","detamin_23","diff_dzdr_max","diff_dzdr_min","diff_dydx_max","diff_dydx_min"]
        self.MM = pd.read_csv(self.hparams["module_map_path"],names=names,header=None, delim_whitespace=True)
        self.MM_1, self.MM_2, self.MM_triplet = self.get_module_features(self.MM)
        
        if self.gpu_available:
            self.MM_1 = cudf.from_pandas(self.MM_1)
            self.MM_2 = cudf.from_pandas(self.MM_2)
            self.MM_triplet = cudf.from_pandas(self.MM_triplet)

    def build_graphs(self, dataset, data_name):
        """
        Build the graphs for the data.
        """

        logging.info("Building graphs for {}".format(data_name))
        for hits, graph in tqdm(dataset):
            if self.gpu_available:
                cudf.from_pandas(hits)

            merged_hits_1 = hits.merge(self.MM_1, how="inner", left_on="mid", right_on="mid_2").drop(columns="mid")
            merged_hits_2 = hits.merge(self.MM_2, how="inner", left_on="mid", right_on="mid_2").drop(columns="mid")

            doublet_edges_1 = get_doublet_edges(merged_hits_1, "mid", "mid_1", first_doublet=True)
            doublet_edges_1 = doublet_edges_1[["hid_1", "hid_2", "mid_1", "mid_2", "mid_3", "x_1", "x_2", "y_1", "y_2", "r_1", "r_2", "z_1", "z_2"]]

            doublet_edges_2 = get_doublet_edges(merged_hits_2, "mid_3", "mid", first_doublet=False)
            doublet_edges_2 = doublet_edges_2[["hid_1", "hid_2", "mid_1", "mid_2", "mid_3", "x_2", "y_2", "r_2", "z_2"]].rename(columns={"hid_1": "hid_2", "hid_2": "hid_3", "x_2": "x_3", "y_2": "y_3", "r_2": "r_3", "z_2": "z_3"})

            doublet_edges_2 = doublet_edges_2.merge(self.MM_triplet, how="inner", left_on=["mid_1", "mid_2", "mid_3"], right_on=["mid_1", "mid_2", "mid_3"])
            triplet_edges = doublet_edges_1.merge(doublet_edges_2, how="inner", on=["hid_1", "hid_2", "mid_3"], suffixes=("_1", "_2"))

            triplet_edges = apply_triplet_cuts(triplet_edges)

            if self.gpu_available:
                doublet_edges = cudf.concat([triplet_edges[["hid_1", "hid_2"]], triplet_edges[["hid_2", "hid_3"]].rename(columns={"hid_2": "hid_1", "hid_3": "hid_2"})])
                doublet_edges = doublet_edges.to_pandas()
            else:
                doublet_edges = pd.concat([triplet_edges[["hid_1", "hid_2"]], triplet_edges[["hid_2", "hid_3"]].rename(columns={"hid_2": "hid_1", "hid_3": "hid_2"})])

            doublet_edges = doublet_edges.drop_duplicates()
            graph.edge_index = torch.tensor(doublet_edges.values.T, dtype=torch.long)
            # TODO: Graph name file??
            torch.save(graph, os.path.join(self.hparams.stage_dir, data_name, "{}.pyg".format(graph.name)))



    def get_doublet_edges(self, merged_hits, left_merge, right_merge, first_doublet=True):
        """
        Get the doublet edges for the merged hits.
        """

        doublet_edges = []

        for batch_idx in np.arange(0, merged_hits.shape[0], self.batch_size):
            start_idx, end_idx = batch_idx, batch_idx + self.batch_size
            subset_edges = merged_hits.iloc[start_idx:end_idx].merge(hits, how="inner", left_on=left_merge, right_on=right_merge, suffixes=("_1", "_2"))
            subset_edges = apply_doublet_cuts(subset_edges, first_doublet)
            doublet_edges.append(subset_edges)

        if isinstance(merged_hits, cudf.DataFrame):
            doublet_edges = cudf.concat(doublet_edges)
        else:
            doublet_edges = pd.concat(doublet_edges)


    @staticmethod
    def get_module_features(MM):

        cols_1 = ["mid_1", "mid_2", "mid_3", "z0max_12", "z0min_12", "dphimax_12", "dphimin_12", "phiSlopemax_12", "phiSlopemin_12", "detamax_12", "detamin_12"]
        MM_ids_1 = MM[cols_1]

        cols_2 = ["mid_1", "mid_2", "mid_3", "z0max_23", "z0min_23", "dphimax_23", "dphimin_23", "phiSlopemax_23", "phiSlopemin_23", "detamax_23", "detamin_23"]
        MM_ids_2 = MM[cols_2]

        cols_3 = ["mid_1", "mid_2", "mid_3", "diff_dzdr_max","diff_dzdr_min","diff_dydx_max","diff_dydx_min"]
        MM_triplet = MM[cols_3]

        return MM_ids_1, MM_ids_2, MM_triplet

    @staticmethod
    def get_deltas(hits):
    
        delta_eta = hits.eta_1 - hits.eta_2
        delta_z = hits.z_2 - hits.z_1
        delta_r = hits.r_2 - hits.r_1

        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = reset_angle(delta_phi)

        z0 = hits.z_1 - (hits.r_1 * delta_z / delta_r)

        phi_slope = delta_phi / delta_r
        
        return delta_eta, delta_phi, z0, phi_slope

    @staticmethod
    def apply_doublet_cuts(hits, first_doublet=True):
        suffix = "12" if first_doublet else "23"

        delta_eta = hits.eta_1 - hits.eta_2
        eta_mask = (delta_eta < hits[f"detamax_{suffix}"]) & (delta_eta > hits[f"detamin_{suffix}"])

        hits = hits[eta_mask]

        delta_z = hits.z_2 - hits.z_1
        delta_r = hits.r_2 - hits.r_1
        z0 = hits.z_1 - (hits.r_1 * delta_z / delta_r)

        r_z_mask = (z0 < hits[f"z0max_{suffix}"]) & (z0 > hits[f"z0min_{suffix}"])
        hits = hits[r_z_mask]

        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = reset_angle(delta_phi)

        phi_mask = (delta_phi < hits[f"dphimax_{suffix}"]) & (delta_phi > hits[f"dphimin_{suffix}"])
        hits = hits[phi_mask]

        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = reset_angle(delta_phi)
        delta_r = hits.r_2 - hits.r_1
        phi_slope = delta_phi / delta_r
        phi_slope_mask = (phi_slope < hits[f"phiSlopemax_{suffix}"]) & (phi_slope > hits[f"phiSlopemin_{suffix}"])

        hits = hits[phi_slope_mask]

        return hits

    @staticmethod
    def apply_triplet_cuts(triplet_edges):

        dy_12 = triplet_edges.y_2 - triplet_edges.y_1
        dy_23 = triplet_edges.y_2 - triplet_edges.y_3
        dx_12 = triplet_edges.x_1 - triplet_edges.x_2
        dx_23 = triplet_edges.x_2 - triplet_edges.x_3
        diff_dydx = dy_12 / dx_12 - dy_23 / dx_23
        diff_dydx[(dx_12 == 0) & (dx_23 == 0)] = 0

        dydz_mask = (diff_dydx < triplet_edges.diff_dydx_max) & (diff_dydx > triplet_edges.diff_dydx_min)
        triplet_edges = triplet_edges[dydz_mask]

        dz_12 = triplet_edges.z_2 - triplet_edges.z_1
        dz_23 = triplet_edges.z_3 - triplet_edges.z_2
        dr_12 = triplet_edges.r_2 - triplet_edges.r_1
        dr_23 = triplet_edges.r_3 - triplet_edges.r_2

        diff_dzdr = dz_12 / dr_12 - dz_23 / dr_23
        diff_dzdr[(dr_12 == 0) & (dr_23 == 0)] = 0

        dzdr_mask = (diff_dzdr < triplet_edges.diff_dzdr_max) & (diff_dzdr > triplet_edges.diff_dzdr_min)
        triplet_edges = triplet_edges[dzdr_mask]

        return triplet_edges

    # TODO: Refactor the apply_triplet_cuts function to use this function
    @staticmethod
    def get_triplet_mask(triplet_edges, feats_A, feats_B, feats_min_max):

        dA_12 = triplet_edges[feats_A[1]] - triplet_edges[feats_A[0]]
        dA_23 = triplet_edges[feats_A[1]] - triplet_edges[feats_A[2]]
        dB_12 = triplet_edges[feats_B[1]] - triplet_edges[feats_B[0]]
        dB_23 = triplet_edges[feats_B[1]] - triplet_edges[feats_B[2]]
        diff_dAdB = dA_12 / dB_12 - dA_23 / dB_23
        diff_dAdB[(dB_12 == 0) & (dB_23 == 0)] = 0

        dAdB_mask = (diff_dAdB < triplet_edges[feats_min_max[1]]) & (diff_dAdB > triplet_edges[feats_min_max[0]])

        return dAdB_mask

    @property
    def batch_size(self):
        
        return self.hparams.batch_size if "batch_size" in self.hparams else int(1e6)