from __future__ import annotations

import operator
from functools import partial
from pathlib import Path

import frnn
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data as Data
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import sort_edge_index, to_networkx
from torch_sparse import SparseTensor

# torch.set_float32_matmul_precision("high")
torch.set_float32_matmul_precision("highest")
torch.manual_seed(42)

# from acorn.stages.graph_construction.models.utils import build_edges

def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    r_max: float = 0.1,
    k_max: int = 1000,
) -> torch.Tensor:
    # Compute edges
    _, idxs, _, _ = frnn.frnn_grid_points(
        points1=query.unsqueeze(0),
        points2=database.unsqueeze(0),
        lengths1=None,
        lengths2=None,
        K=k_max,
        r=r_max,
        grid=None,
        return_nn=False,
        return_sorted=True,
    )

    idxs: torch.Tensor = idxs.squeeze().int()
    ind = torch.arange(idxs.shape[0], device=query.device).repeat(idxs.shape[1], 1).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    print("edge_list.shape", edge_list.shape)

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]
    return edge_list


def find_next_hits(
    G: nx.DiGraph,
    current_hit: int,
    used_hits: set,
    score_name: str,
    th_min: float,
    th_add: float,
):
    """
    Find what are the next hits we keep to build trakc candidates
    G : the graph (usually pre-filtered)
    current_hit : index of the current_hit considered
    used_hits : a set of already used hits (to avoid re-using them)
    th_min : minimal threshold required to build at least one track candidate
             (we take the hit with the highest score)
    th_add : additional threshold above which we keep all hit neighbors, and not only the
             the one with the highest score. It results in several track candidates
             (th_add should be larger than th_min)
    path is previous hits.
    """
    # Sanity check
    if th_add < th_min:
        print(
            f"WARNING : the minimal threshold {th_min} is above the additional"
            f" threshold {th_add},               this is not how the walkthrough is"
            " supposed to be run."
        )

    # Check if current hit still have unused neighbor(s) hit(s)
    neighbors = [n for n in G.neighbors(current_hit) if n not in used_hits]
    if not neighbors:
        return None

    neighbors_scores = [(n, G.edges[(current_hit, n)][score_name]) for n in neighbors]

    # Stop here if none of the remaining neighbors are above the minimal threshold
    best_neighbor = max(neighbors_scores, key=operator.itemgetter(1))
    if best_neighbor[1] <= th_min:
        return None

    # Always add the neighoors with a score above th_add
    next_hits = [n for n, score in neighbors_scores if score > th_add]

    # Always add the highest scoring neighbor above th_min if it's not already in next_hits
    if not next_hits:
        best_hit = best_neighbor[0]
        if best_hit not in next_hits:
            next_hits.append(best_hit)

    return next_hits


def build_roads(G, starting_node, next_hit_fn, used_hits: set) -> list[tuple]:
    """Build roads starting from a given node.

    Args
    ----
        G : nx.DiGraph, the input graph after GNN.
        starting_node : int, the starting node.
        next_hit_fn : callable, the function to find the next hit.
        used_hits : set, the set of used hits.

    Returns
    -------
        path : list of list, the list of possible paths starting from the given node.
    """
    # Initialize the path with the starting node
    path = [(starting_node,)]

    while True:
        new_path = []
        is_all_none = True

        # Loop on all paths and see if we can find more hits to be added
        for pp in path:
            start = pp[-1]

            # Case where we are at the end of an interesting path
            if start is None:
                new_path.append(pp)
                continue

            # Call next hits function
            current_used_hits = used_hits | set(pp)
            next_hits = next_hit_fn(G, start, current_used_hits)
            if next_hits is None:
                new_path.append((*pp, None))
            else:
                is_all_none = False
                # branching the paths for the next hits.
                new_path.append((*pp, *next_hits))

        path = new_path
        # If all paths have reached the end, break
        if is_all_none:
            break

    return path


def get_tracks(G, th_min, th_add, score_name):
    """Run walkthrough and return subgraphs."""
    used_nodes = set()
    sub_graphs = []
    next_hit_fn = partial(find_next_hits, th_min=th_min, th_add=th_add, score_name=score_name)

    # Rely on the fact the graph was already topologically sorted
    # to start looking first on nodes without incoming edges
    for node in nx.topological_sort(G):
        # Ignore already used nodes
        if node in used_nodes:
            continue

        road = build_roads(G, node, next_hit_fn, used_nodes)
        a_road = max(road, key=len)[:-1]  # [:-1] is to remove the last None

        # Case where there is only one hit: a_road = (<node>,None)
        if len(a_road) < 3:
            used_nodes.add(node)
            continue

        # Need to drop the last item of the a_road tuple, since it is None
        sub_graphs.append([G.nodes[n]["hit_id"] for n in a_road])
        used_nodes.update(a_road)

    return sub_graphs


def get_simple_path(G, do_copy: bool = False):
    in_graph = G.copy() if not do_copy else G
    final_tracks = []
    connected_components = sorted(nx.weakly_connected_components(in_graph))
    for i in range(len(connected_components)):
        is_signal_path = True
        sub_graph = connected_components[i]
        for node in sub_graph:
            if not (in_graph.out_degree(node) <= 1 and in_graph.in_degree(node) <= 1):
                is_signal_path = False
        if is_signal_path:
            track = [in_graph.nodes[node]["hit_id"] for node in sub_graph]
            if len(track) > 2:
                final_tracks.append(track)
            in_graph.remove_nodes_from(sub_graph)
    return final_tracks, in_graph


def run_gnn_filter(model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor, batches: int = 10):
    with torch.no_grad():
        num_nodes: int = x.size(0)
        sorted_edge_index = sort_edge_index(edge_index, sort_by_row=False)
        adj_t = SparseTensor(
            row=sorted_edge_index[1], 
            col=sorted_edge_index[0], 
            sparse_sizes=(num_nodes, num_nodes),
            is_sorted=True,
            trust_data=True)

        print(adj_t)
        gnn_embedding = model.gnn(x, adj_t)
        filter_scores = [
            model.net(
                torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1)
            ).squeeze(-1)
            for subset in torch.tensor_split(sorted_edge_index, batches, dim=1)
        ]
    filter_scores = torch.cat(filter_scores).sigmoid()
    return filter_scores, sorted_edge_index, gnn_embedding


class MetricLearningInference:
    def __init__(
        self,
        model_path: str,
        r_max: float = 0.1,
        k_max: int = 300,
        filter_cut: float = 0.2,
        filter_batches: int = 10,
        cc_cut: float = 0.01,
        walk_min: float = 0.1,
        walk_max: float = 0.6,
        device: str = "cuda",
        r_index: int = 0,
        z_index: int = 2,
        debug: bool = False,
        embedding_node_features: str = "r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2",  # noqa: E501
        embedding_node_scale: str = "1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14",  # noqa: E501
        filter_node_features: str = "r, phi, z, cluster_x_1, cluster_y_1, cluster_z_1, cluster_x_2, cluster_y_2, cluster_z_2, count_1, charge_count_1, loc_eta_1, loc_phi_1, localDir0_1, localDir1_1, localDir2_1, lengthDir0_1, lengthDir1_1, lengthDir2_1, glob_eta_1, glob_phi_1, eta_angle_1, phi_angle_1, count_2, charge_count_2, loc_eta_2, loc_phi_2, localDir0_2, localDir1_2, localDir2_2, lengthDir0_2, lengthDir1_2, lengthDir2_2, glob_eta_2, glob_phi_2, eta_angle_2, phi_angle_2",  # noqa: E501
        filter_node_scale: str = "1000, 3.14, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14, 1, 1, 3.14, 3.14, 1, 1, 1, 1, 1, 1, 3.14, 3.14, 3.14, 3.14",  # noqa: E501
        gnn_node_features: str = "r, phi, z, eta, cluster_r_1, cluster_phi_1, cluster_z_1, cluster_eta_1, cluster_r_2, cluster_phi_2, cluster_z_2, cluster_eta_2",  # noqa: E501
        gnn_node_scale: str = "1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0, 1000.0, 3.14159265359, 1000.0, 1.0",  # noqa: E501
    ):
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.r_max = r_max
        self.k_max = k_max
        self.filter_cut = filter_cut
        self.filter_batches = filter_batches
        self.cc_cut = cc_cut
        self.walk_min = walk_min
        self.walk_max = walk_max
        self.device = device
        self.r_index = r_index
        self.z_index = z_index
        self.debug = debug

        self.embedding_node_features = [x.strip() for x in embedding_node_features.split(",")]
        self.embedding_node_scale = [float(x.strip()) for x in embedding_node_scale.split(",")]
        self.filter_node_features = [x.strip() for x in filter_node_features.split(",")]
        self.filter_node_scale = [float(x.strip()) for x in filter_node_scale.split(",")]
        assert self.embedding_node_features == self.filter_node_features

        self.gnn_node_features = [x.strip() for x in gnn_node_features.split(",")]
        self.gnn_node_scale = [float(x.strip()) for x in gnn_node_scale.split(",")]

        embedding_path = model_path / "embedding.pt"
        filtering_path = model_path / "filter.pt"
        gnn_path = model_path / "gnn.pt"

        self.embedding_model = torch.jit.load(embedding_path).to(device)
        self.filter_model = torch.jit.load(filtering_path).to(device)
        self.gnn_model = torch.jit.load(gnn_path).to(device)
        self.transform = ToSparseTensor(remove_edge_index=False)

        self.input_node_features = [
            "r",
            "phi",
            "z",
            "cluster_x_1",
            "cluster_y_1",
            "cluster_z_1",
            "cluster_x_2",
            "cluster_y_2",
            "cluster_z_2",
            "count_1",
            "charge_count_1",
            "loc_eta_1",
            "loc_phi_1",
            "localDir0_1",
            "localDir1_1",
            "localDir2_1",
            "lengthDir0_1",
            "lengthDir1_1",
            "lengthDir2_1",
            "glob_eta_1",
            "glob_phi_1",
            "eta_angle_1",
            "phi_angle_1",
            "count_2",
            "charge_count_2",
            "loc_eta_2",
            "loc_phi_2",
            "localDir0_2",
            "localDir1_2",
            "localDir2_2",
            "lengthDir0_2",
            "lengthDir1_2",
            "lengthDir2_2",
            "glob_eta_2",
            "glob_phi_2",
            "eta_angle_2",
            "phi_angle_2",
            "eta",
            "cluster_r_1",
            "cluster_phi_1",
            "cluster_eta_1",
            "cluster_r_2",
            "cluster_phi_2",
            "cluster_eta_2",
            "region",
        ]

    def forward(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        node_features = node_features.to(self.device).float()

        if hit_id is None:
            hit_id = torch.arange(node_features.shape[0], device=self.device)

        # Metric Learning
        embedding_inputs = node_features[
            :, [self.input_node_features.index(x) for x in self.embedding_node_features]
        ]
        embedding_inputs /= torch.tensor(self.embedding_node_scale, device=self.device).float()
        with torch.no_grad():
            embedding = self.embedding_model(embedding_inputs)

        if self.debug:
            print(f"after embedding, shape = {embedding.shape}")

        print("embedding data", embedding[0])

        # Build edges
        edge_index = build_edges(embedding, embedding, r_max=self.r_max, k_max=self.k_max)

        if self.debug:
            print(f"Number of edges after embedding: {edge_index.shape[1]:,}")

        # make it undirected and remove duplicates.
        edge_index[:, edge_index[0] > edge_index[1]] = edge_index[:, edge_index[0] > edge_index[1]].flip(0)
        edge_index = torch.unique(edge_index, dim=-1)

        # random flip
        random_flip = torch.randint(2, (edge_index.shape[1],), dtype=torch.bool)
        edge_index[:, random_flip] = edge_index[:, random_flip].flip(0)

        if self.debug:
            print("after removing duplications: ", edge_index.shape, edge_index[:, 0])

        if edge_index.shape[1] == 0:
            return torch.full((1,), -1, dtype=torch.long, device=self.device)

        # GNNFiltering
        print("edges before filtering: ", edge_index[:, :10])
        edge_scores, _, _ = run_gnn_filter(self.filter_model, embedding_inputs, edge_index, self.filter_batches)

        if self.debug:
            print("edge_score", edge_scores[:10])
            print("edge_index", edge_index[:, :10])
            out_data = Data(edge_index=edge_index, edge_scores=edge_scores, embedding=embedding)


        edge_index = edge_index[:, edge_scores >= self.filter_cut]
        # edge_index[:, edge_index[0] > edge_index[1]] = edge_index[:, edge_index[0] > edge_index[1]].flip(0)
        if self.debug:
            print(f"Number of edges after filtering: {edge_index.shape[1]:,}")

        # rearrange the edges by their distances from collision point.
        R = (
            node_features[:, self.input_node_features.index("r")] ** 2
            + node_features[:, self.input_node_features.index("z")] ** 2
        )
        edge_flip_mask = R[edge_index[0]] > R[edge_index[1]]
        edge_index[:, edge_flip_mask] = edge_index[:, edge_flip_mask].flip(0)

        # apply GNN
        gnn_input = node_features[
            :, [self.input_node_features.index(x) for x in self.gnn_node_features]
        ]
        gnn_input /= torch.tensor(self.gnn_node_scale, device=self.device).float()

        # same features on the 3 channels in the STRIP ENDCAP.
        hit_region = node_features[:, self.input_node_features.index("region")]
        gnn_input_mask = torch.logical_or(hit_region == 2, hit_region == 6).reshape(
            -1
        )
        gnn_input[gnn_input_mask] = torch.cat([
            gnn_input[gnn_input_mask, 0:4], 
            gnn_input[gnn_input_mask, 0:4], 
            gnn_input[gnn_input_mask, 0:4]], dim=1)

        # calculate edge features: dr, dphi, dz, deta, phislope, rphislope
        def reset_angle(angles):
            angles[angles > torch.pi] -= 2 * torch.pi
            angles[angles < -torch.pi] += 2 * torch.pi
            return angles

        def calculate_edge_features():
            r = node_features[:, self.input_node_features.index("r")]
            src, dst = edge_index
            dr = r[dst] - r[src]
            phi = node_features[:, self.input_node_features.index("phi")]
            dphi = reset_angle((phi[dst] - phi[src]) * torch.pi) / torch.pi
            z = node_features[:, self.input_node_features.index("z")]
            dz = z[dst] - z[src]
            eta = node_features[:, self.input_node_features.index("eta")]
            deta = eta[dst] - eta[src]
            phislope = dphi / dr
            r_avg = (r[dst] + r[src]) / 2.0
            phislope_new = torch.clamp(phislope, -100, 100)
            rphislope = torch.multiply(r_avg, phislope_new)
            return torch.stack([dr, dphi, dz, deta, phislope, rphislope], dim=1)

        edge_features = calculate_edge_features()
        with torch.no_grad():
            gnn_scores = self.gnn_model(gnn_input, edge_index, edge_features)
        gnn_scores = gnn_scores.sigmoid()

        # CC and Walkthrough
        if self.debug:
            print("After GNN...")
            out_data.gnn_scores = gnn_scores
            out_data.gnn_input_edges = edge_index
            torch.save(out_data, "debug.pt")

        return None


        score_name = "edge_scores"
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            hit_id=hit_id,
            edge_scores=edge_scores,
        )
        G = to_networkx(graph, ["hit_id"], [score_name], to_undirected=False)

        # Remove edges below threshold
        list_fake_edges = [(u, v) for u, v, e in G.edges(data=True) if e[score_name] <= self.cc_cut]
        G.remove_edges_from(list_fake_edges)

        G.remove_nodes_from(list(nx.isolates(G)))
        # print(f"After removing fake edges; Number of nodes
        # = {G.number_of_nodes():,}, number of edges = {G.number_of_edges():,}")

        all_trkx = {}
        all_trkx["cc"], G = get_simple_path(G)
        all_trkx["walk"] = get_tracks(G, self.walk_min, self.walk_max, score_name)
        if self.debug:
            print(f"Number of tracks found by CC: {len(all_trkx['cc'])}")
            print(f"Number of tracks found by Walkthrough: {len(all_trkx['walk'])}")

        # label each hits.
        track_candidates = np.array([
            item for track in all_trkx["cc"] + all_trkx["walk"] for item in [*track, -1]
        ])
        # write candidates to a file.
        if self.debug:
            out_str = [
                "".join([f"{item} " for item in track]) + "\n" for track in all_trkx["cc"] + all_trkx["walk"]
            ]
            with open("track_candidates.txt", "w") as f:
                f.writelines(out_str)
        return track_candidates

    def __call__(self, node_features: torch.Tensor, hit_id: torch.Tensor | None = None):
        return self.forward(node_features, hit_id)
    

def test_filtering():
    from acorn.stages.edge_classifier import GNNFilter
    device = "cuda"
    ckpt_base_path = "/global/cfs/cdirs/m3443/data/GNN4ITK/AcornModels"
    filtering_ckpt_path = f"{ckpt_base_path}/Filter/best-28572075-auc=0.993669-epoch=207.ckpt"
    gnn_filter = GNNFilter.load_from_checkpoint(filtering_ckpt_path).to(device)
    input_filter_data = torch.load("/pscratch/sd/x/xju/ITk/ForFinalPaper/acorn-jay-rel24/examples/JayInputs/filter_data.pt", map_location=device)
    x = input_filter_data.x
    edge_index = input_filter_data.edge_index
    adj_t = input_filter_data.adj_t
    batches = 10


    def run_forward(model):
        with torch.no_grad():
            gnn_embedding = model.gnn(x, adj_t)
            filter_scores = [
                model.net(
                    torch.cat([gnn_embedding[subset[0]], gnn_embedding[subset[1]]], dim=-1)
                ).squeeze(-1)
                for subset in torch.tensor_split(edge_index, batches, dim=1)
            ]
        filter_scores = torch.cat(filter_scores).sigmoid()
        return gnn_embedding, filter_scores

    acorn_gnn_embedding, acorn_filter_scores = run_forward(gnn_filter)

    my_filter_model = torch.jit.load("/global/cfs/cdirs/m3443/data/GNN4ITK/AcornModels/Filter/filter.pt").to(device)
    my_gnn_embedding, my_filter_scores = run_forward(my_filter_model)
    

    print(torch.allclose(acorn_gnn_embedding, my_gnn_embedding, atol=1e-6))
    print(torch.allclose(acorn_filter_scores, my_filter_scores, atol=1e-6))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference for Metric Learning")
    parser.add_argument("-i", "--input", type=str, default="node_features.pt")
    parser.add_argument("-m", "--model", type=str, default="./")


    args = parser.parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model path {args.model} does not exist.")
    
    inference = MetricLearningInference(model_path=args.model, r_max=0.12, k_max=1000, filter_cut=0.05, debug=True)
    node_features = torch.load(args.input)
    track_ids = inference(node_features)
    print("total tracks", np.sum(track_ids == -1))

    # test_filtering()