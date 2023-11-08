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

import networkx as nx
from torch_geometric.utils import to_networkx
from functools import partial
import torch
import pandas as pd
import numpy as np
from itertools import chain


def remove_cycles(graph):
    """
    Remove cycles from the graph, simply by pointing all edges outwards
    """

    R = graph.r**2 + graph.z**2
    edge_flip_mask = R[graph.edge_index[0]] > R[graph.edge_index[1]]
    graph.edge_index[:, edge_flip_mask] = graph.edge_index[:, edge_flip_mask].flip(0)

    return graph


def filter_graph(graph, score_name, threshold):
    """
    Remove edges from the graph that have a score below the given threshold
    And remove nodes that become isolated after the edge filtering
    score_name :  the name of the edge "feature" corresponding to the GNN score
    """
    # Convert to networkx graph
    G = to_networkx(graph, ["hit_id"], [score_name], to_undirected=False)

    # Remove edges below threshold
    list_fake_edges = [
        (u, v) for u, v, e in G.edges(data=True) if e[score_name] <= threshold
    ]
    G.remove_edges_from(list_fake_edges)

    # self.log.debug(f"Number of isolated hits = {nx.number_of_isolates(G)}")
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def topological_sort_graph(G):
    """
    Sort Topologcially the graph such node u appears befroe v if the connection is u->v
    This ordering is valid only if the graph has no directed cycles
    """
    H = nx.DiGraph()
    # Add nodes w/o any features attached
    # maybe this is not needed given line 48?
    H.add_nodes_from(nx.topological_sort(G))

    # put it after the add nodes
    H.add_edges_from(G.edges(data=True))
    sorted_nodes = []

    # Add corresponding nodes features
    for i in list(nx.topological_sort(G)):
        sorted_nodes.append((i, G.nodes[i]))
    H.add_nodes_from(sorted_nodes)

    return H


# =================
# CC specific utils
# =================


def get_simple_path(G):
    """
    Get the graph connected components and return only the simple paths
    (ie without any branching)
    """
    list_tracks_singlePath = []
    L_connectedComponent = sorted(nx.weakly_connected_components(G))
    for i in range(len(L_connectedComponent)):
        singlePath = True
        subGraph = L_connectedComponent[i]
        for node in subGraph:
            if not (G.out_degree(node) <= 1 and G.in_degree(node) <= 1):
                singlePath = False
                # print(G.nodes[node]["hit_id"])
        if singlePath:
            track = [G.nodes[node]["hit_id"] for node in subGraph]
            if len(track) > 2:
                list_tracks_singlePath.append(track)
            G.remove_nodes_from(subGraph)

    return list_tracks_singlePath


# ==========================
# Walkthrough specific utils
# ==========================


def walk_through(G, score_name, cut_min, cut_add):
    """
    Call walkthrough and return track candidates as list of hit ids
    """

    tracks_in_graph = get_tracks(
        G, score_name=score_name, th_min=cut_min, th_add=cut_add
    )

    pred_list = []

    # Convert subgraphs in list of node indices
    for subG in tracks_in_graph:
        subL = [subG.nodes[node]["hit_id"] for node in subG]
        pred_list.append(subL)

    return pred_list


def get_tracks(G, th_min, th_add, score_name):
    """
    Run walkthrough and return subgraphs
    """
    used_nodes = []
    sub_graphs = []
    next_hit_fn = partial(
        find_next_hits, th_min=th_min, th_add=th_add, score_name=score_name
    )

    # Rely on the fact the graph was already topologically sorted
    # to start looking first on nodes without incoming edges
    for node in G.nodes():
        # Ignore already used nodes
        if node in used_nodes:
            continue

        road = build_roads(G, node, next_hit_fn, used_nodes)
        a_road = choose_longest_road(road)

        # Case where there is only one hit: a_road = (<node>,None)
        if len(a_road) < 3:
            used_nodes.append(node)
            sub_graphs.append(G.subgraph([node]))
            continue

        # Need to drop the last item of the a_road tuple, since it is None
        a_track = list(pairwise(a_road[:-1]))
        sub = G.edge_subgraph(a_track)
        sub_graphs.append(sub)
        used_nodes += list(sub.nodes())

    return sub_graphs


# TODO understand better
def build_roads(G, starting_node, next_hit_fn, used_hits):
    """
    Build roads strating from a given node, using a choosen function
    to find the next hits
    next_hit_fn: a function return next hits, could be find_next_hits
    """

    # Get next hits from the starting node
    next_hits = next_hit_fn(G, starting_node, used_hits)

    # Case where no next hits where found
    if next_hits is None:
        return [(starting_node, None)]

    path = []
    for hit in next_hits:
        path.append((starting_node, hit))

    while True:
        new_path = []
        is_all_none = True

        # Check if we found at least one interesting next hit candidates
        for pp in path:
            if pp[-1] is not None:
                is_all_none = False
                break
        if is_all_none:
            break

        # Loop on all paths at see if we can find more hits to be added
        for pp in path:
            start = pp[-1]

            # Case where we are at the end of an interesting path
            if start is None:
                new_path.append(pp)
                continue

            # Call next hits function
            used_hits_cc = np.unique(used_hits + list(pp))
            next_hits = next_hit_fn(G, pp[-1], used_hits_cc)
            if next_hits is None:
                new_path.append(pp + (None,))
            else:
                for hit in next_hits:
                    new_path.append(pp + (hit,))

        path = new_path

    return path


def choose_longest_road(road):
    res = road[0]
    for i in range(1, len(road)):
        if len(road[i]) >= len(res):
            res = road[i]
    return res


def find_next_hits(G, current_hit, used_hits, score_name, th_min, th_add):
    """
    Find what are the next hits we keep to build trakc candidates
    G : the graph (usually pre-filtered)
    current_hit : index of the current_hit considered
    used_hits : list of already used hits (to avoid re-using them)
    th_min : minimal threshold required to build at least one track candidate
             (we take the hit with the highest score)
    th_add : additional threshold above which we keep all hit neighbors, and not only the
             the one with the highest score. It results in several track candidates
             (th_add should be larger than th_min)
    path is previous hits."""

    # Sanity check
    if th_add < th_min:
        print(
            f"WARNING : the minimal threshold {th_min} is above the additional"
            f" threshold {th_multi},               this is not how the walkthrough is"
            " supposed to be run."
        )

    # Check if current hit still have unused neighbor(s) hit(s)
    neighbors = list(set(G.neighbors(current_hit)).difference(set(used_hits)))
    if len(neighbors) < 1:
        return None

    neighbors_scores = [G.edges[(current_hit, i)][score_name] for i in neighbors]

    # Stop here if none of the remaining neighbors are above the minimal threshold
    if max(neighbors_scores) <= th_min:
        return None

    # Follow at least the hit with the highest score (above the minimal threshold)
    sorted_idx = list(reversed(np.argsort(neighbors_scores)))
    next_hits = [neighbors[sorted_idx[0]]]

    # Look if we have other "good" neighbors with score above the additional threshold
    if len(sorted_idx) > 1:
        for ii in range(1, len(sorted_idx)):
            idx = sorted_idx[ii]
            score = neighbors_scores[idx]
            if score > th_add:
                next_hits.append(neighbors[idx])
            else:
                break

    return next_hits


def add_track_labels(graph, all_trks):
    """
    Add to the graph the track labels based on tracks candidates resulting
    from connected components and walkthrough with an additional string
    to specify with which method the track was built from
    """
    flat_trks = []
    flat_trkid = []
    flat_method = []
    trkid_offset = 0

    for method, trks in all_trks.items():
        flat_trks += list(chain.from_iterable(trks))
        flat_trkid += list(
            chain.from_iterable(
                [[i + trkid_offset] * len(p) for i, p in enumerate(trks)]
            )
        )
        flat_method += list(chain.from_iterable([[method] * len(p) for p in trks]))
        trkid_offset += len(trks)

    track_df = pd.DataFrame(
        {"hit_id": flat_trks, "track_id": flat_trkid, "reco_method": flat_method}
    )

    # Remove duplicates on hit_id: TODO: In very near future, handle multiple tracks through the same hit!
    # Alexis : this line seems suspicious to me
    track_df = track_df.drop_duplicates(subset="hit_id")

    hit_id = track_df.hit_id
    track_id = track_df.track_id
    reco_method = track_df.reco_method

    # In the case that the dataframe hits are out of order with the input graph hits
    hit_id_df = pd.DataFrame({"hit_id": graph.hit_id})
    hit_id_df = hit_id_df.merge(track_df, on="hit_id", how="left")
    hit_id_df.fillna(-1, inplace=True)
    track_id_tensor = torch.from_numpy(hit_id_df.track_id.values).long()

    graph.labels = track_id_tensor
    graph.reco_method = reco_method


def pairwise(l):
    """
    Return successive overlapping pairs taken from the input list
    (not available in itertools of python 3.9)
    """
    l1 = l[:-1]
    l2 = l[1:]
    return list(zip(l1, l2))
