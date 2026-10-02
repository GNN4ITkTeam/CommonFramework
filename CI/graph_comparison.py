import torch
import numpy as np
import os
from tqdm import tqdm

from acorn.utils.loading_utils import load_pyg


def sorted_edges_order(edge_index: torch.Tensor) -> torch.Tensor:
    """Get the order of edges in the edge index tensor to sort the
    edges index tensor lexicographically , ie by source node (row 0)
    and then by target node (row 1).
    input: (2D tensor of shape [2, num_edges])
    Returns: order tensor.
    """
    if edge_index.ndim != 2:
        print(f"edges tensor shape: {edge_index.shape}")
        print(edge_index)
        raise ValueError("'edges' tensor must be 2D (rows, cols).")

    device = edge_index.device
    src = 0  # we wan to order first based on the src note (row0)
    dst = 1  # then based on the destination node (row1)
    keys = (edge_index[dst].cpu().numpy(), edge_index[src].cpu().numpy())
    order_np = np.lexsort(keys)
    order = torch.from_numpy(order_np).to(device)

    return order


def sort_edges(edges_info: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Sort edges info tensor in input following the provided order.
    Returns: reordered edges info tensor.
    This is nedeed to ensure that the edges are in a consistent order for comparison.
    When building the graph, the edges index is not guaranteed to be in a consistent order,
    so we need to sort it before comparing.
    """
    if edges_info.ndim != 1 and edges_info.ndim != 2:
        print(f"edges_info tensor shape: {edges_info.shape}")
        print(edges_info)
        raise ValueError("'edges_info' tensor must be 1D or 2D (rows, cols).")

    if order.ndim != 1:
        print(f"order tensor shape: {order.shape}")
        print(order)
        raise ValueError("'order' tensor must be 1D.")

    if edges_info.ndim == 1:
        return edges_info[order]
    else:
        return edges_info[:, order]


def sort_map(map, order):
    """Sort map tensor in input following the provided order.
    Returns: reordered map tensor.
    This is needed to ensure that the map is in a consistent order for comparison.
    When building the graph, the edges index is not guaranteed to be in a consistent order,
    so we need to sort it before comparing. Therefore the track_to_edge_map has to be sorted as well.
    """
    if map.ndim != 1:
        print(f"map tensor shape: {map.shape}")
        print(map)
        raise ValueError("'map' tensor must be 1D.")

    # `order` is a permutation such that `edges[:, order]` is the sorted edges.
    # We need the inverse permutation `inv` that maps old indices -> new indices.
    # Build inverse on the `map` device and use it to remap values, keeping -1 as missing.
    order = order.to(map.device)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.size(0), device=map.device, dtype=order.dtype)

    mask = map >= 0
    out = map.clone()
    if mask.any():
        out[mask] = inv[map[mask]]
    out[~mask] = -1
    return out


def compare_pyg(ref_dir, test_dir, graph_is_built):
    """
    Compare the reference and test directories for differences in .pyg files.
    graph_is_built option is used to tuned the comparison for the case where the graph is built or not.
    To compare pyg from data_reading stage: use graph_is_built=False,
    to compare pyg from graph_building stage: use graph_is_built=True.
    """

    # Get list of .pyg files in both directories
    ref_files = {
        f for f in os.listdir(ref_dir) if f.endswith(".pyg") or f.endswith(".pyg.gz")
    }
    test_files = {
        f for f in os.listdir(test_dir) if f.endswith(".pyg") or f.endswith(".pyg.gz")
    }

    if not ref_files:
        print(f"No .pyg files found in reference directory: {ref_dir}")
        return False

    if not test_files:
        print(f"No .pyg files found in test directory: {test_dir}")
        return False

    # Check for missing files
    tmp_ref_files = {f[:-3] if f.endswith(".gz") else f for f in ref_files}
    tmp_test_files = {f[:-3] if f.endswith(".gz") else f for f in test_files}
    missing_in_test = tmp_ref_files - tmp_test_files
    missing_in_ref = tmp_test_files - tmp_ref_files

    if missing_in_test:
        print(f"Missing files in test directory: {missing_in_test}")
        return False
    if missing_in_ref:
        print(f"Missing files in reference directory: {missing_in_ref}")
        return False

    print(f"Comparing {len(test_files)} .pyg files")

    # Compare contents of each file
    for file_name in tqdm(ref_files, desc="Comparing .pyg files"):
        ref_file_path = os.path.join(ref_dir, file_name)
        test_file_path = os.path.join(test_dir, file_name)

        ref_tensor = load_pyg(ref_file_path)
        test_tensor = load_pyg(test_file_path)

        if graph_is_built:
            # Sort edges and track_to_edge_map for comparison
            edge_order_ref = sorted_edges_order(ref_tensor["edge_index"])
            edge_order_test = sorted_edges_order(test_tensor["edge_index"])

            ref_tensor["edge_index"] = sort_edges(
                ref_tensor["edge_index"], edge_order_ref
            )
            test_tensor["edge_index"] = sort_edges(
                test_tensor["edge_index"], edge_order_test
            )

            ref_tensor["edge_y"] = sort_edges(ref_tensor["edge_y"], edge_order_ref)
            test_tensor["edge_y"] = sort_edges(test_tensor["edge_y"], edge_order_test)

            ref_tensor["track_to_edge_map"] = sort_map(
                ref_tensor["track_to_edge_map"], edge_order_ref
            )
            test_tensor["track_to_edge_map"] = sort_map(
                test_tensor["track_to_edge_map"], edge_order_test
            )

        ref_keys = set(ref_tensor.keys())
        test_keys = set(test_tensor.keys())

        if ref_keys != test_keys:
            print(
                f"Keys mismatch for file {file_name}: "
                f"ref-only={sorted(ref_keys - test_keys)}, "
                f"test-only={sorted(test_keys - ref_keys)}"
            )
            return False

        for key in ref_tensor.keys():

            if key in ["config"]:
                continue  # Skip config key as it may contain non-tensor data

            if (
                type(ref_tensor[key]) == int
                or type(ref_tensor[key]) == float
                or type(ref_tensor[key]) == str
            ):
                if ref_tensor[key] != test_tensor[key]:
                    print(
                        f"!!! Data mismatch for file {file_name}, key {key}: ref={ref_tensor[key]}, test={test_tensor[key]}"
                    )
                    return False
            else:

                if ref_tensor[key].dtype != test_tensor[key].dtype:
                    print(f"!!! Type mismatch for file {file_name}, key {key}")
                    print(f"Ref type:  {ref_tensor[key].dtype}")
                    print(f"Test type: {test_tensor[key].dtype}")
                    return False

                if not torch.allclose(
                    ref_tensor[key], test_tensor[key], equal_nan=True
                ):

                    print(f"!!! Data mismatch for file {file_name}, key {key}")
                    print(f"Ref data: {ref_tensor[key]}")
                    print(f"Test data: {test_tensor[key]}")

                    if ref_tensor[key].shape != test_tensor[key].shape:
                        print(
                            f"Shape mismatch: ref {ref_tensor[key].shape} "
                            f"vs test {test_tensor[key].shape}"
                        )
                    else:
                        n_diff = int(
                            torch.count_nonzero(ref_tensor[key] != test_tensor[key])
                        )
                        print(
                            f"{n_diff} differing elements (of {ref_tensor[key].numel()})"
                        )

                    return False

    print("All .pyg files match between reference and test directories.")
    return True
