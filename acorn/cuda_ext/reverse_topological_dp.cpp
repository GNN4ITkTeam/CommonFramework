#include <tuple>

#include <torch/extension.h>

namespace {

void check_cuda_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
reverse_topological_dp_cuda_launcher(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor edge_weight,
    torch::Tensor incoming_row_ptr,
    torch::Tensor incoming_col_idx,
    torch::Tensor topo_order,
    torch::Tensor active_nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reverse_topological_dp_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor edge_weight,
    torch::Tensor incoming_row_ptr,
    torch::Tensor incoming_col_idx,
    torch::Tensor topo_order,
    torch::Tensor active_nodes) {
  check_cuda_vector(row_ptr, "row_ptr");
  check_cuda_vector(col_idx, "col_idx");
  check_cuda_vector(edge_weight, "edge_weight");
  check_cuda_vector(incoming_row_ptr, "incoming_row_ptr");
  check_cuda_vector(incoming_col_idx, "incoming_col_idx");
  check_cuda_vector(topo_order, "topo_order");
  check_cuda_vector(active_nodes, "active_nodes");

  TORCH_CHECK(row_ptr.scalar_type() == torch::kInt32, "row_ptr must have dtype int32");
  TORCH_CHECK(col_idx.scalar_type() == torch::kInt32, "col_idx must have dtype int32");
  TORCH_CHECK(
      edge_weight.scalar_type() == torch::kFloat32,
      "edge_weight must have dtype float32");
  TORCH_CHECK(
      incoming_row_ptr.scalar_type() == torch::kInt32,
      "incoming_row_ptr must have dtype int32");
  TORCH_CHECK(
      incoming_col_idx.scalar_type() == torch::kInt32,
      "incoming_col_idx must have dtype int32");
  TORCH_CHECK(
      topo_order.scalar_type() == torch::kInt32,
      "topo_order must have dtype int32");
  TORCH_CHECK(
      active_nodes.scalar_type() == torch::kBool,
      "active_nodes must have dtype bool");
  TORCH_CHECK(row_ptr.numel() == topo_order.numel() + 1, "row_ptr must have length num_nodes + 1");
  TORCH_CHECK(
      incoming_row_ptr.numel() == topo_order.numel() + 1,
      "incoming_row_ptr must have length num_nodes + 1");
  TORCH_CHECK(active_nodes.numel() == topo_order.numel(), "active_nodes must have length num_nodes");

  return reverse_topological_dp_cuda_launcher(
      row_ptr,
      col_idx,
      edge_weight,
      incoming_row_ptr,
      incoming_col_idx,
      topo_order,
      active_nodes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "reverse_topological_dp_cuda",
      &reverse_topological_dp_cuda,
      "Reverse-topological DP on CUDA CSR graph tensors");
}