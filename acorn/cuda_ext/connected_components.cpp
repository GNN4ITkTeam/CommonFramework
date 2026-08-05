#include <cstdint>
#include <tuple>

#include <torch/extension.h>

namespace {

void check_edge_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt32 || tensor.scalar_type() == torch::kInt64,
      name,
      " must have dtype int32 or int64");
}

}  // namespace

std::tuple<torch::Tensor, int64_t> connected_components_cuda_launcher(
    torch::Tensor src,
    torch::Tensor dst,
    int64_t num_nodes);

std::tuple<torch::Tensor, torch::Tensor> process_components_cuda_launcher(
  torch::Tensor src,
  torch::Tensor dst,
  torch::Tensor labels,
  torch::Tensor large_component_mask,
  int64_t num_nodes,
  int64_t num_components);

std::tuple<torch::Tensor, int64_t> connected_components_cuda(
    torch::Tensor src,
    torch::Tensor dst,
    int64_t num_nodes) {
  check_edge_tensor(src, "src");
  check_edge_tensor(dst, "dst");
  TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src and dst must share a dtype");
  TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same length");
  TORCH_CHECK(num_nodes >= 0, "num_nodes must be non-negative");
  TORCH_CHECK(
      src.scalar_type() == torch::kInt32,
      "connected_components_cuda currently expects int32 edge tensors");

  return connected_components_cuda_launcher(src, dst, num_nodes);
}

  std::tuple<torch::Tensor, torch::Tensor> process_components_cuda(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor labels,
    torch::Tensor large_component_mask,
    int64_t num_nodes,
    int64_t num_components) {
    check_edge_tensor(src, "src");
    check_edge_tensor(dst, "dst");
    check_edge_tensor(labels, "labels");
    TORCH_CHECK(labels.scalar_type() == torch::kInt32, "labels must have dtype int32");
    TORCH_CHECK(
      large_component_mask.is_cuda(),
      "large_component_mask must be a CUDA tensor");
    TORCH_CHECK(large_component_mask.dim() == 1, "large_component_mask must be 1D");
    TORCH_CHECK(
      large_component_mask.scalar_type() == torch::kBool,
      "large_component_mask must have dtype bool");
    TORCH_CHECK(
      large_component_mask.is_contiguous(),
      "large_component_mask must be contiguous");
    TORCH_CHECK(src.scalar_type() == torch::kInt32, "src must have dtype int32");
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src and dst must share a dtype");
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same length");
    TORCH_CHECK(labels.numel() == num_nodes, "labels length must equal num_nodes");
    TORCH_CHECK(
      large_component_mask.numel() == num_nodes,
      "large_component_mask length must equal num_nodes");
    TORCH_CHECK(num_nodes >= 0, "num_nodes must be non-negative");
    TORCH_CHECK(num_components >= 0, "num_components must be non-negative");

    return process_components_cuda_launcher(
      src,
      dst,
      labels,
      large_component_mask,
      num_nodes,
      num_components);
  }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "connected_components_cuda",
      &connected_components_cuda,
      "Connected components on CUDA edge tensors");
    m.def(
      "process_components_cuda",
      &process_components_cuda,
      "Simple-path classification masks on CUDA");
}