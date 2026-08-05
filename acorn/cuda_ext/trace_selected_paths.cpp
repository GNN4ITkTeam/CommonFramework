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
trace_selected_paths_cuda_launcher(
    torch::Tensor best_child,
    torch::Tensor selected_roots);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> trace_selected_paths_cuda(
    torch::Tensor best_child,
    torch::Tensor selected_roots) {
  check_cuda_vector(best_child, "best_child");
  check_cuda_vector(selected_roots, "selected_roots");

  TORCH_CHECK(
      best_child.scalar_type() == torch::kInt64,
      "best_child must have dtype int64");
  TORCH_CHECK(
      selected_roots.scalar_type() == torch::kInt64,
      "selected_roots must have dtype int64");

  return trace_selected_paths_cuda_launcher(best_child, selected_roots);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "trace_selected_paths_cuda",
      &trace_selected_paths_cuda,
      "Trace selected best-child paths on CUDA");
}