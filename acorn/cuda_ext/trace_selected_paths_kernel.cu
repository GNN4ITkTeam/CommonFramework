#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void trace_selected_paths_kernel(
    const int64_t* best_child,
    const int64_t* selected_roots,
    int64_t* selected_track_labels,
    int64_t* selected_nodes,
    int64_t* selected_count,
    int64_t num_nodes,
    int64_t num_paths) {
  const int64_t path_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (path_index >= num_paths) {
    return;
  }

  int64_t node = selected_roots[path_index];
  int64_t step = 0;

  while (node != -1 && step < num_nodes) {
    const auto slot = static_cast<int64_t>(atomicAdd(
        reinterpret_cast<unsigned long long*>(selected_count),
        static_cast<unsigned long long>(1)));
    selected_track_labels[slot] = path_index;
    selected_nodes[slot] = node;
    node = best_child[node];
    ++step;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
trace_selected_paths_cuda_launcher(
    torch::Tensor best_child,
    torch::Tensor selected_roots) {
  c10::cuda::CUDAGuard device_guard(best_child.device());

  const int64_t num_nodes = best_child.numel();
  const int64_t num_paths = selected_roots.numel();
  auto index_options = torch::TensorOptions().device(best_child.device()).dtype(torch::kInt64);

  auto selected_track_labels = torch::empty({num_nodes}, index_options);
  auto selected_nodes = torch::empty({num_nodes}, index_options);
  auto selected_count = torch::zeros({1}, index_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(best_child.device().index()).stream();
  const dim3 path_grid((num_paths + kBlockSize - 1) / kBlockSize);
 
  trace_selected_paths_kernel<<<path_grid, kBlockSize, 0, stream>>>(
      best_child.data_ptr<int64_t>(),
      selected_roots.data_ptr<int64_t>(),
      selected_track_labels.data_ptr<int64_t>(),
      selected_nodes.data_ptr<int64_t>(),
      selected_count.data_ptr<int64_t>(),
      num_nodes,
      num_paths);
  C10_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(selected_track_labels, selected_nodes, selected_count);
}