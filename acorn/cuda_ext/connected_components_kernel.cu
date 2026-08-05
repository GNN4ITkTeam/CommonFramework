#include <cstdint>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <torch/extension.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kRoundsPerSync = 4;

__global__ void init_labels_kernel(int32_t* labels, int32_t* labels_next, int64_t num_nodes) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes) {
    return;
  }

  labels[index] = static_cast<int32_t>(index);
  labels_next[index] = static_cast<int32_t>(index);
}

__global__ void hook_edges_kernel(
    const int32_t* src,
    const int32_t* dst,
    int64_t num_edges,
    const int32_t* labels,
    int32_t* labels_next,
    int32_t* changed) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_edges) {
    return;
  }

  const int32_t u = src[index];
  const int32_t v = dst[index];
  const int32_t label_u = labels[u];
  const int32_t label_v = labels[v];

  if (label_u == labels[label_u] && label_v < label_u) {
    atomicMin(labels_next + label_u, label_v);
    *changed = 1;
  } else if (label_v == labels[label_v] && label_u < label_v) {
    atomicMin(labels_next + label_v, label_u);
    *changed = 1;
  }
}

__global__ void shortcut_kernel(
    const int32_t* labels,
    int32_t* labels_next,
    int64_t num_nodes,
    int32_t* changed) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes) {
    return;
  }

  const int32_t label = labels[index];
  const int32_t parent = labels[label];
  if (label != parent) {
    labels_next[index] = parent;
    *changed = 1;
  }
}

__global__ void make_label_mask_kernel(
    const int32_t* labels,
    int32_t* label_mask,
    int64_t num_nodes) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes) {
    return;
  }

  label_mask[labels[index]] = 1;
}

__global__ void remap_labels_kernel(
    int32_t* labels,
    const int32_t* mapping,
    int64_t num_nodes) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes) {
    return;
  }

  labels[index] = mapping[labels[index]];
}

__global__ void compute_degrees_kernel(
    const int32_t* src,
    const int32_t* dst,
    int64_t num_edges,
    int32_t* in_degrees,
    int32_t* out_degrees) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_edges) {
    return;
  }

  atomicAdd(out_degrees + src[index], int32_t{1});
  atomicAdd(in_degrees + dst[index], int32_t{1});
}

__global__ void mark_bad_components_kernel(
    const int32_t* in_degrees,
    const int32_t* out_degrees,
    const int32_t* labels,
    int64_t num_nodes,
    int32_t* bad_components) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes) {
    return;
  }

  if (max(in_degrees[index], out_degrees[index]) > 1) {
    bad_components[labels[index]] = 1;
  }
}

__global__ void build_component_masks_kernel(
    const int32_t* labels,
    const bool* large_component_mask,
    const int32_t* bad_components,
    int64_t num_nodes,
    bool* simple_path_mask,
    bool* complex_node_mask) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes) {
    return;
  }

  const bool is_large = large_component_mask[index];
  const bool is_simple = is_large && bad_components[labels[index]] == 0;
  simple_path_mask[index] = is_simple;
  complex_node_mask[index] = is_large && !is_simple;
}

__global__ void build_complex_edge_mask_kernel(
    const int32_t* src,
    const int32_t* dst,
    const bool* complex_node_mask,
    int64_t num_edges,
    bool* complex_edge_mask) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_edges) {
    return;
  }

  complex_edge_mask[index] =
      complex_node_mask[src[index]] && complex_node_mask[dst[index]];
}

}  // namespace

std::tuple<torch::Tensor, int64_t> connected_components_cuda_launcher(
    torch::Tensor src,
    torch::Tensor dst,
    int64_t num_nodes) {
  c10::cuda::CUDAGuard device_guard(src.device());

  auto options = torch::TensorOptions()
                     .device(src.device())
                     .dtype(torch::kInt32);

  if (num_nodes == 0) {
    return std::make_tuple(torch::empty({0}, options), int64_t{0});
  }

  auto labels = torch::empty({num_nodes}, options);
  auto labels_next = torch::empty({num_nodes}, options);
  auto changed = torch::zeros({1}, options);
  auto label_mask = torch::zeros({num_nodes}, options);
  auto mapping = torch::zeros({num_nodes}, options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(src.device().index()).stream();
  const dim3 node_grid((num_nodes + kBlockSize - 1) / kBlockSize);
  const dim3 edge_grid((src.numel() + kBlockSize - 1) / kBlockSize);

  init_labels_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      labels.data_ptr<int32_t>(),
      labels_next.data_ptr<int32_t>(),
      num_nodes);
  C10_CUDA_CHECK(cudaGetLastError());

  bool has_changed = false;
  do {
    changed.zero_();
    for (int round = 0; round < kRoundsPerSync; ++round) {
      labels_next.copy_(labels);
      hook_edges_kernel<<<edge_grid, kBlockSize, 0, stream>>>(
          src.data_ptr<int32_t>(),
          dst.data_ptr<int32_t>(),
          src.numel(),
          labels.data_ptr<int32_t>(),
          labels_next.data_ptr<int32_t>(),
          changed.data_ptr<int32_t>());
      C10_CUDA_CHECK(cudaGetLastError());
      std::swap(labels, labels_next);

      labels_next.copy_(labels);
      shortcut_kernel<<<node_grid, kBlockSize, 0, stream>>>(
          labels.data_ptr<int32_t>(),
          labels_next.data_ptr<int32_t>(),
          num_nodes,
          changed.data_ptr<int32_t>());
      C10_CUDA_CHECK(cudaGetLastError());
      std::swap(labels, labels_next);
    }

    has_changed = changed.cpu().item<int32_t>() != 0;
  } while (has_changed);

  make_label_mask_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      labels.data_ptr<int32_t>(),
      label_mask.data_ptr<int32_t>(),
      num_nodes);
  C10_CUDA_CHECK(cudaGetLastError());

  mapping.copy_(label_mask);
  thrust::exclusive_scan(
      thrust::cuda::par.on(stream),
      mapping.data_ptr<int32_t>(),
      mapping.data_ptr<int32_t>() + num_nodes,
      mapping.data_ptr<int32_t>());

  remap_labels_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      labels.data_ptr<int32_t>(),
      mapping.data_ptr<int32_t>(),
      num_nodes);
  C10_CUDA_CHECK(cudaGetLastError());

  int64_t num_components = label_mask.sum().item<int64_t>();
  return std::make_tuple(labels, num_components);
}

  std::tuple<torch::Tensor, torch::Tensor> process_components_cuda_launcher(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor labels,
    torch::Tensor large_component_mask,
    int64_t num_nodes,
    int64_t num_components) {
    c10::cuda::CUDAGuard device_guard(src.device());

    auto int_options = torch::TensorOptions()
               .device(src.device())
               .dtype(torch::kInt32);
    auto bool_options = torch::TensorOptions()
                .device(src.device())
                .dtype(torch::kBool);

    auto in_degrees = torch::zeros({num_nodes}, int_options);
    auto out_degrees = torch::zeros({num_nodes}, int_options);
    auto bad_components = torch::zeros({num_components}, int_options);
    auto simple_path_mask = torch::zeros({num_nodes}, bool_options);
    auto complex_node_mask = torch::zeros({num_nodes}, bool_options);
    auto complex_edge_mask = torch::zeros({src.numel()}, bool_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(src.device().index()).stream();
    const dim3 node_grid((num_nodes + kBlockSize - 1) / kBlockSize);
    const dim3 edge_grid((src.numel() + kBlockSize - 1) / kBlockSize);
    const dim3 component_grid((num_components + kBlockSize - 1) / kBlockSize);
    (void)component_grid;

    compute_degrees_kernel<<<edge_grid, kBlockSize, 0, stream>>>(
      src.data_ptr<int32_t>(),
      dst.data_ptr<int32_t>(),
      src.numel(),
      in_degrees.data_ptr<int32_t>(),
      out_degrees.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());

    mark_bad_components_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      in_degrees.data_ptr<int32_t>(),
      out_degrees.data_ptr<int32_t>(),
      labels.data_ptr<int32_t>(),
      num_nodes,
      bad_components.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());

    build_component_masks_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      labels.data_ptr<int32_t>(),
      large_component_mask.data_ptr<bool>(),
      bad_components.data_ptr<int32_t>(),
      num_nodes,
      simple_path_mask.data_ptr<bool>(),
      complex_node_mask.data_ptr<bool>());
    C10_CUDA_CHECK(cudaGetLastError());

    build_complex_edge_mask_kernel<<<edge_grid, kBlockSize, 0, stream>>>(
      src.data_ptr<int32_t>(),
      dst.data_ptr<int32_t>(),
      complex_node_mask.data_ptr<bool>(),
      src.numel(),
      complex_edge_mask.data_ptr<bool>());
    C10_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(simple_path_mask, complex_edge_mask);
  }