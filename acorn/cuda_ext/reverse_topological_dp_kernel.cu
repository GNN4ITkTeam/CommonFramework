#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void initialize_dp_outputs_kernel(
    float* best_score,
    int32_t* best_child,
    bool* source_mask,
    int32_t* in_degree,
    int32_t* remaining_out_degree,
    int64_t num_nodes) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes) {
    return;
  }

  best_score[index] = 0.0f;
  best_child[index] = -1;
  source_mask[index] = false;
  in_degree[index] = 0;
  remaining_out_degree[index] = 0;
}

__global__ void compute_active_degree_kernel(
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const bool* active_nodes,
    int32_t* in_degree,
    int32_t* remaining_out_degree,
    int64_t num_nodes) {
  int64_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= num_nodes || !active_nodes[node]) {
    return;
  }

  int32_t active_out_degree = 0;
  for (int32_t edge_index = row_ptr[node]; edge_index < row_ptr[node + 1]; ++edge_index) {
    int32_t child = col_idx[edge_index];
    if (active_nodes[child]) {
      ++active_out_degree;
      atomicAdd(in_degree + child, 1);
    }
  }

  remaining_out_degree[node] = active_out_degree;
}

__global__ void initialize_frontier_kernel(
    const bool* active_nodes,
    const int32_t* remaining_out_degree,
    int32_t* frontier,
    int32_t* frontier_size,
    int64_t num_nodes) {
  int64_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= num_nodes) {
    return;
  }

  if (active_nodes[node] && remaining_out_degree[node] == 0) {
    int32_t slot = atomicAdd(frontier_size, 1);
    frontier[slot] = static_cast<int32_t>(node);
  }
}

__global__ void process_frontier_kernel(
    const int32_t* frontier,
    int32_t frontier_size,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const float* edge_weight,
    const bool* active_nodes,
    const float* best_score,
    float* frontier_best_score,
    int32_t* frontier_best_child) {
  int32_t frontier_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (frontier_index >= frontier_size) {
    return;
  }

  int32_t node = frontier[frontier_index];
  float best_value = 0.0f;
  int32_t best_next = -1;
  for (int32_t edge_index = row_ptr[node]; edge_index < row_ptr[node + 1]; ++edge_index) {
    int32_t child = col_idx[edge_index];
    if (!active_nodes[child]) {
      continue;
    }

    float candidate = edge_weight[edge_index] + best_score[child];
    if (candidate > best_value) {
      best_value = candidate;
      best_next = child;
    }
  }

  frontier_best_score[frontier_index] = best_value;
  frontier_best_child[frontier_index] = best_next;
}

__global__ void finalize_frontier_kernel(
    const int32_t* frontier,
    int32_t frontier_size,
    const float* frontier_best_score,
    const int32_t* frontier_best_child,
    float* best_score,
    int32_t* best_child) {
  int32_t frontier_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (frontier_index >= frontier_size) {
    return;
  }

  int32_t node = frontier[frontier_index];
  best_score[node] = frontier_best_score[frontier_index];
  best_child[node] = frontier_best_child[frontier_index];
}

__global__ void enqueue_parent_frontier_kernel(
    const int32_t* frontier,
    int32_t frontier_size,
    const int32_t* incoming_row_ptr,
    const int32_t* incoming_col_idx,
    const bool* active_nodes,
    int32_t* remaining_out_degree,
    int32_t* next_frontier,
    int32_t* next_frontier_size) {
  int32_t frontier_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (frontier_index >= frontier_size) {
    return;
  }

  int32_t node = frontier[frontier_index];
  for (int32_t edge_index = incoming_row_ptr[node]; edge_index < incoming_row_ptr[node + 1];
       ++edge_index) {
    int32_t parent = incoming_col_idx[edge_index];
    if (!active_nodes[parent]) {
      continue;
    }

    int32_t old_value = atomicSub(remaining_out_degree + parent, 1);
    if (old_value == 1) {
      int32_t slot = atomicAdd(next_frontier_size, 1);
      next_frontier[slot] = parent;
    }
  }
}

__global__ void build_source_mask_kernel(
    const bool* active_nodes,
    const int32_t* in_degree,
    const float* best_score,
    bool* source_mask,
    int64_t num_nodes) {
  int64_t node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= num_nodes) {
    return;
  }

  source_mask[node] = active_nodes[node] && in_degree[node] == 0 && best_score[node] > 0.0f;
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
    torch::Tensor active_nodes) {
  c10::cuda::CUDAGuard device_guard(row_ptr.device());

  const int64_t num_nodes = topo_order.numel();
  auto score_options = torch::TensorOptions().device(row_ptr.device()).dtype(torch::kFloat32);
  auto index_options = torch::TensorOptions().device(row_ptr.device()).dtype(torch::kInt32);
  auto mask_options = torch::TensorOptions().device(row_ptr.device()).dtype(torch::kBool);

  auto best_score = torch::empty({num_nodes}, score_options);
  auto best_child = torch::empty({num_nodes}, index_options);
  auto source_mask = torch::empty({num_nodes}, mask_options);
  auto in_degree = torch::empty({num_nodes}, index_options);
    auto remaining_out_degree = torch::empty({num_nodes}, index_options);
    auto frontier = torch::empty({num_nodes}, index_options);
    auto next_frontier = torch::empty({num_nodes}, index_options);
    auto frontier_size = torch::zeros({1}, index_options);
    auto next_frontier_size = torch::zeros({1}, index_options);
    auto frontier_best_score = torch::empty({num_nodes}, score_options);
    auto frontier_best_child = torch::empty({num_nodes}, index_options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(row_ptr.device().index()).stream();
  const dim3 node_grid((num_nodes + kBlockSize - 1) / kBlockSize);

  initialize_dp_outputs_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      best_score.data_ptr<float>(),
      best_child.data_ptr<int32_t>(),
      source_mask.data_ptr<bool>(),
      in_degree.data_ptr<int32_t>(),
      remaining_out_degree.data_ptr<int32_t>(),
      num_nodes);
  C10_CUDA_CHECK(cudaGetLastError());

    compute_active_degree_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      row_ptr.data_ptr<int32_t>(),
      col_idx.data_ptr<int32_t>(),
      active_nodes.data_ptr<bool>(),
      in_degree.data_ptr<int32_t>(),
      remaining_out_degree.data_ptr<int32_t>(),
      num_nodes);
  C10_CUDA_CHECK(cudaGetLastError());

    initialize_frontier_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      active_nodes.data_ptr<bool>(),
      remaining_out_degree.data_ptr<int32_t>(),
      frontier.data_ptr<int32_t>(),
      frontier_size.data_ptr<int32_t>(),
      num_nodes);
  C10_CUDA_CHECK(cudaGetLastError());

    int32_t current_frontier_size = frontier_size.cpu().item<int32_t>();
    while (current_frontier_size > 0) {
    const dim3 frontier_grid((current_frontier_size + kBlockSize - 1) / kBlockSize);

    process_frontier_kernel<<<frontier_grid, kBlockSize, 0, stream>>>(
      frontier.data_ptr<int32_t>(),
      current_frontier_size,
      row_ptr.data_ptr<int32_t>(),
      col_idx.data_ptr<int32_t>(),
      edge_weight.data_ptr<float>(),
      active_nodes.data_ptr<bool>(),
      best_score.data_ptr<float>(),
      frontier_best_score.data_ptr<float>(),
      frontier_best_child.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());

    finalize_frontier_kernel<<<frontier_grid, kBlockSize, 0, stream>>>(
      frontier.data_ptr<int32_t>(),
      current_frontier_size,
      frontier_best_score.data_ptr<float>(),
      frontier_best_child.data_ptr<int32_t>(),
      best_score.data_ptr<float>(),
      best_child.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());

    next_frontier_size.zero_();
    enqueue_parent_frontier_kernel<<<frontier_grid, kBlockSize, 0, stream>>>(
      frontier.data_ptr<int32_t>(),
      current_frontier_size,
      incoming_row_ptr.data_ptr<int32_t>(),
      incoming_col_idx.data_ptr<int32_t>(),
      active_nodes.data_ptr<bool>(),
      remaining_out_degree.data_ptr<int32_t>(),
      next_frontier.data_ptr<int32_t>(),
      next_frontier_size.data_ptr<int32_t>());
    C10_CUDA_CHECK(cudaGetLastError());

    current_frontier_size = next_frontier_size.cpu().item<int32_t>();
    std::swap(frontier, next_frontier);
    std::swap(frontier_size, next_frontier_size);
    }

  build_source_mask_kernel<<<node_grid, kBlockSize, 0, stream>>>(
      active_nodes.data_ptr<bool>(),
      in_degree.data_ptr<int32_t>(),
      best_score.data_ptr<float>(),
      source_mask.data_ptr<bool>(),
      num_nodes);
  C10_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(best_score, best_child, source_mask);
}