#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>

#include <cstddef>
#include <nvbench/nvbench.cuh>

template <typename T, std::size_t TILE_DIM = 32>
__global__ void kernel_transpose_impl(T* dst, const T* src, std::size_t M,
                                      std::size_t N) {
  __shared__ T s_tile[TILE_DIM][TILE_DIM + 1];

  {
    const auto x = blockIdx.x * TILE_DIM;
    const auto y = blockIdx.y * TILE_DIM;

    for (int i = 0; i < TILE_DIM && (i + y) < M; i++) {
      if (x + threadIdx.x < N) {
        s_tile[i][threadIdx.x] = src[(y + i) * N + x + threadIdx.x];
      }
    }
    __syncthreads();
  }
  {
    const auto x = blockIdx.y * TILE_DIM;
    const auto y = blockIdx.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM && (i + y) < N; i++) {
      if (x + threadIdx.x < M) {
        dst[(y + i) * M + x + threadIdx.x] = s_tile[i][threadIdx.x];
      }
    }
  }
}
template <typename T>
void kernel_transpose_wrap(T* dst, const T* src, std::size_t M, std::size_t N) {
  const int threadsPerBlock = 32;
  dim3 numBlocks((N + threadsPerBlock - 1) / threadsPerBlock,
                 (M + threadsPerBlock - 1) / threadsPerBlock);
  kernel_transpose_impl<T><<<numBlocks, threadsPerBlock>>>(dst, src, M, N);
}

template <typename T>
void kernel_transpose(nvbench::state& state, nvbench::type_list<T>) {
  // 64MB
  const std::size_t M = 1024;
  const std::size_t N = 1024;
  const std::size_t num_values = M * N;
  thrust::device_vector<T> input(num_values);
  thrust::device_vector<T> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<T>(num_values);
  state.add_global_memory_writes<T>(num_values);

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec([=, &input, &output](nvbench::launch& launch) {
    kernel_transpose_wrap<T>(thrust::raw_pointer_cast(output.data()),
                             thrust::raw_pointer_cast(input.data()), M, N);
  });
}
NVBENCH_BENCH_TYPES(
    kernel_transpose,
    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int8_t, nvbench::int16_t,
                                         nvbench::int32_t, nvbench::int64_t>));

struct transpose_index : public thrust::unary_function<size_t, size_t> {
  size_t m, n;

  __host__ __device__ transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

  __host__ __device__ size_t operator()(size_t linear_index) {
    size_t i = linear_index / n;
    size_t j = linear_index % n;

    return m * j + i;
  }
};
template <typename T>
void thrust_transpose_impl(size_t m, size_t n, thrust::device_vector<T>& src,
                           thrust::device_vector<T>& dst) {
  thrust::counting_iterator<size_t> indices(0);

  thrust::gather(
      thrust::make_transform_iterator(indices, transpose_index(n, m)),
      thrust::make_transform_iterator(indices, transpose_index(n, m)) +
          dst.size(),
      src.begin(), dst.begin());
}

template <typename T>
void thrust_transpose(nvbench::state& state, nvbench::type_list<T>) {
  // 64MB
  const std::size_t M = 1024;
  const std::size_t N = 1024;
  const std::size_t num_values = M * N;
  thrust::device_vector<T> input(num_values);
  thrust::device_vector<T> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<T>(num_values);
  state.add_global_memory_writes<T>(num_values);

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec(nvbench::exec_tag::sync,
             [=, &input, &output](nvbench::launch& launch) {
               thrust_transpose_impl(M, N, input, output);
             });
}
NVBENCH_BENCH_TYPES(
    thrust_transpose,
    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int8_t, nvbench::int16_t,
                                         nvbench::int32_t, nvbench::int64_t>));
