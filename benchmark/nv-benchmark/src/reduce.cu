#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <cstddef>
#include <cub/cub.cuh>
#include <nvbench/nvbench.cuh>

template <typename T>
void thrust_reduce(nvbench::state &state, nvbench::type_list<T>) {
  // 64MB
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(T);
  thrust::device_vector<T> input(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<T>(num_values);

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec(nvbench::exec_tag::sync, [=, &input](nvbench::launch &launch) {
    thrust::reduce(input.begin(), input.end(), T(), thrust::plus<T>());
  });
}
NVBENCH_BENCH_TYPES(
    thrust_reduce,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t,
                           nvbench::int64_t, nvbench::float32_t,
                           nvbench::float64_t>));

template <typename T>
void cub_reduce(nvbench::state &state, nvbench::type_list<T>) {
  // 64MB
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(T);
  thrust::device_vector<T> input(num_values);
  thrust::device_vector<T> output(1);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
                            thrust::raw_pointer_cast(input.data()),
                            thrust::raw_pointer_cast(output.data()),
                            input.size(), thrust::plus<T>(), T());
  thrust::device_vector<char> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<T>(num_values);

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec(nvbench::exec_tag::sync, [=, &input, &output](
                                          nvbench::launch &launch) mutable {
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
                              thrust::raw_pointer_cast(input.data()),
                              thrust::raw_pointer_cast(output.data()),
                              input.size(), thrust::plus<T>(), T());
  });
}
NVBENCH_BENCH_TYPES(
    cub_reduce,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t,
                           nvbench::int64_t, nvbench::float32_t,
                           nvbench::float64_t>));

template <typename T>
T __inline__ __device__ WarpReduce(T val) {
  // all thread return sum val
  // Use XOR mode to perform butterfly reduction
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, i, 32);
  }
  return val;
}

template <typename T>
T __inline__ __device__ BlockReduce(T val) {
  // first thread return sum val
  auto warpIdx = threadIdx.x / 32;
  auto laneIdx = threadIdx.x % 32;

  __shared__ T s[32];
  val = WarpReduce(val);
  if (laneIdx == 0) {
    s[warpIdx] = val;
  }
  __syncthreads();
  val = T();
  if (threadIdx.x < 32 && threadIdx.x * 32 < blockDim.x) {
    val = s[threadIdx.x];
  }
  val = WarpReduce(val);

  return val;
}

template <typename T>
void __global__ reduce_kernel(T *dst, T const *src, std::size_t len) {
  T val = T();

  // thread reduce
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len;
       idx += blockDim.x * gridDim.x) {
    val += src[idx];
  }

  val = BlockReduce(val);
  if (!threadIdx.x) {
    dst[blockIdx.x] = val;
  }
}

template <typename ValueType>
void kernel_reduce(nvbench::state &state, nvbench::type_list<ValueType>) {
  const auto num_vals = 64 * 1024 * 1024 / sizeof(ValueType);
  const auto block_size = state.get_int64("BlockSize");
  const auto num_blocks = state.get_int64("NumBlocks");

  ValueType *dst, *src, *temp;
  cudaMalloc(&dst, sizeof(ValueType));
  cudaMalloc(&src, num_vals * sizeof(ValueType));
  cudaMalloc(&temp, num_blocks * sizeof(ValueType));

  state.add_element_count(num_vals, "NumVals");
  state.add_global_memory_reads<ValueType>(num_vals);
  state.add_global_memory_writes<ValueType>(num_blocks);

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec([=](nvbench::launch &launch) {
    reduce_kernel<<<num_blocks, block_size, 0, launch.get_stream()>>>(temp, src,
                                                                      num_vals);
    reduce_kernel<<<1, 256, 0, launch.get_stream()>>>(dst, temp, num_blocks);
  });
  cudaFree(dst);
  cudaFree(src);
  cudaFree(temp);
}
NVBENCH_BENCH_TYPES(
    kernel_reduce,
    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t, nvbench::float32_t>))
    .add_int64_axis("NumBlocks", {128, 256, 512, 1024})
    .add_int64_axis("BlockSize", {128, 256, 512, 1024});
