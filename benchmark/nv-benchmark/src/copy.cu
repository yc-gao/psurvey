#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <cstddef>
#include <nvbench/nvbench.cuh>

void cuda_copy(nvbench::state &state) {
  // 64MB
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values);
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec([=, &input, &output](nvbench::launch &launch) {
    cudaMemcpyAsync(thrust::raw_pointer_cast(output.data()),
                    thrust::raw_pointer_cast(input.data()),
                    num_values * sizeof(nvbench::int32_t),
                    cudaMemcpyDeviceToDevice, launch.get_stream());
  });
}
NVBENCH_BENCH(cuda_copy);

template <typename T>
void thrust_copy(nvbench::state &state, nvbench::type_list<T>) {
  // 64MB
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(T);
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
             [=, &input, &output](nvbench::launch &launch) {
               thrust::transform(thrust::device, input.begin(), input.end(),
                                 output.begin(), thrust::identity<T>());
             });
}
NVBENCH_BENCH_TYPES(
    thrust_copy,
    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int8_t, nvbench::int16_t,
                                         nvbench::int32_t, nvbench::int64_t>));

template <typename T>
__global__ void kernel_copy_impl(T *dst, const T *src, std::size_t count) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < count) {
    dst[idx] = src[idx];
    idx += gridDim.x * blockDim.x;
  }
}

template <typename T>
void kernel_copy(nvbench::state &state, nvbench::type_list<T>) {
  // 64MB
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(T);
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
             [=, &input, &output](nvbench::launch &launch) {
               kernel_copy_impl<T><<<(num_values + 255) / 256, 256>>>(
                   thrust::raw_pointer_cast(output.data()),
                   thrust::raw_pointer_cast(input.data()), num_values);
             });
}
NVBENCH_BENCH_TYPES(
    kernel_copy,
    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int8_t, nvbench::int16_t,
                                         nvbench::int32_t, nvbench::int64_t>));
