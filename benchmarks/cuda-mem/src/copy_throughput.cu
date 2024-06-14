#include <thrust/device_vector.h>

#include <nvbench/nvbench.cuh>

template <typename T>
__global__ void copy_kernel(const T *in, T *out, std::size_t n) {
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step) {
    out[i] = in[i];
  }
}

template <typename T>
void kernel_copy(nvbench::state &state, nvbench::type_list<T>) {
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(T);

  // Allocate input data:
  thrust::device_vector<T> input(num_values);
  thrust::device_vector<T> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<T>(num_values, "DataSize");
  state.add_global_memory_writes<T>(num_values);

  state.exec([=, &input, &output](nvbench::launch &launch) {
    copy_kernel<<<256, 256, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()), num_values);
  });
}
NVBENCH_BENCH_TYPES(
    kernel_copy,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<
            nvbench::int8_t, nvbench::int16_t, nvbench::int32_t,
            nvbench::int64_t, std::array<char, 8>, std::array<std::int64_t, 1>,
            std::array<std::int64_t, 2>, std::array<std::int64_t, 4>>));

void api_copy(nvbench::state &state) {
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values, "DataSize");
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.exec([=, &input, &output](nvbench::launch &launch) {
    cudaMemcpyAsync(thrust::raw_pointer_cast(input.data()),
                    thrust::raw_pointer_cast(output.data()),
                    num_values * sizeof(nvbench::int32_t),
                    cudaMemcpyDeviceToDevice, launch.get_stream());
  });
}

NVBENCH_BENCH(api_copy);
