#include <thrust/device_vector.h>

#include <nvbench/nvbench.cuh>

void cuda_copy(nvbench::state &state) {
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values);
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.exec([=, &input, &output](nvbench::launch &launch) {
    cudaMemcpyAsync(thrust::raw_pointer_cast(input.data()),
                    thrust::raw_pointer_cast(output.data()),
                    num_values * sizeof(nvbench::int32_t),
                    cudaMemcpyDeviceToDevice, launch.get_stream());
  });
}
NVBENCH_BENCH(cuda_copy);

void thrust_copy(nvbench::state &state) {
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values);
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.exec(nvbench::exec_tag::sync,
             [=, &input, &output](nvbench::launch &launch) {
               thrust::copy(output.begin(), output.end(), input.begin());
             });
}
NVBENCH_BENCH(thrust_copy);

template <typename T>
__global__ void kernel_copy_impl(T *dest, const T *src, int N) {
  auto step = gridDim.x * blockDim.x;
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid < N; tid += step) {
    dest[tid] = src[tid];
  }
}
template <typename T>
void kernel_copy(nvbench::state &state, nvbench::type_list<T>) {
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(T);
  thrust::device_vector<T> input(num_values);
  thrust::device_vector<T> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<T>(num_values);
  state.add_global_memory_writes<T>(num_values);

  state.exec([=, &state, &input, &output](nvbench::launch &launch) {
    kernel_copy_impl<<<state.get_int64("blocks"), state.get_int64("threads"), 0,
                       launch.get_stream()>>>(
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()), num_values);
  });
}
NVBENCH_BENCH_TYPES(
    kernel_copy,
    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int8_t, nvbench::int16_t,
                                         nvbench::int32_t, nvbench::int64_t>))
    .add_int64_axis("blocks", nvbench::range(256, 1024, 256))
    .add_int64_axis("threads", nvbench::range(256, 1024, 256));
