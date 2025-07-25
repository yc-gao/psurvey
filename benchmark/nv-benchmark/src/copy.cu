#include <cstddef>
#include <cstdint>

#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

void cuda_copy_bench(nvbench::state &state) {
  const auto mem_size = state.get_int64("MemSize");

  thrust::device_vector<std::int8_t> src_vec(mem_size);
  thrust::device_vector<std::int8_t> dst_vec(mem_size);

  state.add_element_count(mem_size);
  state.add_global_memory_reads<std::int8_t>(mem_size);
  state.add_global_memory_writes<std::int8_t>(mem_size);

  state.exec([&](nvbench::launch &launch) {
    cudaMemcpyAsync(thrust::raw_pointer_cast(dst_vec.data()),
                    thrust::raw_pointer_cast(src_vec.data()), mem_size,
                    cudaMemcpyDeviceToDevice, launch.get_stream());
  });
}
NVBENCH_BENCH(cuda_copy_bench)
    .add_int64_axis("MemSize", {1ul << 10, 1ul << 20, 1ul << 30});

template <typename T>
__global__ void kernel_copy_bench_kernel(T *dst, T const *src, std::size_t n) {
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    dst[idx] = src[idx];
  }
}
template <typename T>
void kernel_copy_bench(nvbench::state &state, nvbench::type_list<T>) {
  const auto mem_size = state.get_int64("MemSize");
  const auto grid_size = state.get_int64("GridSize");
  const auto block_size = state.get_int64("BlockSize");

  const auto elem_count = mem_size / sizeof(T);

  thrust::device_vector<std::int8_t> src_vec(mem_size);
  thrust::device_vector<std::int8_t> dst_vec(mem_size);

  state.add_element_count(elem_count);
  state.add_global_memory_reads<T>(elem_count);
  state.add_global_memory_writes<T>(elem_count);

  state.exec([&](nvbench::launch &launch) {
    kernel_copy_bench_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
            reinterpret_cast<T*>(thrust::raw_pointer_cast(dst_vec.data())),
            reinterpret_cast<T const *>(thrust::raw_pointer_cast(src_vec.data())),
            elem_count);
  });
}
NVBENCH_BENCH_TYPES(
    kernel_copy_bench,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<std::int8_t, std::int16_t, std::int32_t, std::int64_t, int4>))
    .add_int64_axis("MemSize", {1ul << 10, 1ul << 20, 1ul << 30})
    .add_int64_axis("GridSize", {128, 512, 1024})
    .add_int64_axis("BlockSize", {128, 512, 1024});

