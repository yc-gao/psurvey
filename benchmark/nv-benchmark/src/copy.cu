#include <cstddef>
#include <nvbench/nvbench.cuh>

template <typename T>
__global__ void copy_kernel(T *dst, const T *src, std::size_t len) {
  for (auto i = threadIdx.x + blockIdx.x * blockDim.x; i < len;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
  }
}

template <typename ValueType>
void copy_bench(nvbench::state &state, nvbench::type_list<ValueType>) {
  const auto mem_size =
      state.get_int64("MemSize") / sizeof(ValueType) * sizeof(ValueType);
  const auto block_size = state.get_int64("BlockSize");
  const auto num_blocks = state.get_int64("NumBlocks");

  state.add_global_memory_reads<nvbench::uint8_t>(mem_size);
  state.add_global_memory_writes<nvbench::uint8_t>(mem_size);
  state.collect_cupti_metrics();

  ValueType *dst, *src;
  cudaMalloc(&dst, mem_size);
  cudaMalloc(&src, mem_size);

  state.exec([=](nvbench::launch &launch) {
    copy_kernel<<<num_blocks, block_size, 0, launch.get_stream()>>>(
        dst, src, mem_size / sizeof(ValueType));
  });
  cudaFree(dst);
  cudaFree(src);
}
NVBENCH_BENCH_TYPES(
    copy_bench,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<nvbench::uint8_t, nvbench::uint16_t,
                           nvbench::uint32_t, nvbench::uint64_t, int4>))
    .add_int64_power_of_two_axis("MemSize", nvbench::range(10, 30, 10))
    .add_int64_power_of_two_axis("BlockSize", nvbench::range(5, 10, 1))
    .add_int64_power_of_two_axis("NumBlocks", nvbench::range(5, 10, 2));
