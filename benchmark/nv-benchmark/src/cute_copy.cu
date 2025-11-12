#include <nvbench/nvbench.cuh>

#include "CuteCopyBasic.h"
#include "device_vector.h"

void cute_copy_bench(nvbench::state &state) {
  const auto mem_size = state.get_int64("MemSize");
  state.add_global_memory_reads(mem_size);
  state.add_global_memory_writes(mem_size);

  device_vector<std::int8_t> src_vec(mem_size);
  device_vector<std::int8_t> dst_vec(mem_size);

  state.exec([&](nvbench::launch &launch) {
    CuteCopyBasic()(dst_vec.data(), src_vec.data(), dst_vec.size(),
                    launch.get_stream());
  });
}
NVBENCH_BENCH(cute_copy_bench)
    .add_int64_axis("MemSize", {1ul << 20, 1ul << 30});
