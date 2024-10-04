#include <cutlass/gemm/device/gemm.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <nvbench/nvbench.cuh>

void cutlass_gemm(nvbench::state &state) {
  thrust::device_vector<float> input(1024 * 1024);
  thrust::device_vector<float> output(1024 * 1024);

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,         // Data-type of A matrix
                                  ColumnMajor,   // Layout of A matrix
                                  float,         // Data-type of B matrix
                                  ColumnMajor,   // Layout of B matrix
                                  float,         // Data-type of C matrix
                                  ColumnMajor>;  // Layout of C matrix
  CutlassGemm gemm_operator;
  CutlassGemm::Arguments args({1024, 1024, 1024},  // Gemm Problem dimensions
                              {thrust::raw_pointer_cast(input.data()), 1024},
                              {thrust::raw_pointer_cast(input.data()), 1024},
                              {thrust::raw_pointer_cast(output.data()), 1024},
                              {thrust::raw_pointer_cast(output.data()), 1024},
                              {1, 0});  // Scalars used in the Epilogue

  // Provide throughput information:
  state.add_global_memory_reads<float>(input.size());
  state.add_global_memory_reads<float>(input.size());
  state.add_global_memory_writes<float>(output.size());

  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  state.exec([&](nvbench::launch &launch) { gemm_operator(args); });
}
NVBENCH_BENCH(cutlass_gemm);
