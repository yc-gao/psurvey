#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

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
