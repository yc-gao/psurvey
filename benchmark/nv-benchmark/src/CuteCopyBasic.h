#pragma once
#include <cute/tensor.hpp>

#include "device_vector.h"

class CuteCopyBasic {
 public:
  template <typename T>
  void operator()(T *dst, T const *src, int n, cudaStream_t stream = 0) {
    auto tensor_shape = cute::make_shape(n);
    auto tensor_s = cute::make_tensor(src, cute::make_layout(tensor_shape));
    auto tensor_d = cute::make_tensor(dst, cute::make_layout(tensor_shape));

    auto block_shape = cute::make_shape(cute::Int<1024>{});  // elem per block
    auto block_tiled_tensor_s = cute::tiled_divide(tensor_s, block_shape);
    auto block_tiled_tensor_d = cute::tiled_divide(tensor_d, block_shape);

    auto thread_layout =
        cute::make_shape(cute::Int<256>{});  // thread per block
    auto thread_tiled_tensor_s =
        cute::tiled_divide(block_tiled_tensor_s, thread_layout);
    auto thread_tiled_tensor_d =
        cute::tiled_divide(block_tiled_tensor_d, thread_layout);

    dim3 gridDim(cute::size<1>(block_tiled_tensor_d));
    dim3 blockDim(cute::size(thread_layout));

    kernel_launcher<<<gridDim, blockDim, 0, stream>>>(
        *this, thread_tiled_tensor_d, thread_tiled_tensor_s, n);
  }

  template <typename TensorDst, typename TensorSrc>
  __device__ void Launch(TensorDst dst, TensorSrc src, int n) {
    auto tensor_s = src(threadIdx.x, cute::_, blockIdx.x);
    auto tensor_d = dst(threadIdx.x, cute::_, blockIdx.x);
    auto fragment = cute::make_fragment_like(tensor_d);
    cute::copy(tensor_s, fragment);
    cute::copy(fragment, tensor_d);
  }
};
