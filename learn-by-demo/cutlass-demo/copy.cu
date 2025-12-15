#include "cute/tensor.hpp"
#include "device_vector.cuh"

template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D,
                                       Tiled_Copy tiled_copy) {
  using namespace cute;

  // Slice the tensors to obtain a view into each tile.
  Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)

  // Construct a Tensor corresponding to each thread's slice.
  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);  // (CopyOp, CopyM, CopyN)
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);  // (CopyOp, CopyM, CopyN)

  // Construct a register-backed Tensor with the same shape as each thread's
  // partition Use make_fragment because the first mode is the instruction-local
  // mode
  Tensor fragment = make_fragment_like(thr_tile_D);  // (CopyOp, CopyM, CopyN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}

int main(int argc, char* argv[]) {
  auto tensor_shape = cute::make_shape(256, 512);

  device_vector<float> d_S(size(tensor_shape));
  device_vector<float> d_D(size(tensor_shape));

  auto tensor_S = cute::make_tensor(cute::make_gmem_ptr(d_S.data()),
                                    cute::make_layout(tensor_shape));
  auto tensor_D = cute::make_tensor(cute::make_gmem_ptr(d_D.data()),
                                    cute::make_layout(tensor_shape));

  auto block_shape = make_shape(cute::Int<128>{}, cute::Int<64>{});
  auto tiled_tensor_S =
      cute::tiled_divide(tensor_S, block_shape);  // ((M, N), m', n')
  auto tiled_tensor_D = cute::tiled_divide(tensor_D, block_shape);

  auto thr_layout =
      cute::make_layout(cute::make_shape(cute::Int<32>{}, cute::Int<8>{}));
  auto val_layout =
      cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<1>{}));

  using CopyOp =
      cute::UniversalCopy<cute::uint_byte_t<sizeof(float) * size(val_layout)>>;
  auto tiled_copy =
      make_tiled_copy(cute::Copy_Atom<CopyOp, float>{},  // Access strategy
                      thr_layout,  // thread layout (e.g. 32x4 Col-Major)
                      val_layout);

  dim3 gridDim(
      cute::size<1>(tiled_tensor_D),
      cute::size<2>(
          tiled_tensor_D));  // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));
  copy_kernel_vectorized<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                                tiled_copy);
  cudaDeviceSynchronize();
  return 0;
}
