func.func @m16n8k16_fp16(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}

func.func @ldmatrix_x4(%arg0: memref<128x128xf16, 3>) ->  vector<4x2xf16> {
  %c0  = arith.constant 0 : index
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf16, 3> -> vector<4x2xf16>
  return %a : vector<4x2xf16>
}
