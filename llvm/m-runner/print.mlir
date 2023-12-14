llvm.mlir.global internal constant @str_global("String to print\0A")
llvm.func @puts(!llvm.ptr<i8>)

func.func @main() {
  %0 = llvm.mlir.addressof @str_global : !llvm.ptr<array<16 x i8>>
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.getelementptr %0[%1, %1]
    : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
  llvm.call @puts(%2) : (!llvm.ptr<i8>) -> ()
  return
}
