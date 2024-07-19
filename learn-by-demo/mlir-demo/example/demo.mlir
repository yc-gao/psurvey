func.func @other_func(%arg1 : memref<4xf64>) {
    "demo.print" (%arg1) : (memref<4xf64>) -> ()
    func.return
}
