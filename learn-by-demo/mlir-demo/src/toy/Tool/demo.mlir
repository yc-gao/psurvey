module {
    func.func @main() {
        %0 = "toy.constant" () {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
        "toy.print" (%0) : (tensor<2x3xf64>) -> ()
        func.return
    }
}
