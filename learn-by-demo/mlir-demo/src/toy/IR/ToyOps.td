#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/BuiltinDialect.td"

def ToyDialect : Dialect {
    let name = "toy";
    let cppNamespace = "::mlir::toy";
}

class ToyOp<string mnemonic, list<Trait> traits = []> :
    Op<ToyDialect, mnemonic, traits>;

def ToyConstantOp : ToyOp<"constant"> {
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:
        ```mlir
        %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                            : tensor<2x3xf64>
        ```
    }];
    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);
}

def ToyPrintOp : ToyOp<"print"> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}

#endif

