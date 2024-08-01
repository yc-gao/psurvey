#ifndef DEMO_OPS
#define DEMO_OPS

include "mlir/IR/OpBase.td"

def DemoDialect : Dialect {
    let name = "demo";
    let cppNamespace = "::mlir::demo";
}

class DemoOp<string mnemonic, list<Trait> traits = []> :
    Op<DemoDialect, mnemonic, traits>;

def PrintOp : DemoOp<"print"> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    let arguments = (ins F64MemRef:$input);
}

def PrintImplOp : DemoOp<"print_impl"> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    let arguments = (ins F64MemRef:$input);
}

#endif
