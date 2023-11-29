#ifndef DEMO_OPS
#define DEMO_OPS

include "mlir/IR/OpBase.td"

def Demo_Dialect : Dialect {
    let name = "demo";
    let cppNamespace = "::mlir::demo";
    let useFoldAPI = kEmitFoldAdaptorFolder;
}

class Demo_Op<string mnemonic, list<Trait> traits = []> :
    Op<Demo_Dialect, mnemonic, traits>;

def ConstantOp : Demo_Op<"constant"> {
    let summary = "constant";
    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);
}


#endif
