#ifndef DEMO_DIALECT
#define DEMO_DIALECT

include "mlir/IR/OpBase.td"

def Demo_Dialect : Dialect {
    let name = "demo";
    let cppNamespace = "::mlir::demo";
    let useFoldAPI = kEmitFoldAdaptorFolder;
}

class Demo_Op<string mnemonic, list<Trait> traits = []> : Op<Demo_Dialect, mnemonic, traits>;

#endif
