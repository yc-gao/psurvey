#ifndef DEMO_OPS
#define DEMO_OPS

include "Demo.td"

def ConstantOp: Demo_Op<"constant"> {
    let arguments = (ins I32);
    let results = (outs I32);
}

#endif
