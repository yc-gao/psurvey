#include <iostream>

#include "degine/degine.h"
#include "gflags/gflags.h"

DEFINE_string(model, "", "model to load");
DEFINE_string(format, "onnx", "format of model");

int main(int argc, char *argv[]) {
  degine::DegineContext ctx;

  degine::Model m = ctx.Load(FLAGS_model, FLAGS_format);
  degine::Engine engine = ctx.Build(m); // TODO: params for engine build

  engine.Infer(); // TODO: params for infer

  std::cout << "Hello World!!!" << std::endl;
  return 0;
}
