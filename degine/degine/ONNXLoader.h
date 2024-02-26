#pragma once
#include "degine/Loader.h"

namespace degine {

class ONNXLoader : public LoaderBase {
public:
  using LoaderBase::LoaderBase;
  Model Load(std::string fname) {
    // TODO: impl
    return Model{};
  }
};

} // namespace degine

DECLARE_MODEL_LOADER("onnx", degine::ONNXLoader)
