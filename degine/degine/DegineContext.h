#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "degine/Engine.h"
#include "degine/Loader.h"
#include "degine/ir/Model.h"

namespace degine {

class DegineContext;

class DegineContext {

public:
  Model Load(std::string fname, std::string format = "") {
    if (format.empty()) {
      format = fname.substr(fname.find_last_of(".") + 1);
    }
    if (format.empty()) {
      throw std::invalid_argument("empty model format");
    }
    return LoaderRegistry::Instance()->GetLoader(format)->Load(fname);
  }

  Engine Build(Model m) {
    // TODO: impl
    return Engine{};
  }
};

} // namespace degine
