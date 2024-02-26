#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "degine/ir/Model.h"

namespace degine {
class DegineContext;

class LoaderBase {
  DegineContext *ctx;

public:
  LoaderBase(DegineContext *ctx);

  DegineContext *GetCtx() { return ctx; }
  virtual Model Load(std::string fname) = 0;
};

class LoaderRegistry {
  std::unordered_map<std::string, std::unique_ptr<LoaderBase>> format2loader;

public:
  LoaderBase *GetLoader(std::string format) {
    return format2loader.at(format).get();
  }

  static LoaderRegistry *Instance() {
    static LoaderRegistry inst;
    return &inst;
  }
};

} // namespace degine

#define DECLARE_MODEL_LOADER(format, cls)
// TODO : impl
