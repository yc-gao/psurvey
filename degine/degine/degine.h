#pragma once

#include <memory>
#include <string>

namespace degine {

struct EngineOptions {
  std::string model;
  std::string format;
  std::string accelerator;
};

struct Engine {
  static std::unique_ptr<Engine> BuildEngine(const EngineOptions &);
};

} // namespace degine
