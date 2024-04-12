#pragma once

#include <deque>
#include <functional>
#include <memory>

#include "CoTask.h"

class CoRunner {
  friend class CoTask;

  bool running_{true};

  std::deque<std::unique_ptr<CoTask>> tasks;
  void *sp;

  void Yield(CoTask *);

  CoRunner(const CoRunner &) = delete;
  CoRunner(CoRunner &&) = delete;
  CoRunner operator=(CoRunner) = delete;

public:
  CoRunner() = default;
  void Dispatch(std::function<void()> func);
  void Run();
};
