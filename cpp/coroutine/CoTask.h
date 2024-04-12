#pragma once

#include <functional>

#include "arch.h"

class CoRunner;

class CoTask {
  friend class CoRunner;
  CoRunner *runner_;
  std::function<void()> func_;

  char stack[8192];
  void *sp;

  CoTask(const CoTask &) = delete;
  CoTask(CoTask &&) = delete;
  CoTask &operator=(CoTask) = delete;

  static void Init(CoTask *task) {
    task->func_();
    std::exchange(task->func_, std::function<void()>());
    task->Yield();
  }

public:
  CoTask(CoRunner *runner, std::function<void()> func)
      : runner_(runner), func_(std::move(func)), sp((void *)std::end(stack)) {
    CoInit(&sp, (void (*)(void *))CoTask::Init, this);
  }
  void Yield();
};
