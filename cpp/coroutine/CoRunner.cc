#include "CoRunner.h"

#include "CoTask.h"
#include "arch.h"

void CoRunner::Yield(CoTask *task) { CoSwitch(&task->sp, &sp); }

void CoRunner::Dispatch(std::function<void()> func) {
  tasks.emplace_back(std::make_unique<CoTask>(this, std::move(func)));
}

void CoRunner::Run() {
  while (running_ && !tasks.empty()) {
    auto task = std::move(tasks.front());
    tasks.pop_front();
    CoSwitch(&sp, &task->sp);
    if (task->func_) {
      tasks.emplace_back(std::move(task));
    }
  }
}
