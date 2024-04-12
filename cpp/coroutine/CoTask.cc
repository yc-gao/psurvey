#include "CoTask.h"

#include "CoRunner.h"

void CoTask::Yield() { runner_->Yield(this); }
