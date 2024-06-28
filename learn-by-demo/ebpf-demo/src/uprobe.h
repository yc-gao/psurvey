#pragma once

#define TASK_COMM_LEN 16

#ifndef TASK_COMM_LEN
#define TASK_COMM_LEN 16
#endif

#ifndef MAX_STACK_DEPTH
#define MAX_STACK_DEPTH 128
#endif

typedef __u64 stack_trace_t[MAX_STACK_DEPTH];

struct stacktrace_event {
  int pid;
  char comm[TASK_COMM_LEN];
  __s32 ustack_sz;
  stack_trace_t ustack;
};
