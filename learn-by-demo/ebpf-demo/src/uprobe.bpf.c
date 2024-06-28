// clang-format off
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#include "uprobe.h"
// clang-format on

char LICENSE[] SEC("license") = "Dual BSD/GPL";

const volatile int filter_pid = -1;
const volatile int filter_ppid = -1;
const volatile int filter_tgid = -1;

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 256 * 1024);
} rb SEC(".maps");

SEC("uprobe")
int BPF_KPROBE(uprobe_add) {
  struct task_struct *task = (struct task_struct *)bpf_get_current_task();

  int pid = BPF_CORE_READ(task, pid);
  int ppid = BPF_CORE_READ(task, real_parent, tgid);
  int tgid = BPF_CORE_READ(task, tgid);

  if (filter_pid != -1 && filter_pid != pid) {
    return 0;
  }
  if (filter_ppid != -1 && filter_ppid != ppid) {
    return 0;
  }
  if (filter_tgid != -1 && filter_tgid != tgid) {
    return 0;
  }

  struct stacktrace_event *event;
  event = bpf_ringbuf_reserve(&rb, sizeof(*event), 0);
  if (!event) return 1;

  event->pid = pid;
  if (bpf_get_current_comm(event->comm, sizeof(event->comm)))
    event->comm[0] = 0;
  event->ustack_sz = bpf_get_stack(ctx, event->ustack, sizeof(event->ustack),
                                   BPF_F_USER_STACK);
  bpf_ringbuf_submit(event, 0);

  return 0;
}
