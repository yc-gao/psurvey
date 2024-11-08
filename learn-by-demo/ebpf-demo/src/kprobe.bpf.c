// clang-format off
#include "vmlinux.h"
#include "bpf/bpf_helpers.h"
#include "bpf/bpf_tracing.h"
#include "bpf/bpf_core_read.h"

#include "kprobe_common.h"
// clang-format on

char LICENSE[] SEC("license") = "Dual BSD/GPL";

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 256 * 1024);
} rb SEC(".maps");

SEC("kprobe")
int BPF_KPROBE(kprobe_hook) {
  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  int pid = BPF_CORE_READ(task, pid);

  struct stacktrace_event *event;
  event = bpf_ringbuf_reserve(&rb, sizeof(*event), 0);
  if (!event) return 1;

  event->pid = pid;
  if (bpf_get_current_comm(event->comm, sizeof(event->comm)))
    event->comm[0] = 0;

  event->ustack_sz = bpf_get_stack(ctx, event->ustack, sizeof(event->ustack),
                                   BPF_F_USER_STACK);
  event->kstack_sz =
      bpf_get_stack(ctx, event->kstack, sizeof(event->kstack), 0);
  bpf_ringbuf_submit(event, 0);
  return 0;
}
