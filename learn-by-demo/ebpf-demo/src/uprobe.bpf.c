#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "uprobe.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 256 * 1024);
} rb SEC(".maps");

SEC("uprobe")
int BPF_KPROBE(uprobe_add) {
  int pid = bpf_get_current_pid_tgid() >> 32;
  int cpu_id = bpf_get_smp_processor_id();

  struct stacktrace_event *event;
  event = bpf_ringbuf_reserve(&rb, sizeof(*event), 0);
  if (!event) return 1;

  event->pid = pid;
  event->cpu_id = cpu_id;
  if (bpf_get_current_comm(event->comm, sizeof(event->comm)))
    event->comm[0] = 0;
  event->ustack_sz = bpf_get_stack(ctx, event->ustack, sizeof(event->ustack),
                                   BPF_F_USER_STACK);
  bpf_ringbuf_submit(event, 0);

  return 0;
}
