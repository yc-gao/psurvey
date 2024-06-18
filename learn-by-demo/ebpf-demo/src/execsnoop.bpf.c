#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "execsnoop.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 256 * 1024);
} rb SEC(".maps");

const volatile unsigned long long min_duration_ns = 0;

SEC("tp/sched/sched_process_exec")
int handle_exec(struct trace_event_raw_sched_process_exec *ctx) {
  struct event *e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
  if (!e) return 0;

  e->pid = bpf_get_current_pid_tgid() >> 32;
  bpf_get_current_comm(&e->comm, sizeof(e->comm));
  bpf_probe_read_str(&e->filename, sizeof(e->filename), (char*)ctx + (ctx->__data_loc_filename & 0xFFFF));

  bpf_ringbuf_submit(e, 0);
  return 0;
}

