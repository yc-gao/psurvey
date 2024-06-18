#include <csignal>

#include "execsnoop.h"
#include "execsnoop_bpf.skel.h"

static int handle_event([[maybe_unused]] void *ctx, void *data,
                        [[maybe_unused]] size_t data_sz) {
  const struct event *e = reinterpret_cast<const struct event *>(data);
  printf("%-16s%-7d %s\n", e->comm, e->pid, e->filename);
  return 0;
}

bool running = true;

int main(int, char *[]) {
  std::signal(SIGINT, [](int) { running = false; });
  std::signal(SIGTERM, [](int) { running = false; });

  int ret = 0;
  struct execsnoop_bpf *skel = nullptr;
  struct ring_buffer *rb = nullptr;

  libbpf_set_print([](enum libbpf_print_level, const char *format,
                      va_list args) { return vfprintf(stderr, format, args); });

  skel = execsnoop_bpf__open_and_load();
  if (!skel) {
    fprintf(stderr, "Failed to open BPF skeleton\n");
    ret = 1;
    goto err_open_load;
  }

  if (execsnoop_bpf__attach(skel)) {
    fprintf(stderr, "Failed to attach BPF skeleton\n");
    ret = 1;
    goto err_attach;
  }

  rb = ring_buffer__new(bpf_map__fd(skel->maps.rb), handle_event, NULL, NULL);
  if (!rb) {
    fprintf(stderr, "Failed to create ring buffer\n");
    ret = 1;
    goto err_ring;
  }

  while (running) {
    int err = ring_buffer__poll(rb, 100 /* timeout, ms */);
    if (err == -EINTR) {
      break;
    }
    if (err < 0) {
      fprintf(stderr, "Error polling perf buffer: %d\n", err);
      ret = 1;
      goto err_poll;
    }
  }

err_poll:
  ring_buffer__free(rb);
err_ring:
err_attach:
  execsnoop_bpf__destroy(skel);
err_open_load:
  return ret;
}
