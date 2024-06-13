#include <csignal>

#include "bootstrap.h"
#include "bootstrap_bpf.skel.h"

static int libbpf_print_fn([[maybe_unused]] enum libbpf_print_level level,
                           const char *format, va_list args) {
  return vfprintf(stderr, format, args);
}

static int handle_event([[maybe_unused]] void *ctx, void *data,
                        [[maybe_unused]] size_t data_sz) {
  const struct event *e = reinterpret_cast<const struct event *>(data);
  printf("%-8d%-8s\n", e->pid, e->comm);
  return 0;
}

bool running = true;

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
  std::signal(SIGINT, [](int) { running = false; });
  std::signal(SIGTERM, [](int) { running = false; });

  libbpf_set_print(libbpf_print_fn);
  int ret = 0;

  struct bootstrap_bpf *skel = nullptr;
  struct ring_buffer *rb = nullptr;

  skel = bootstrap_bpf__open_and_load();
  if (!skel) {
    fprintf(stderr, "Failed to open BPF skeleton\n");
    ret = 1;
    goto err_open_load;
  }

  if (bootstrap_bpf__attach(skel)) {
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
  bootstrap_bpf__destroy(skel);
err_open_load:
  return ret;
}
