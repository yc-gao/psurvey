#include <bpf/libbpf.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

#include "minimal_bpf.skel.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
                           va_list args) {
  return vfprintf(stderr, format, args);
}

int main(int argc, char **argv) {
  libbpf_set_print(libbpf_print_fn);

  struct minimal_bpf *skel = minimal_bpf__open_and_load();
  if (!skel) {
    fprintf(stderr, "Failed to open BPF skeleton\n");
    return 1;
  }

  if (minimal_bpf__attach(skel)) {
    fprintf(stderr, "Failed to attach BPF skeleton\n");
    goto cleanup;
  }

  printf(
      "Successfully started! Please run `sudo cat "
      "/sys/kernel/debug/tracing/trace_pipe` "
      "to see output of the BPF programs.\n");

  for (;;) {
    fprintf(stderr, ".");
    sleep(1);
  }

cleanup:
  minimal_bpf__destroy(skel);
  return 0;
}
