#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

#include "example_bpf.skel.h"

static volatile sig_atomic_t stop;
static void sig_int(int signo) { stop = 1; }

int main(int argc, char **argv) {
  if (signal(SIGINT, sig_int) == SIG_ERR) {
    fprintf(stderr, "can't set signal handler: %s\n", strerror(errno));
    return 1;
  }

  struct example_bpf *skel;
  skel = example_bpf__open_and_load();
  if (!skel) {
    fprintf(stderr, "Failed to open/load BPF skeleton\n");
    return 1;
  }

  if (example_bpf__attach(skel)) {
    fprintf(stderr, "Failed to attach BPF skeleton\n");
    goto cleanup;
  }

  printf("Successfully started! Please run `sudo cat "
         "/sys/kernel/debug/tracing/trace_pipe` "
         "to see output of the BPF programs.\n");
  while (!stop) {
    sleep(1);
  }

  example_bpf__detach(skel);
cleanup:
  example_bpf__destroy(skel);
  return 0;
}
