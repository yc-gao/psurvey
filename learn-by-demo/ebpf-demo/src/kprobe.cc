// clang-format off
#include <csignal>
#include <iostream>
#include <cstdio>

#include "blazesym.h"
#include "bpf/libbpf.h"

#include "kprobe_bpf.skel.h"
#include "utils.h"
#include "kprobe_common.h"
// clang-format on

static volatile bool running = true;
static struct blaze_symbolizer *symbolizer;

static void print_frame(const char *name, uintptr_t input_addr, uintptr_t addr,
                        uint64_t offset,
                        const blaze_symbolize_code_info *code_info) {
  // If we have an input address  we have a new symbol.
  if (input_addr != 0) {
    printf("%016lx: %s @ 0x%lx+0x%lx", input_addr, name, addr, offset);
    if (code_info != NULL && code_info->dir != NULL &&
        code_info->file != NULL) {
      printf(" %s/%s:%u\n", code_info->dir, code_info->file, code_info->line);
    } else if (code_info != NULL && code_info->file != NULL) {
      printf(" %s:%u\n", code_info->file, code_info->line);
    } else {
      printf("\n");
    }
  } else {
    printf("%16s  %s", "", name);
    if (code_info != NULL && code_info->dir != NULL &&
        code_info->file != NULL) {
      printf("@ %s/%s:%u [inlined]\n", code_info->dir, code_info->file,
             code_info->line);
    } else if (code_info != NULL && code_info->file != NULL) {
      printf("@ %s:%u [inlined]\n", code_info->file, code_info->line);
    } else {
      printf("[inlined]\n");
    }
  }
}

void print_stack(pid_t pid, const __u64 *stack, int stack_sz) {
  const struct blaze_result *result;

  if (pid) {
    struct blaze_symbolize_src_process src = {
        .type_size = sizeof(src),
        .pid = static_cast<uint32_t>(pid),
    };
    result = blaze_symbolize_process_abs_addrs(
        symbolizer, &src, (const uintptr_t *)stack, stack_sz);
  } else {
    struct blaze_symbolize_src_kernel src = {
        .type_size = sizeof(src),
    };
    result = blaze_symbolize_kernel_abs_addrs(
        symbolizer, &src, (const uintptr_t *)stack, stack_sz);
  }

  for (auto i = 0; i < stack_sz; i++) {
    if (!result || result->cnt <= i || result->syms[i].name == NULL) {
      printf("%016llx: <no-symbol>\n", stack[i]);
      continue;
    }

    const struct blaze_sym *sym = &result->syms[i];
    print_frame(sym->name, stack[i], sym->addr, sym->offset, &sym->code_info);
  }

  blaze_result_free(result);
}

static int handle_event([[maybe_unused]] void *ctx, void *data,
                        [[maybe_unused]] size_t data_sz) {
  const struct stacktrace_event *e =
      reinterpret_cast<const struct stacktrace_event *>(data);
  FormatPrefix(std::cout);
  printf(" <%s-%d>\n", e->comm, e->pid);
  if (e->kstack_sz > 0) {
    printf("Kstack:\n");
    print_stack(0, e->kstack, e->kstack_sz / sizeof(__u64));
  }
  if (e->ustack_sz > 0) {
    printf("Ustack:\n");
    print_stack(e->pid, e->ustack, e->ustack_sz / sizeof(__u64));
  }
  return 0;
}

int main(int argc, char *argv[]) {
  symbolizer = blaze_symbolizer_new();
  if (!symbolizer) {
    fprintf(stderr, "Fail to create a symbolizer\n");
    return -1;
  }
  FINALLY([=]() { blaze_symbolizer_free(symbolizer); });

  libbpf_set_print([](enum libbpf_print_level, const char *format,
                      va_list args) { return vfprintf(stderr, format, args); });

  struct kprobe_bpf *skel = kprobe_bpf__open_and_load();
  if (!skel) {
    fprintf(stderr, "Failed to open BPF skeleton\n");
    return -1;
  }
  FINALLY([=]() { kprobe_bpf__destroy(skel); });

  for (int i = 1; i < argc; i++) {
    skel->links.kprobe_hook =
        bpf_program__attach_kprobe(skel->progs.kprobe_hook, false, argv[i]);
    if (!skel->links.kprobe_hook) {
      fprintf(stderr, "Failed to attach uprobe: %s, error: %d\n", argv[i],
              -errno);
      return -1;
    }
  }

  struct ring_buffer *rb =
      ring_buffer__new(bpf_map__fd(skel->maps.rb), handle_event, NULL, NULL);
  if (!rb) {
    fprintf(stderr, "Failed to create ring buffer\n");
    return -1;
  }
  FINALLY([=]() { ring_buffer__free(rb); });

  std::signal(SIGINT, [](int) { running = false; });
  std::signal(SIGTERM, [](int) { running = false; });
  while (running) {
    int err = ring_buffer__poll(rb, 100 /* timeout, ms */);
    if (err == -EINTR) {
      break;
    }
    if (err < 0) {
      fprintf(stderr, "Error polling perf buffer: %d\n", err);
      return -1;
    }
  }

  return 0;
}
