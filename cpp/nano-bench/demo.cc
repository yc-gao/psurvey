#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <utility>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[];
};

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return ret;
}

struct PerfMonitor {
  long fd;
  std::uint64_t id;
  std::uint64_t val;

  virtual ~PerfMonitor() {
    if (fd != -1) {
      close(fd);
    }
  }
  PerfMonitor(const PerfMonitor &) = delete;
  PerfMonitor operator==(const PerfMonitor &) = delete;
  PerfMonitor(PerfMonitor &&other)
      : fd(std::exchange(other.fd, -1)), id(std::exchange(other.id, 0)) {}
  PerfMonitor &operator==(PerfMonitor &&other) {
    if (this == &other) {
      return *this;
    }
    fd = std::exchange(other.fd, -1);
    id = std::exchange(other.id, 0);
    return *this;
  }

  PerfMonitor(long fd, std::uint64_t id) : fd(fd), id(id) {}

  static std::shared_ptr<PerfMonitor> Build(struct perf_event_attr *attr,
                                            pid_t pid, int cpu, int group_fd,
                                            unsigned long flags) {
    long fd = perf_event_open(attr, pid, cpu, group_fd, flags);
    if (fd == -1) {
      return nullptr;
    }
    std::uint64_t id;
    if (-1 == ioctl(fd, PERF_EVENT_IOC_ID, &id)) {
      close(fd);
      return nullptr;
    }
    return std::make_shared<PerfMonitor>(PerfMonitor{fd, id});
  }

  friend std::ostream &operator<<(std::ostream &os, const PerfMonitor &p) {
    os << "PerfMonitor{fd=" << p.fd << ", id=" << p.id << "}";
    return os;
  }
};

int test_func() {
  int s = 0;
  for (int i = 0; i < 100; i++) {
    s += i;
  };
  return s;
}

void do_perf() {
  struct perf_event_attr attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof(attr);
  attr.config = PERF_COUNT_HW_CPU_CYCLES;

  attr.disabled = 1;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;

  attr.read_format = PERF_FORMAT_ID | PERF_FORMAT_GROUP;

  auto cycles = PerfMonitor::Build(&attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
  assert(cycles);

  attr.config = PERF_COUNT_HW_INSTRUCTIONS;
  auto insts =
      PerfMonitor::Build(&attr, 0, -1, cycles->fd, PERF_FLAG_FD_CLOEXEC);
  assert(insts);

  ioctl(cycles->fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(cycles->fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
  test_func();
  ioctl(cycles->fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

  char buf[sizeof(read_format) + sizeof(std::uint64_t) * 2 * 2];
  read(cycles->fd, buf, sizeof(read_format) + sizeof(std::uint64_t) * 2 * 2);
  read_format *rbuf = reinterpret_cast<struct read_format *>(buf);

  for (uint64_t i = 0; i < rbuf->nr; i++) {
    if (rbuf->values[i].id == cycles->id) {
      cycles->val = rbuf->values[i].value;
    } else if (rbuf->values[i].id == insts->id) {
      insts->val = rbuf->values[i].value;
    } else {
      assert(0);
    }
  }

  std::cout << "cycles " << cycles->val << std::endl;
  std::cout << "insts " << insts->val << std::endl;
}

void do_nanobench() {
  ankerl::nanobench::Bench().run("some double ops", [] { test_func(); });
}

int main() {
  do_perf();
  do_nanobench();
  return 0;
}
