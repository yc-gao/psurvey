#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

class PerfMonitor {
  struct read_format {
    std::uint64_t nr;
    std::uint64_t time_enabled;
    std::uint64_t time_running;
    struct record {
      std::uint64_t value;
      std::uint64_t id;
    } cntr[];
  };

  int group_fd{-1};

  std::vector<std::uint64_t> perf_buf;
  std::unordered_map<std::uint64_t, std::uint64_t *> id2buf;

  std::uint64_t time_enabled;
  std::uint64_t time_running;

public:
  int Fd() { return group_fd; }
  std::uint64_t TimeEnabled() const { return time_enabled; }
  std::uint64_t TimeRunning() const { return time_running; }

  void Monitor(struct perf_event_attr *attr, pid_t pid, int cpu,
               std::uint64_t *buf) {
    int fd = syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, 0);
    if (fd == -1) {
      throw std::system_error(errno, std::generic_category());
    }
    std::uint64_t id;
    if (-1 == ioctl(fd, PERF_EVENT_IOC_ID, &id)) {
      throw std::system_error(errno, std::generic_category());
    }
    id2buf.emplace(id, buf);
    if (group_fd == -1) {
      group_fd = fd;
    }
  }
  void Monitor(perf_type_id type, std::uint64_t config, pid_t pid, int cpu,
               std::uint64_t *buf) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = type;
    attr.size = sizeof(attr);
    attr.config = config;

    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID |
                       PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING;
    Monitor(&attr, pid, cpu, buf);
  }
  void Monitor(perf_hw_id config, int pid, int cpu, std::uint64_t *buf) {
    Monitor(PERF_TYPE_HARDWARE, config, pid, cpu, buf);
  }

  void Monitor(perf_sw_ids config, int pid, int cpu, std::uint64_t *buf) {
    Monitor(PERF_TYPE_SOFTWARE, config, pid, cpu, buf);
  }

  void Reset() {
    if (-1 == ioctl(group_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP)) {
      throw std::runtime_error("can not reset perf event");
    }
  }
  void Begin(bool reset = true) {
    if (reset) {
      Reset();
    }
    if (-1 == ioctl(group_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)) {
      throw std::runtime_error("can not enable perf event");
    }
    perf_buf.resize(sizeof(read_format) +
                    sizeof(read_format::record) * id2buf.size());
  }
  void End() {
    if (-1 == ioctl(group_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)) {
      throw std::runtime_error("can not enable perf event");
    }
  }
  void Update() {
    auto size = read(group_fd, perf_buf.data(), perf_buf.size());
    if (size == -1) {
      throw std::system_error(errno, std::generic_category());
    } else if (size == 0) {
      throw std::runtime_error("can not read perf event");
    }
    read_format *header = (read_format *)perf_buf.data();
    time_enabled = header->time_enabled;
    time_running = header->time_running;

    for (std::uint64_t i = 0; i < header->nr; i++) {
      *id2buf.at(header->cntr[i].id) = header->cntr[i].value;
    }
  }
};
