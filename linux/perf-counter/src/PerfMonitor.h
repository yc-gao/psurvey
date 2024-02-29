#pragma once

#include <cstdint>
#include <cstring>

#include <linux/perf_event.h>
#include <memory>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>

class PerfMonitor {
  int perf_fd;

  std::uint64_t buf_size;
  std::unique_ptr<char[]> buf;
  std::unordered_map<std::uint64_t, std::uint64_t *> id2buf;

  std::uint64_t time_enabled;
  std::uint64_t time_running;

public:
  PerfMonitor(std::size_t buf_size = 4096)
      : perf_fd(-1), buf_size(buf_size), buf(new char[buf_size]) {}

  void Monitor(struct perf_event_attr *attr, int pid = 0, int cpu = -1,
               std::uint64_t *buf = nullptr) {
    int fd = syscall(SYS_perf_event_open, attr, pid, cpu, perf_fd, 0);
    if (fd == -1) {
      throw std::system_error(errno, std::generic_category());
    }
    std::uint64_t id;
    if (-1 == ioctl(fd, PERF_EVENT_IOC_ID, &id)) {
      throw std::system_error(errno, std::generic_category());
    }
    if (perf_fd == -1) {
      perf_fd = fd;
    }
    id2buf.emplace(id, buf);
  }

  void Monitor(perf_type_id type, std::uint64_t eventid, int pid = 0,
               int cpu = -1, std::uint64_t *buf = nullptr) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = type;
    attr.size = sizeof(attr);
    attr.config = eventid;

    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID |
                       PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING;
    Monitor(&attr, pid, cpu, buf);
  }

  void Monitor(perf_hw_id eventid, int pid = 0, int cpu = -1,
               std::uint64_t *buf = nullptr) {
    Monitor(PERF_TYPE_HARDWARE, eventid, pid, cpu, buf);
  }

  void Monitor(perf_sw_ids eventid, int pid = 0, int cpu = -1,
               std::uint64_t *buf = nullptr) {
    Monitor(PERF_TYPE_SOFTWARE, eventid, pid, cpu, buf);
  }

  void Begin() {
    if (-1 == ioctl(perf_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP)) {
      throw std::runtime_error("can not reset perf event");
    }
    if (-1 == ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)) {
      throw std::runtime_error("can not enable perf event");
    }
  }
  void End() {
    if (-1 == ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)) {
      throw std::runtime_error("can not enable perf event");
    }
  }

  void Update() {
    auto size = read(perf_fd, buf.get(), buf_size);
    if (size == -1) {
      throw std::system_error(errno, std::generic_category());
    } else if (size == 0) {
      throw std::runtime_error("can not read perf event");
    }
    struct read_format {
      std::uint64_t nr;
      std::uint64_t time_enabled;
      std::uint64_t time_running;
      struct {
        std::uint64_t value;
        std::uint64_t id;
      } cntr[];
    };
    read_format *header = (read_format *)buf.get();
    time_enabled = header->time_enabled;
    time_running = header->time_running;

    for (std::uint64_t i = 0; i < header->nr; i++) {
      *id2buf.at(header->cntr[i].id) = header->cntr[i].value;
    }
  }
};
