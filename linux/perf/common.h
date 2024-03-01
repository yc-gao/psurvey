#pragma once

#include <linux/perf_event.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>

namespace {

bool running{true};
long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                     int group_fd, unsigned long flags) {
  int ret;

  ret = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return ret;
}

} // namespace

inline int do_perf0(int argc, char *argv[]) {
  pid_t pid = 0;
  int cpu = -1;
  std::uint64_t tm = 1000000000;
  for (int i = 1; i < argc;) {
    if (strcmp(argv[i], "--pid") == 0 || strcmp(argv[i], "-p") == 0) {
      pid = std::stoul(argv[i + 1]);
      i += 2;
    } else if (strcmp(argv[i], "--cpu") == 0) {
      cpu = std::stoul(argv[i + 1]);
      i += 2;
    } else if (strcmp(argv[i], "-d") == 0) {
      tm = std::stoul(argv[i + 1]);
      i += 2;
    } else {
      i++;
    }
  }

  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(pe);
  pe.config = PERF_COUNT_HW_CPU_CYCLES;

  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  pe.exclude_guest = 1;

  int fd = perf_event_open(&pe, pid, cpu, -1, 0);
  if (fd == -1) {
    perror("");
    return 1;
  }
  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  std::this_thread::sleep_for(std::chrono::nanoseconds(tm));
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

  std::uint64_t count;
  read(fd, &count, sizeof(count));
  std::cout << "count " << count << std::endl;
  close(fd);

  return 0;
}

inline int do_perf1(int argc, char *argv[]) {
  pid_t pid = 0;
  int cpu = -1;
  std::uint64_t tm = 1000000000;
  for (int i = 1; i < argc;) {
    if (strcmp(argv[i], "--pid") == 0 || strcmp(argv[i], "-p") == 0) {
      pid = std::stoul(argv[i + 1]);
      i += 2;
    } else if (strcmp(argv[i], "--cpu") == 0) {
      cpu = std::stoul(argv[i + 1]);
      i += 2;
    } else if (strcmp(argv[i], "-d") == 0) {
      tm = std::stoul(argv[i + 1]);
      i += 2;
    } else {
      i++;
    }
  }

  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_TRACEPOINT;
  pe.size = sizeof(pe);
  pe.config = 379;

  pe.disabled = 1;
  // pe.inherit = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  pe.exclude_guest = 1;

  // pe.freq = 1;
  // pe.sample_freq = 99;
  pe.sample_period = 1;
  pe.wakeup_events = 1;
  pe.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_CPU;

  int fd = perf_event_open(&pe, pid, cpu, -1, 0);
  if (fd == -1) {
    perror("");
    return 1;
  }

  void *buf =
      mmap(0, 9 * getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (buf == MAP_FAILED) {
    perror("");
    return 1;
  }

  struct pollfd pfd = {.fd = fd, .events = POLLIN};
  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  while (running) {
    switch (poll(&pfd, 1, -1)) {
    case -1: {
      if (errno == EINTR) {
        continue;
      } else {
        perror("");
        return 1;
      }

    } break;
    case 0:
      continue;
      break;
    default:
      break;
    }
    struct perf_event_mmap_page *info = (struct perf_event_mmap_page *)buf;
    while (info->data_tail < info->data_head) {
      struct perf_event_header *header =
          (struct perf_event_header *)((char *)buf + info->data_offset +
                                       (info->data_tail % info->data_size));
      switch (header->type) {
      case PERF_RECORD_SAMPLE: {
        std::cout << *(std::uint64_t *)(header + 1) << ", "
                  << ((std::uint32_t *)(header + 1))[2] << ", "
                  << ((std::uint32_t *)(header + 1))[3] << '\n';
      } break;
      default: {
        std::cout << "unknown type " << header->type << std::endl;
      } break;
      }
      info->data_tail += header->size;
    }
  }
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
  close(fd);
  return 0;
}

inline int do_perf2(int argc, char *argv[]) {
  pid_t pid = 0;
  int cpu = -1;
  std::uint64_t tm = 1000000000;
  for (int i = 1; i < argc;) {
    if (strcmp(argv[i], "--pid") == 0 || strcmp(argv[i], "-p") == 0) {
      pid = std::stoul(argv[i + 1]);
      i += 2;
    } else if (strcmp(argv[i], "--cpu") == 0) {
      cpu = std::stoul(argv[i + 1]);
      i += 2;
    } else if (strcmp(argv[i], "-d") == 0) {
      tm = std::stoul(argv[i + 1]);
      i += 2;
    } else {
      i++;
    }
  }

  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_TRACEPOINT;
  pe.size = sizeof(pe);
  pe.config = 379; //   sys_enter_clock_nanosleep

  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  pe.exclude_guest = 1;

  pe.sample_period = 1;
  pe.wakeup_events = 1;
  pe.sample_type = PERF_SAMPLE_IDENTIFIER | PERF_SAMPLE_TIME | PERF_SAMPLE_ID;

  int fd1 = perf_event_open(&pe, pid, cpu, -1, 0);
  if (fd1 == -1) {
    perror("");
    return 1;
  }
  void *buf1 =
      mmap(0, 9 * getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd1, 0);
  if (buf1 == MAP_FAILED) {
    perror("");
    return 1;
  }

  pe.config = 378; //   sys_exit_clock_nanosleep

  int fd2 = perf_event_open(&pe, pid, cpu, -1, 0);
  if (fd2 == -1) {
    perror("");
    return 1;
  }
  void *buf2 =
      mmap(0, 9 * getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd2, 0);
  if (buf2 == MAP_FAILED) {
    perror("");
    return 1;
  }

  int epollfd = epoll_create1(0);
  struct epoll_event ev, events[2];
  ev.events = EPOLLIN;
  ev.data.ptr = buf1;
  if (epoll_ctl(epollfd, EPOLL_CTL_ADD, fd1, &ev) == -1) {
    perror("");
    return 1;
  }
  ev.data.ptr = buf2;
  if (epoll_ctl(epollfd, EPOLL_CTL_ADD, fd2, &ev) == -1) {
    perror("");
    return 1;
  }

  ioctl(fd1, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd2, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd1, PERF_EVENT_IOC_ENABLE, 0);
  ioctl(fd2, PERF_EVENT_IOC_ENABLE, 0);
  while (running) {
    int nfds = epoll_wait(epollfd, events, 2, -1);
    if (nfds == -1 && errno != EINTR) {
      perror("");
      return 1;
    }
    for (int i = 0; i < nfds; i++) {
      void *buf = events[i].data.ptr;
      struct perf_event_mmap_page *info = (struct perf_event_mmap_page *)buf;
      while (info->data_tail < info->data_head) {
        struct perf_event_header *header =
            (struct perf_event_header *)((char *)buf + info->data_offset +
                                         (info->data_tail % info->data_size));
        switch (header->type) {
        case PERF_RECORD_SAMPLE: {
          std::cout << ((std::uint64_t *)(header + 1))[0] << '\t'
                    << ((std::uint64_t *)(header + 1))[1] << '\n';
        } break;
        default: {
          std::cout << "unknown type " << header->type << std::endl;
        } break;
        }
        info->data_tail += header->size;
      }
    }
  }
  ioctl(fd1, PERF_EVENT_IOC_DISABLE, 0);
  ioctl(fd2, PERF_EVENT_IOC_DISABLE, 0);
  if (epoll_ctl(epollfd, EPOLL_CTL_DEL, fd1, nullptr) == -1) {
    perror("");
    return 1;
  }
  if (epoll_ctl(epollfd, EPOLL_CTL_DEL, fd2, nullptr) == -1) {
    perror("");
    return 1;
  }
  close(epollfd);
  close(fd2);
  close(fd1);
  return 0;
}
