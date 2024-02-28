#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <linux/perf_event.h>    /* Definition of PERF_* constants */
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <system_error>
#include <thread>
#include <unistd.h>

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return ret;
}

bool running{true};

void do_payload() {
  while (running) {
  }
}

void do_read(pid_t pid = 266267) {
  struct perf_event_attr attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);

  attr.type = PERF_TYPE_HARDWARE;
  attr.config = PERF_COUNT_HW_INSTRUCTIONS;

  attr.disabled = 1;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;

  attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                     PERF_FORMAT_TOTAL_TIME_RUNNING | PERF_FORMAT_ID |
                     PERF_FORMAT_LOST;

  int perf_fd = perf_event_open(&attr, pid, -1, -1, PERF_FLAG_FD_CLOEXEC);
  if (perf_fd == -1) {
    throw std::system_error(errno, std::generic_category());
  }
  ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0);
  while (running) {
    ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
    struct read_format {
      std::uint64_t value;        /* The value of the event */
      std::uint64_t time_enabled; /* if PERF_FORMAT_TOTAL_TIME_ENABLED */
      std::uint64_t time_running; /* if PERF_FORMAT_TOTAL_TIME_RUNNING */
      std::uint64_t id;           /* if PERF_FORMAT_ID */
      std::uint64_t lost;         /* if PERF_FORMAT_LOST */
    } buf;
    auto l = read(perf_fd, &buf, sizeof(buf));
    if (l < 0) {
      throw std::system_error(errno, std::generic_category());
    } else if (l == 0) {
      continue;
    } else {
      std::cout << "time_enabled " << buf.time_enabled << " time_running "
                << buf.time_running << " id " << buf.id << " lost " << buf.lost
                << " value " << buf.value << std::endl;
    }
  }
  close(perf_fd);
}

void do_period(pid_t pid = 291080) {
  struct perf_event_attr attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.size = sizeof(attr);

  attr.type = PERF_TYPE_SOFTWARE;
  attr.config = PERF_COUNT_SW_CPU_CLOCK;

  // attr.freq = 1;
  // attr.sample_freq = 10;
  attr.sample_period = 100000;
  attr.wakeup_events = 1;
  attr.sample_type =
      PERF_SAMPLE_TIME | PERF_SAMPLE_TID | PERF_SAMPLE_READ | PERF_SAMPLE_CPU;

  attr.disabled = 1;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;

  int perf_fd = perf_event_open(&attr, pid, -1, -1, PERF_FLAG_FD_CLOEXEC);
  if (perf_fd == -1) {
    throw std::system_error(errno, std::generic_category());
  }

  int page_size = getpagesize();
  int page_count = 16;
  void *perf_mmap = mmap(0, page_size + page_count * page_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED, perf_fd, 0);
  if (perf_mmap == MAP_FAILED) {
    throw std::system_error(errno, std::generic_category());
  }

  ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0);
  struct pollfd pfd = {.fd = perf_fd, .events = POLLIN};
  struct perf_event_mmap_page *perf_info =
      (struct perf_event_mmap_page *)perf_mmap;
  while (running) {
    if (poll(&pfd, 1, -1) == -1) {
      throw std::system_error(errno, std::generic_category());
    }
    struct perf_format {
      std::uint32_t pid, tid; /* if PERF_SAMPLE_TID */
      std::uint64_t time;     /* if PERF_SAMPLE_TIME */
      // std::uint64_t addr;      /* if PERF_SAMPLE_ADDR */
      // std::uint64_t id;        /* if PERF_SAMPLE_ID */
      // std::uint64_t stream_id; /* if PERF_SAMPLE_STREAM_ID */
      std::uint32_t cpu, res; /* if PERF_SAMPLE_CPU */
      // std::uint64_t period;    /* if PERF_SAMPLE_PERIOD */
      struct {
        std::uint64_t value;
      } v; /* if PERF_SAMPLE_READ */
    };

    while (perf_info->data_tail < perf_info->data_head) {
      struct perf_event_header *header =
          (struct perf_event_header *)((char *)perf_mmap +
                                       perf_info->data_offset +
                                       (perf_info->data_tail % page_size));

      if (header->type == PERF_RECORD_SAMPLE) {
        perf_format *perf_data =
            (perf_format *)((char *)header + sizeof(perf_event_header));
        std::cout << header->size << "\t" << perf_data->time << "\t"
                  << perf_data->v.value << "\n";
        break;
      }
      perf_info->data_tail += header->size;
    }
  }
  ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);

  munmap(perf_mmap, page_count * page_size);
  close(perf_fd);
}

int main(int argc, char *argv[]) {
  std::signal(SIGINT, [](int sig) { running = false; });

  std::thread t(do_payload);

  // do_read();
  do_period();

  return 0;
}
