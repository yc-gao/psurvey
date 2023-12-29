#pragma once
#include <fcntl.h>
#include <sys/epoll.h>
#include <unistd.h>

#include <cstdint>
#include <system_error>

struct Callback {
  virtual void operator()(const std::error_code &ec) = 0;
};

class EventLoop {
  constexpr static int kMaxEvent = 100;

  int epoll_fd;
  struct epoll_event events[kMaxEvent];

  bool running_{true};

  void Dispatch(struct epoll_event &event) {
    Callback *func = (Callback *)(event.data.ptr);
    (*func)(std::error_code());
  }

public:
  EventLoop() : epoll_fd(::epoll_create1(0)) {}

  void Run() {
    while (running_) {
      int nfd = ::epoll_wait(epoll_fd, events, kMaxEvent, -1);
      for (int i = 0; i < nfd; i++) {
        Dispatch(events[i]);
      }
      // TODO: exit when empty
    }
    // TODO: call cb for every event remain
  }
  void Attach(int fd, std::uint32_t events, Callback *cb) {
    struct epoll_event event;
    event.events = events;
    event.data.ptr = (void *)cb;
    if (::epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event) == -1) {
      (*cb)(std::error_code(errno, std::generic_category()));
    }
  }
  void Detach(int fd) { ::epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL); }

  void Stop() { running_ = false; }
};

class FileReader {
  class CallbackImpl : public Callback {
    void operator()(const std::error_code &ec) override {}
  };

  EventLoop *loop;
  int fd;
  CallbackImpl cb;

public:
  ~FileReader() {
    loop->Detach(fd);
    close(fd);
  }
  FileReader(EventLoop *loop, const char *fname)
      : loop(loop), fd(::open(fname, O_RDONLY | O_NONBLOCK)) {
    loop->Attach(fd, EPOLLIN, &cb);
  }
};
