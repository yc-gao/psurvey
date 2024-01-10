#pragma once
#include <fcntl.h>
#include <set>
#include <stdexcept>
#include <sys/epoll.h>
#include <unistd.h>

#include <cstdint>
#include <system_error>

enum class Event { NONE, ABORT, READ, WRITE };

struct Callback {
  virtual void Dispatch(Event) = 0;
};

class EventLoop {
  constexpr static int kMaxEvent = 100;

  volatile bool running_{true};

  int epoll_fd_;
  struct epoll_event event_pool_[kMaxEvent];
  std::set<Callback *> cbs_;

public:
  ~EventLoop() {
    if (epoll_fd_ != -1) {
      close(epoll_fd_);
    }
  }
  EventLoop(const EventLoop &) = delete;
  EventLoop(EventLoop &&) = delete;
  EventLoop &operator=(const EventLoop &) = delete;
  EventLoop &operator=(EventLoop &&) = delete;

  EventLoop() : epoll_fd_(::epoll_create1(0)) {
    if (epoll_fd_ == -1) {
      throw std::system_error(errno, std::generic_category(),
                              "epoll_create1 failed");
    }
  }

  void Stop() { running_ = false; }

  void Run() {
    while (running_) {
      if (cbs_.empty()) {
        break;
      }

      int nfd = ::epoll_wait(epoll_fd_, event_pool_, kMaxEvent, -1);
      if (nfd < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "epoll_wait failed");
      }
      for (int i = 0; i < nfd; i++) {
        Event e;
        if (event_pool_[i].events | EPOLLIN) {
          e = Event::READ;
        } else if (event_pool_[i].events | EPOLLOUT) {
          e = Event::WRITE;
        } else {
          throw std::system_error(errno, std::generic_category(),
                                  "unknown event");
        }
        reinterpret_cast<Callback *>(event_pool_[i].data.ptr)->Dispatch(e);
      }
    }
    for (auto cb : cbs_) {
      cb->Dispatch(Event::ABORT);
    }
  }

  void Attach(int fd, Event e, Callback *cb) {
    struct epoll_event event;
    switch (e) {
    case Event::READ:
      event.events = EPOLLIN;
      break;
    case Event::WRITE:
      event.events = EPOLLOUT;
      break;
    default:
      throw std::logic_error("unsupport event");
      break;
    }
    event.data.ptr = cb;
    if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &event)) {
      throw std::system_error(errno, std::generic_category(),
                              "epoll_ctl failed");
    }
    if (!cbs_.insert(cb).second) {
      throw std::logic_error("fd already inserted");
    }
  }
  void Detach(int fd, Callback *cb) {
    if (!cbs_.erase(cb)) {
      throw std::logic_error("fd not inserted");
    }
    if (::epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, NULL)) {
      throw std::system_error(errno, std::generic_category(),
                              "epoll_ctl failed");
    }
  }
};
