#pragma once

#include <fcntl.h>
#include <sys/socket.h>

#include "EventLoop.h"

class Socket {
  EventLoop *loop_;
  int fd_;

public:
  Socket(const Socket &) = delete;
  Socket(Socket &&) = delete;
  Socket &operator=(const Socket &) = delete;
  Socket &operator=(Socket &&) = delete;

  Socket(EventLoop *loop) : loop_(loop) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
      throw std::system_error(errno, std::generic_category());
    }
    int flags = ::fcntl(fd_, F_GETFL);
    flags |= O_NONBLOCK;
    ::fcntl(fd_, F_SETFL, flags);
  }

  // TODO: enhance
  template <typename F> void async_connect() {}
};
