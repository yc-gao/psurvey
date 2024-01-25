#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <functional>
#include <memory>

class Defer {
  std::function<void()> cb_;

public:
  Defer(const Defer &) = default;
  Defer(Defer &&) = default;
  Defer &operator=(const Defer &) = default;
  Defer &operator=(Defer &&) = default;

  ~Defer() {
    if (cb_) {
      cb_();
    }
  }
  template <typename F> Defer(F &&cb) : cb_(std::forward<F>(cb)) {}
};
#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_DEFER(func) Defer CONCATENATE(defer__, __LINE__)([&]() { func; })

struct ShmArea {
  static std::shared_ptr<void> Create(const char *name, std::size_t size,
                                      int oflags, mode_t mode, int mprot) {
    int fd = -1;
    void *addr = MAP_FAILED;

    fd = shm_open(name, oflags, mode);
    if (fd < 0) {
      goto err_out;
    }
    if (ftruncate(fd, size)) {
      goto truncate_err;
    }

    addr = mmap(0, size, mprot, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
      goto map_err;
    }

    return std::shared_ptr<void>(
        addr, [path = std::string(name), size, oflags](void *addr) {
          munmap(addr, size);
          if (oflags | O_CREAT) {
            shm_unlink(path.c_str());
          }
        });
  map_err:
  truncate_err:
    if (oflags | O_CREAT) {
      shm_unlink(name);
    }
  err_out:
    return nullptr;
  }

  static std::shared_ptr<void> Create(const char *name, std::size_t size) {
    return Create(name, size, O_RDWR | O_CREAT, 0777, PROT_READ | PROT_WRITE);
  }

  static std::shared_ptr<void> Attach(const char *name, std::size_t size,
                                      bool readonly = false) {
    return Create(name, size, readonly ? O_RDONLY : O_RDWR, 0777,
                  readonly ? PROT_READ : (PROT_READ | PROT_WRITE));
  }
};
