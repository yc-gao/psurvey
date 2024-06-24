#include <cstdio>
#include <functional>
#include <iostream>

template <typename T>
class HandleHolder {
  T handle_;
  std::function<void(T)> deallocator_;

 public:
  friend void swap(HandleHolder& a, HandleHolder& b) {
    using std::swap;
    swap(a.handle_, b.handle_);
    swap(a.deallocator_, b.deallocator_);
  }

  ~HandleHolder() {
    if (deallocator_) {
      deallocator_(handle_);
    }
  }

  HandleHolder(HandleHolder const& other)
      : handle_(other.handle_), deallocator_(std::move(other.deallocator_)) {}
  HandleHolder(HandleHolder&&) = delete;

  HandleHolder operator=(HandleHolder other) { swap(*this, other); }

  HandleHolder() = default;
  template <typename F>
  HandleHolder(T handle, F&& deallocator)
      : handle_(handle), deallocator_(std::forward<F>(deallocator)) {}

  T Handle() const { return handle_; }
  T Release() {
    deallocator_ = nullptr;
    return handle_;
  }
};

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("testing HandleHolder") {
  HandleHolder<FILE*> fhandler(fopen("demo.log", "a+"), [](FILE* file) {
    if (file) {
      std::cout << "close file" << std::endl;
      fclose(file);
    } else {
      std::cerr << "empty file" << std::endl;
    }
  });
}
