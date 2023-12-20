#pragma once

#include <memory>
#include <system_error>

template <typename T> class ErrorOr;

template <typename T> class ErrorOr {
  std::error_code ec_;
  std::unique_ptr<T> val_;

public:
  ErrorOr(const ErrorOr &) = default;
  ErrorOr(ErrorOr &&) = default;
  ErrorOr &operator=(const ErrorOr &) = default;
  ErrorOr &operator=(ErrorOr &&) = default;

  ErrorOr() : ErrorOr(std::error_code()) {}
  ErrorOr(std::error_code ec) : ec_(std::move(ec)) {}
  template <typename U>
  ErrorOr(U &&val) : val_(std::make_unique<T>(std::forward<U>(val))) {}

  operator bool() const { return bool(ec_); }
  std::error_code Error() const { return ec_; }

  T &Value() { return *val_; }
  const T &Value() const { return *val_; }

  T &operator*() { return *val_; }
  const T &operator*() const { return *val_; }
  T *operator->() { return val_.get(); }
  const T *operator->() const { return val_.get(); }
};

template <> class ErrorOr<void> {
  std::error_code ec_;

public:
  ErrorOr(const ErrorOr &) = default;
  ErrorOr(ErrorOr &&) = default;
  ErrorOr &operator=(const ErrorOr &) = default;
  ErrorOr &operator=(ErrorOr &&) = default;

  ErrorOr() : ErrorOr(std::error_code()) {}
  ErrorOr(std::error_code ec) : ec_(std::move(ec)) {}

  operator bool() const { return bool(ec_); }
  std::error_code Error() const { return ec_; }
};
