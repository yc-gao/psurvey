#pragma once

#include <memory>
#include <system_error>

template <typename T> class Result;

template <> class Result<void> {
  std::error_code ec_;

public:
  virtual ~Result() = default;
  Result(const Result &) = default;
  Result(Result &&) = default;
  Result &operator=(const Result &) = default;
  Result &operator=(Result &&) = default;

  Result() : Result(std::error_code()) {}
  Result(std::error_code ec) : ec_(std::move(ec)) {}

  const std::error_code &Error() const { return ec_; }
};

template <typename T> class Result : public Result<void> {
  std::unique_ptr<T> val_;

public:
  using Result<void>::Result;

  virtual ~Result() = default;
  Result(const Result &) = default;
  Result(Result &&) = default;
  Result &operator=(const Result &) = default;
  Result &operator=(Result &&) = default;

  Result() : Result<void>() {}
  Result(std::unique_ptr<T> val)
      : Result<void>(std::error_code()), val_(std::move(val)) {}
  template <typename U>
  Result(U &&val) : Result(std::make_unique<T>(std::forward<U>(val))) {}

  bool Empty() const { return !val_; }
  operator bool() const { return !Empty(); }

  const T &Value() const { return *val_; }
  T &Value() { return *val_; }

  operator const T &() const { return Value(); }
  operator T &() { return Value(); }
};
