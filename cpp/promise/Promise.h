#pragma once

#include <functional>
#include <memory>
#include <system_error>
#include <type_traits>

namespace detail {
template <typename T> class Type2Type {};

template <typename = void> class PromiseImpl;

template <>
class PromiseImpl<void>
    : public std::enable_shared_from_this<PromiseImpl<void>> {
  enum Status { NONE = 0, RESOLVED, REJECTED };
  Status status_{NONE};

  std::function<void()> resolve_{[]() {}};
  std::function<void(const std::error_code &)> reject_{
      [](const std::error_code &) {}};
  std::function<void()> finally_{[]() {}};

public:
  void Resolve() {
    if (status_) {
      return;
    }
    status_ = RESOLVED;
    resolve_();
    finally_();
  }
  void Reject(const std::error_code &ec) {
    if (status_) {
      return;
    }
    status_ = REJECTED;
    reject_(ec);
    finally_();
  }

  template <typename F, typename R = std::invoke_result_t<F>>
  auto Then(F &&f) -> std::shared_ptr<PromiseImpl<R>> {
    return Then(std::forward<F>(f), Type2Type<R>());
  }
  template <typename F>
  auto Then(F &&f, Type2Type<void>) -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    resolve_ = [result, f = std::forward<F>(f)]() mutable {
      std::forward<F>(f)();
      result->Resolve();
    };
    reject_ = [result, f = std::move(reject_)](const std::error_code &ec) {
      f(ec);
      result->Reject(ec);
    };
    return result;
  }
  template <typename F, typename R>
  auto Then(F &&f, Type2Type<R>) -> std::shared_ptr<PromiseImpl<R>> {
    auto result = std::make_shared<PromiseImpl<R>>();
    resolve_ = [result, f = std::forward<F>(f)]() mutable {
      result->Resolve(std::forward<F>(f)());
    };
    reject_ = [result, f = std::move(reject_)](const std::error_code &ec) {
      f(ec);
      result->Reject(ec);
    };
    return result;
  }
  template <typename F>
  auto Catch(F &&f) -> std::shared_ptr<PromiseImpl<void>> {
    reject_ = [f = std::forward<F>(f),
               reject = std::move(reject_)](const std::error_code &ec) {
      reject(ec);
      f(ec);
    };
    return this->shared_from_this();
  }
  template <typename F>
  auto Finally(F &&f) -> std::shared_ptr<PromiseImpl<void>> {
    finally_ = [f = std::forward<F>(f), finally = std::move(finally_)]() {
      finally();
      f();
    };
    return this->shared_from_this();
  }

  bool Resolved() const { return status_ == RESOLVED; }
  bool Rejected() const { return status_ == REJECTED; }
};

template <typename T>
class PromiseImpl : public std::enable_shared_from_this<PromiseImpl<T>> {
  enum Status { NONE = 0, RESOLVED, REJECTED };
  Status status_{NONE};

  std::function<void(T)> resolve_{[](T) {}};
  std::function<void(const std::error_code &)> reject_{
      [](const std::error_code &) {}};
  std::function<void()> finally_{[]() {}};

public:
  void Resolve(T val) {
    if (status_) {
      return;
    }
    status_ = RESOLVED;
    resolve_(std::move(val));
    finally_();
  }
  void Reject(const std::error_code &ec) {
    if (status_) {
      return;
    }
    status_ = REJECTED;
    reject_(ec);
    finally_();
  }

  template <typename F, typename R = std::invoke_result_t<F, T>>
  auto Then(F &&f) -> std::shared_ptr<PromiseImpl<R>> {
    return Then(std::forward<F>(f), Type2Type<R>());
  }
  template <typename F>
  auto Then(F &&f, Type2Type<void>) -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    resolve_ = [result, f = std::forward<F>(f)](T val) mutable {
      std::forward<F>(f)(std::move(val));
      result->Resolve();
    };
    reject_ = [result, f = std::move(reject_)](const std::error_code &ec) {
      f(ec);
      result->Reject(ec);
    };
    return result;
  }
  template <typename F, typename R>
  auto Then(F &&f, Type2Type<R>) -> std::shared_ptr<PromiseImpl<R>> {
    auto result = std::make_shared<PromiseImpl<R>>();
    resolve_ = [result, f = std::forward<F>(f)](T val) mutable {
      result->Resolve(std::forward<F>(f)(std::move(val)));
    };
    reject_ = [result, f = std::move(reject_)](const std::error_code &ec) {
      f(ec);
      result->Reject(ec);
    };
    return result;
  }

  template <typename F> auto Catch(F &&f) -> std::shared_ptr<PromiseImpl<T>> {
    reject_ = [f = std::forward<F>(f),
               reject = std::move(reject_)](const std::error_code &ec) {
      reject(ec);
      f(ec);
    };
    return this->shared_from_this();
  }
  template <typename F> auto Finally(F &&f) -> std::shared_ptr<PromiseImpl<T>> {
    finally_ = [f = std::forward<F>(f), finally = std::move(finally_)]() {
      finally();
      f();
    };
    return this->shared_from_this();
  }

  bool Resolved() const { return status_ == RESOLVED; }
  bool Rejected() const { return status_ == REJECTED; }
};

} // namespace detail

template <typename = void> class Promise;

template <> class Promise<void> {
  std::shared_ptr<detail::PromiseImpl<void>> impl_;

public:
  Promise() : Promise(std::make_shared<detail::PromiseImpl<void>>()) {}
  Promise(const Promise &) = default;
  Promise(Promise &&) = default;
  Promise &operator=(const Promise &) = default;
  Promise &operator=(Promise &&) = default;

  Promise(std::shared_ptr<detail::PromiseImpl<void>> impl)
      : impl_(std::move(impl)) {}

  void Resolve() { impl_->Resolve(); }
  void Reject(const std::error_code &ec) { impl_->Reject(ec); }

  template <typename F, typename R = std::invoke_result_t<F>>
  auto Then(F &&f) -> Promise<R> {
    return {impl_->Then(std::forward<F>(f))};
  }
  template <typename F> auto Catch(F &&f) -> Promise<void> {
    return {impl_->Catch(std::forward<F>(f))};
  }
  template <typename F> auto Finally(F &&f) -> Promise<void> {
    return {impl_->Finally(std::forward<F>(f))};
  }

  bool Resolved() const { return impl_->Resolved(); }
  bool Rejected() const { return impl_->Rejected(); }

  friend Promise operator+(Promise a, Promise b) {
    Promise r;
    a.Then([b = b.impl_->weak_from_this(),
            r = r.impl_->weak_from_this()]() mutable {
      if (b.lock()->Resolved()) {
        r.lock()->Resolve();
      }
    });
    b.Then([a = a.impl_->weak_from_this(),
            r = r.impl_->weak_from_this()]() mutable {
      if (a.lock()->Resolved()) {
        r.lock()->Resolve();
      }
    });
    r.Finally([a, b]() {});
    if (a.Resolved() && b.Resolved()) {
      r.Resolve();
    }
    return r;
  }
};

template <typename T> class Promise {
  std::shared_ptr<detail::PromiseImpl<T>> impl_;

public:
  Promise() : Promise(std::make_shared<detail::PromiseImpl<T>>()) {}
  Promise(const Promise &) = default;
  Promise(Promise &&) = default;
  Promise &operator=(const Promise &) = default;
  Promise &operator=(Promise &&) = default;

  Promise(std::shared_ptr<detail::PromiseImpl<T>> impl)
      : impl_(std::move(impl)) {}

  void Resolve(T val) { impl_->Resolve(std::move(val)); }
  void Reject(const std::error_code &ec) { impl_->Reject(ec); }

  template <typename F, typename R = std::invoke_result_t<F, T>>
  auto Then(F &&f) -> Promise<R> {
    return {impl_->Then(std::forward<F>(f))};
  }
  template <typename F> auto Catch(F &&f) -> Promise<T> {
    return {impl_->Catch(std::forward<F>(f))};
  }
  template <typename F> auto Finally(F &&f) -> Promise<T> {
    return {impl_->Finally(std::forward<F>(f))};
  }
};
