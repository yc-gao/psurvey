#pragma once

#include <functional>
#include <memory>
#include <system_error>

#include "ErrorOr.h"
#include "Traits.h"

namespace detail {

template <typename T> class PromiseImpl;

template <>
class PromiseImpl<void>
    : public std::enable_shared_from_this<PromiseImpl<void>> {
  enum Status {
    NONE = 0,
    RESOLVED,
    REJECTED,
  };
  Status status_;
  ErrorOr<void> holder_;

  std::function<void(const ErrorOr<void> &)> cb_{[](const ErrorOr<void> &) {}};

  void Populate() { cb_(holder_); }

public:
  PromiseImpl() = default;
  PromiseImpl(const PromiseImpl &) = default;
  PromiseImpl(PromiseImpl &&) = default;
  PromiseImpl &operator=(const PromiseImpl &) = default;
  PromiseImpl &operator=(PromiseImpl &&) = default;

  void Resolve() {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = ErrorOr<void>();
    status_ = RESOLVED;
    Populate();
  }
  void Reject(std::error_code ec) {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = ErrorOr<void>(std::move(ec));
    status_ = REJECTED;
    Populate();
  }

  bool Resolved() const { return status_ == RESOLVED; }
  bool Rejected() const { return status_ == REJECTED; }

  std::error_code Error() const { return holder_.Error(); }

  template <typename F, typename R = std::invoke_result_t<F>>
  auto Then(F &&cb) -> std::shared_ptr<PromiseImpl<R>> {
    return Then(std::forward<F>(cb), Type2Type<R>());
  }

  template <typename F>
  auto Then(F &&resolver, Type2Type<void>)
      -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const ErrorOr<void> &holder) mutable {
      cb(holder);
      if (holder) {
        std::forward<F>(resolver)();
        result->Resolve();
      } else {
        result->Reject(holder.Error());
      }
    };
    return result;
  }
  template <typename F, typename R>
  auto Then(F &&resolver, Type2Type<R>) -> std::shared_ptr<PromiseImpl<R>> {
    auto result = std::make_shared<PromiseImpl<R>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const ErrorOr<void> &holder) mutable {
      cb(holder);
      if (holder) {
        result->Resolve(std::forward<F>(resolver)());
      } else {
        result->Reject(holder.Error());
      }
    };
    return result;
  }
  template <typename F>
  auto Catch(F &&rejecter) -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), rejecter = std::forward<F>(rejecter),
           result](const ErrorOr<void> &holder) mutable {
      cb(holder);
      if (holder) {
        result->Resolve();
      } else {
        std::forward<F>(rejecter)(holder.Error());
        result->Reject(holder.Error());
      }
    };
    return result;
  }
  template <typename F>
  auto Finally(F &&func) -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), func = std::forward<F>(func),
           result](const ErrorOr<void> &holder) mutable {
      cb(holder);
      std::forward<F>(func)();
      if (holder) {
        result->Resolve();
      } else {
        result->Reject(holder.Error());
      }
    };
    return result;
  }
};

template <typename T>
class PromiseImpl : public std::enable_shared_from_this<PromiseImpl<T>> {
  enum Status {
    NONE = 0,
    RESOLVED,
    REJECTED,
  };
  Status status_;
  ErrorOr<T> holder_;

  std::function<void(const ErrorOr<T> &)> cb_{[](const ErrorOr<T> &) {}};

  void Populate() { cb_(holder_); }

public:
  PromiseImpl() = default;
  PromiseImpl(const PromiseImpl &) = default;
  PromiseImpl(PromiseImpl &&) = default;
  PromiseImpl &operator=(const PromiseImpl &) = default;
  PromiseImpl &operator=(PromiseImpl &&) = default;

  template <typename U> void Resolve(U &&val) {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = ErrorOr<T>(std::forward<U>(val));
    status_ = RESOLVED;
    Populate();
  }
  void Reject(std::error_code ec) {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = ErrorOr<T>(std::move(ec));
    status_ = REJECTED;
    Populate();
  }

  bool Resolved() const { return status_ == RESOLVED; }
  bool Rejected() const { return status_ == REJECTED; }

  std::error_code Error() const { return holder_.Error(); }
  T &Value() { return holder_.Value(); }
  const T &Value() const { return holder_.Value(); }

  template <typename F, typename R = std::invoke_result_t<F, T>>
  auto Then(F &&cb) -> std::shared_ptr<PromiseImpl<R>> {
    return Then(std::forward<F>(cb), Type2Type<R>());
  }

  template <typename F>
  auto Then(F &&resolver, Type2Type<void>)
      -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const ErrorOr<T> &holder) mutable {
      cb(holder);
      if (holder) {
        std::forward<F>(resolver)(holder.Value());
        result->Resolve();
      } else {
        result->Reject(holder.Error());
      }
    };
    return result;
  }
  template <typename F, typename R>
  auto Then(F &&resolver, Type2Type<R>) -> std::shared_ptr<PromiseImpl<R>> {
    auto result = std::make_shared<PromiseImpl<R>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const ErrorOr<T> &holder) mutable {
      cb(holder);
      if (holder) {
        result->Resolve(std::forward<F>(resolver)(holder.Value()));
      } else {
        result->Reject(holder.Error());
      }
    };
    return result;
  }
  template <typename F>
  auto Catch(F &&rejecter) -> std::shared_ptr<PromiseImpl<T>> {
    auto result = std::make_shared<PromiseImpl<T>>();
    cb_ = [cb = std::move(cb_), rejecter = std::forward<F>(rejecter),
           result](const ErrorOr<T> &holder) mutable {
      cb(holder);
      if (holder) {
        result->Resolve(holder.Value());
      } else {
        std::forward<F>(rejecter)(holder.Error());
        result->Reject(holder.Error());
      }
    };
    return result;
  }
  template <typename F>
  auto Finally(F &&func) -> std::shared_ptr<PromiseImpl<T>> {
    auto result = std::make_shared<PromiseImpl<T>>();
    cb_ = [cb = std::move(cb_), func = std::forward<F>(func),
           result](const ErrorOr<T> &holder) mutable {
      cb(holder);
      std::forward<F>(func)();
      if (holder) {
        result->Resolve(holder.Value());
      } else {
        result->Reject(holder.Error());
      }
    };
    return result;
  }
};

}; // namespace detail

template <typename T = void> class Promise;

template <> class Promise<void> {
  std::shared_ptr<detail::PromiseImpl<void>> impl_;

public:
  Promise(const Promise &) = default;
  Promise(Promise &&) = default;
  Promise &operator=(const Promise &) = default;
  Promise &operator=(Promise &&) = default;

  Promise() : Promise(std::make_shared<detail::PromiseImpl<void>>()) {}
  Promise(std::shared_ptr<detail::PromiseImpl<void>> impl)
      : impl_(std::move(impl)) {}

  void Resolve() { impl_->Resolve(); }
  void Reject(std::error_code ec) { impl_->Reject(std::move(ec)); }

  bool Resolved() const { return impl_->Resolved(); }
  bool Rejected() const { return impl_->Rejected(); }

  std::error_code Error() const { return impl_->Error(); }

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

  Promise<void> Then(Promise<void> p) {
    Promise<void> ret;
    Then([ret, p = p.impl_->weak_from_this()]() mutable {
      if (p.lock()->Resolved()) {
        ret.Resolve();
      }
    });
    p.Then([ret, p = impl_->weak_from_this()]() mutable {
      if (p.lock()->Resolved()) {
        ret.Resolve();
      }
    });

    return ret;
  }
};

template <typename T> class Promise {
  std::shared_ptr<detail::PromiseImpl<T>> impl_;

public:
  Promise(const Promise &) = default;
  Promise(Promise &&) = default;
  Promise &operator=(const Promise &) = default;
  Promise &operator=(Promise &&) = default;

  Promise() : Promise(std::make_shared<detail::PromiseImpl<T>>()) {}
  Promise(std::shared_ptr<detail::PromiseImpl<T>> impl)
      : impl_(std::move(impl)) {}

  template <typename U> void Resolve(U &&val) {
    impl_->Resolve(std::forward<U>(val));
  }
  void Reject(std::error_code ec) { impl_->Reject(std::move(ec)); }

  bool Resolved() const { return impl_->Resolved(); }
  bool Rejected() const { return impl_->Rejected(); }

  std::error_code Error() const { return impl_->Error(); }
  T &Value() { return impl_->Value(); }
  const T &Value() const { return impl_->Value(); }

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
