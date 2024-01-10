#pragma once

#include <functional>
#include <memory>
#include <system_error>

#include "Result.h"
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
  Result<void> holder_;

  std::function<void(const Result<void> &)> cb_{[](const Result<void> &) {}};

  void Populate() { cb_(holder_); }

public:
  PromiseImpl() = default;
  PromiseImpl(const PromiseImpl &) = default;
  PromiseImpl(PromiseImpl &&) = default;
  PromiseImpl &operator=(const PromiseImpl &) = default;
  PromiseImpl &operator=(PromiseImpl &&) = default;

  bool Resolved() const { return status_ == RESOLVED; }
  bool Rejected() const { return status_ == REJECTED; }

  void Resolve() {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = Result<void>();
    status_ = RESOLVED;
    Populate();
  }
  void Reject(std::error_code ec) {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = Result<void>(std::move(ec));
    status_ = REJECTED;
    Populate();
  }

  std::error_code Error() const { return holder_.Error(); }

  template <typename F, typename R = std::invoke_result_t<F>>
  auto Then(F &&cb) -> std::shared_ptr<PromiseImpl<R>> {
    return Then(Type2Type<R>(), std::forward<F>(cb));
  }

  template <typename F>
  auto Then(Type2Type<void>, F &&resolver)
      -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const Result<void> &holder) mutable {
      cb(holder);
      if (holder.Error()) {
        result->Reject(holder.Error());
      } else {
        std::forward<F>(resolver)();
        result->Resolve();
      }
    };
    return result;
  }
  template <typename F, typename R>
  auto Then(Type2Type<R>, F &&resolver) -> std::shared_ptr<PromiseImpl<R>> {
    auto result = std::make_shared<PromiseImpl<R>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const Result<void> &holder) mutable {
      cb(holder);
      if (holder.Error()) {
        result->Reject(holder.Error());
      } else {
        result->Resolve(std::forward<F>(resolver)());
      }
    };
    return result;
  }
  template <typename F>
  auto Catch(F &&rejecter) -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), rejecter = std::forward<F>(rejecter),
           result](const Result<void> &holder) mutable {
      cb(holder);
      if (holder.Error()) {
        std::forward<F>(rejecter)(holder.Error());
        result->Reject(holder.Error());
      } else {
        result->Resolve();
      }
    };
    return result;
  }
  template <typename F>
  auto Finally(F &&func) -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), func = std::forward<F>(func),
           result](const Result<void> &holder) mutable {
      cb(holder);
      std::forward<F>(func)();
      if (holder.Error()) {
        result->Reject(holder.Error());
      } else {
        result->Resolve();
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
  Result<T> holder_;

  std::function<void(const Result<T> &)> cb_{[](const Result<T> &) {}};

  void Populate() { cb_(holder_); }

public:
  PromiseImpl() = default;
  PromiseImpl(const PromiseImpl &) = default;
  PromiseImpl(PromiseImpl &&) = default;
  PromiseImpl &operator=(const PromiseImpl &) = default;
  PromiseImpl &operator=(PromiseImpl &&) = default;

  bool Resolved() const { return status_ == RESOLVED; }
  bool Rejected() const { return status_ == REJECTED; }

  template <typename U> void Resolve(U &&val) {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = Result<T>(std::forward<U>(val));
    status_ = RESOLVED;
    Populate();
  }
  void Reject(std::error_code ec) {
    if (status_) {
      throw std::logic_error("promise resolved or rejected");
    }
    holder_ = Result<T>(std::move(ec));
    status_ = REJECTED;
    Populate();
  }

  std::error_code Error() const { return holder_.Error(); }
  T &Value() { return holder_.Value(); }
  const T &Value() const { return holder_.Value(); }

  template <typename F, typename R = std::invoke_result_t<F, T>>
  auto Then(F &&cb) -> std::shared_ptr<PromiseImpl<R>> {
    return Then(Type2Type<R>(), std::forward<F>(cb));
  }

  template <typename F>
  auto Then(Type2Type<void>, F &&resolver)
      -> std::shared_ptr<PromiseImpl<void>> {
    auto result = std::make_shared<PromiseImpl<void>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const Result<T> &holder) mutable {
      cb(holder);
      if (holder.Error()) {
        result->Reject(holder.Error());
      } else {
        std::forward<F>(resolver)(holder.Value());
        result->Resolve();
      }
    };
    return result;
  }
  template <typename F, typename R>
  auto Then(Type2Type<R>, F &&resolver) -> std::shared_ptr<PromiseImpl<R>> {
    auto result = std::make_shared<PromiseImpl<R>>();
    cb_ = [cb = std::move(cb_), resolver = std::forward<F>(resolver),
           result](const Result<T> &holder) mutable {
      cb(holder);
      if (holder.Error()) {
        result->Reject(holder.Error());
      } else {
        result->Resolve(std::forward<F>(resolver)(holder.Value()));
      }
    };
    return result;
  }
  template <typename F>
  auto Catch(F &&rejecter) -> std::shared_ptr<PromiseImpl<T>> {
    auto result = std::make_shared<PromiseImpl<T>>();
    cb_ = [cb = std::move(cb_), rejecter = std::forward<F>(rejecter),
           result](const Result<T> &holder) mutable {
      cb(holder);
      if (holder.Error()) {
        std::forward<F>(rejecter)(holder.Error());
        result->Reject(holder.Error());
      } else {
        result->Resolve(holder.Value());
      }
    };
    return result;
  }
  template <typename F>
  auto Finally(F &&func) -> std::shared_ptr<PromiseImpl<T>> {
    auto result = std::make_shared<PromiseImpl<T>>();
    cb_ = [cb = std::move(cb_), func = std::forward<F>(func),
           result](const Result<T> &holder) mutable {
      cb(holder);
      std::forward<F>(func)();
      if (holder.Error()) {
        result->Reject(holder.Error());
      } else {
        result->Resolve(holder.Value());
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

  bool Resolved() const { return impl_->Resolved(); }
  bool Rejected() const { return impl_->Rejected(); }

  void Resolve() { impl_->Resolve(); }
  void Reject(std::error_code ec) { impl_->Reject(std::move(ec)); }

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

  Promise<void> And(Promise<void> p) {
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

  bool Resolved() const { return impl_->Resolved(); }
  bool Rejected() const { return impl_->Rejected(); }

  template <typename U> void Resolve(U &&val) {
    impl_->Resolve(std::forward<U>(val));
  }
  void Reject(std::error_code ec) { impl_->Reject(std::move(ec)); }

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
