__name2optimizer = {}


def optimizer(name):
    def wrapper(cls):
        if name in __name2optimizer:
            raise RuntimeError(f"optimizer '{name}' already registered")
        __name2optimizer[name] = cls
        return cls
    return wrapper


def find_optimizer(name):
    return __name2optimizer.get(name, None)
