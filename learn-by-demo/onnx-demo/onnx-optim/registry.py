_name2optimizer = {}


def optimizer(name):
    def wrapper(cls):
        if name in _name2optimizer:
            raise RuntimeError(f'Optimizer "{name}" already registered')
        _name2optimizer[name] = cls
        return cls
    return wrapper


def find_optimizer(name):
    return _name2optimizer.get(name, None)
