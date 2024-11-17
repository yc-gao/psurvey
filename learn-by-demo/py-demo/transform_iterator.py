#!/usr/bin/env python3
import functools


class transform_iterator:
    def __init__(self, iter, *transforms):
        self.iter = iter
        self.transforms = transforms

    def __len__(self):
        return len(self.iter)

    def __iter__(self):
        class inner:
            def __init__(self, iter, transforms):
                self.iter = iter
                self.transforms = transforms

            def __next__(self):
                return functools.reduce(lambda item, f: f(item), (next(self.iter), ) + self.transforms)
        return inner(iter(self.iter), self.transforms)


def main():
    for n in transform_iterator([1, 2, 3], lambda i: i + 1, lambda i: i * 2):
        print(n)


if __name__ == "__main__":
    main()
