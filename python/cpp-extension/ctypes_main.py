#!/usr/bin/env python3
from ctypes import cdll


def main():
    lib = cdll.LoadLibrary('./libadd.so')
    print(lib.add(1, 2))


if __name__ == "__main__":
    main()
