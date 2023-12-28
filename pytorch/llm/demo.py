#!/usr/bin/env python3

import utils


def main():
    pe = utils.positional_encoding(50, 100)
    print(pe.shape)


if __name__ == '__main__':
    main()
