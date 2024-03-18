#!/usr/bin/env python
import sys
import argparse

import rosbag


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output.bag')
    parser.add_argument('-i', '--input', nargs='+', default=[])
    return parser.parse_args(argv)


def bag_merge(obag, ibag):
    if isinstance(obag, str):
        obag = rosbag.Bag(obag, 'w')
        bag_merge(obag, ibag)
    else:
        with rosbag.Bag(ibag, 'r') as ibag:
            for (topic, msg, timestamp) in ibag:
                obag.write(topic, msg, timestamp)


def main(args):
    obag = args.output
    for ibag in args.input:
        obag = bag_merge(obag, ibag)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
