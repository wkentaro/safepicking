#!/usr/bin/env python

import argparse
import json

import matplotlib.pyplot as plt
import pandas
import path


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json_file", type=path.Path, help="json file")
    args = parser.parse_args()

    with open(args.json_file) as f:
        df = pandas.DataFrame(json.load(f))

    df.plot.bar()
    plt.ylim(0, 15)
    plt.show()


if __name__ == "__main__":
    main()
