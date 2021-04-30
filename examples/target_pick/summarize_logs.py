#!/usr/bin/env python

import argparse
import json

import numpy as np
import path


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("log_dir", type=path.Path, help="log dir")
    args = parser.parse_args()

    success = []
    sum_of_velocities = []
    for scene_dir in sorted(args.log_dir.listdir()):
        assert scene_dir.isdir()
        for result_file in scene_dir.listdir():
            assert result_file.ext == ".json"
            with open(result_file) as f:
                data = json.load(f)
            success.append(data["success"])
            sum_of_velocities.append(data["sum_of_velocities"])
    success = np.array(success, dtype=bool)
    sum_of_velocities = np.array(sum_of_velocities, dtype=float)

    print(f"Log dir: {args.log_dir}")
    print(f"Success: {success.mean():.1%} ({success.sum()} / {success.size})")
    print(f"Unsafety (all): {sum_of_velocities.mean():.2f}")
    print(f"Unsafety (success): {sum_of_velocities[success].mean():.2f}")
    print(f"Unsafety (failure): {sum_of_velocities[~success].mean():.2f}")


if __name__ == "__main__":
    main()
