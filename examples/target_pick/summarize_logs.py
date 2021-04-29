#!/usr/bin/env python

import argparse
import json
import warnings

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

    success = success.reshape(11, 6)
    sum_of_velocities = sum_of_velocities.reshape(11, 6)

    sum_of_velocities_success = sum_of_velocities.copy()
    sum_of_velocities_success[~success] = np.nan
    sum_of_velocities_failure = sum_of_velocities.copy()
    sum_of_velocities_failure[success] = np.nan

    success = success.mean(axis=1)

    print(f"Log dir: {args.log_dir}")
    print(
        f"Success: {success.mean():.1%} +- {success.std():.1%} "
        f"({success.sum():.1f} / {success.size})"
    )
    print(f"Unsafety (all): {sum_of_velocities.mean(axis=1).mean():.2f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(
            "Unsafety (success): "
            f"{np.nanmean(np.nanmean(sum_of_velocities_success, axis=1)):.2f}"
        )
        print(
            "Unsafety (failure): "
            f"{np.nanmean(np.nanmean(sum_of_velocities_failure, axis=1)):.2f}"
        )


if __name__ == "__main__":
    main()
