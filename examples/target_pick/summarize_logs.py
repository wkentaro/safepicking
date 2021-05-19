#!/usr/bin/env python

import json

import numpy as np
import path


def summarize(eval_dir, valid_ids):
    success = []
    sum_of_max_velocities = []
    for id in valid_ids:
        result_file = eval_dir / id
        with open(result_file) as f:
            data = json.load(f)
        success.append(data["success"])
        sum_of_max_velocities.append(data["sum_of_max_velocities"])
    success = np.array(success, dtype=bool)
    sum_of_max_velocities = np.array(sum_of_max_velocities, dtype=float)

    print(f"Eval dir: {eval_dir}")
    print(f"Success: {success.mean():.1%} ({success.sum()} / {success.size})")
    print(f"Unsafety (all): {sum_of_max_velocities.mean():.2f}")
    print(f"Unsafety (success): {sum_of_max_velocities[success].mean():.2f}")
    if not success.all():
        print(
            f"Unsafety (failure): {sum_of_max_velocities[~success].mean():.2f}"
        )
    print()


def main():
    logs_dir = path.Path("logs")

    eval_dir_to_ids = {}
    for log_dir in logs_dir.listdir():
        if not log_dir.isdir():
            continue

        for eval_dir in log_dir.listdir():
            if not eval_dir.stem.startswith("eval-"):
                continue

            ids = []
            for config_dir in eval_dir.listdir():
                for seed_file in config_dir.listdir():
                    ids.append("/".join(seed_file.split("/")[-2:]))
            eval_dir_to_ids[eval_dir] = ids

    all_ids = set(xi for x in eval_dir_to_ids.values() for xi in x)

    valid_ids = set()
    for id in all_ids:
        if all(id in ids for ids in eval_dir_to_ids.values()):
            valid_ids.add(id)

    for eval_dir, ids in eval_dir_to_ids.items():
        summarize(eval_dir, valid_ids)


if __name__ == "__main__":
    main()
