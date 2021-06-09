#!/usr/bin/env python

import json

import numpy as np
import pandas
import path


def summarize(eval_dir, valid_ids):
    data = []
    for id in valid_ids:
        result_file = eval_dir / id
        with open(result_file) as f:
            result = json.load(f)
        data.append(result)
    df = pandas.DataFrame(data)

    df = df[
        [
            "target_object_visibility",
            "sum_of_translations",
            "sum_of_max_velocities",
        ]
    ]

    assert (df["sum_of_translations"] == 0).sum() == 0

    print(f"Eval dir: {eval_dir}")
    print(f"Support: {len(df)}")

    bins = np.linspace(0.2, 0.9, num=8)
    binned = np.digitize(df["target_object_visibility"], bins)

    data = []

    for i in np.arange(5):
        mask = binned == i
        data.append(
            dict(
                visibility=bins[i],
                sum_of_translations=df[mask].mean()["sum_of_translations"],
                sum_of_max_velocities=df[mask].mean()["sum_of_max_velocities"],
            )
        )

    df = pandas.DataFrame(data)
    print(df.dropna())
    print(df.mean()[["sum_of_translations", "sum_of_max_velocities"]])

    print()


def main():
    logs_dir = path.Path("logs")

    eval_dir_to_ids = {}
    for log_dir in logs_dir.listdir():
        if not log_dir.isdir():
            continue

        for eval_dir in log_dir.listdir():
            if not eval_dir.stem.startswith("eval"):
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

    for eval_dir, ids in sorted(eval_dir_to_ids.items()):
        summarize(eval_dir, valid_ids)


if __name__ == "__main__":
    main()
