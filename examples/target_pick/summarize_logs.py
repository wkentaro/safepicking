#!/usr/bin/env python

import json
import re

import pandas
import path


def main():
    logs_dir = path.Path("logs")

    data = []
    for log_dir in sorted(logs_dir.listdir()):
        if not log_dir.isdir():
            continue

        for eval_dir in log_dir.glob("eval-*"):
            for json_file in eval_dir.walk("*.json"):
                with open(json_file) as f:
                    try:
                        json_data = json.load(f)
                    except json.decoder.JSONDecodeError:
                        continue

                assert int(json_file.stem) == 0
                data.append(
                    {
                        "eval_dir": "/".join(eval_dir.split("/")[-2:]),
                        "scene_id": str(json_file.parent.stem),
                        "target_object_visibility": json_data[
                            "target_object_visibility"
                        ],
                        "sum_of_translations": json_data[
                            "sum_of_translations"
                        ],
                        "sum_of_max_velocities": json_data[
                            "sum_of_max_velocities"
                        ],
                    }
                )

    pandas.set_option("display.max_colwidth", 400)
    pandas.set_option("display.max_columns", 500)
    pandas.set_option("display.width", 1000)

    df = pandas.DataFrame(data)
    df2 = df.sort_values(["scene_id", "eval_dir"]).set_index(
        ["scene_id", "eval_dir"]
    )
    df3 = df2.count(level=0)
    valid_scene_ids = df3[
        df3["target_object_visibility"] == df["eval_dir"].unique().size
    ].index.values

    print(f"Support: {len(valid_scene_ids)}")
    print()

    df = df[df["scene_id"].isin(valid_scene_ids)]
    df = (
        df.sort_values(["scene_id", "eval_dir"])
        .set_index(["scene_id", "eval_dir"])
        .mean(level=1)
    )

    df = df.reset_index()

    data = []
    for _, row in df.iterrows():
        row["log_dir"] = row["eval_dir"].split("/")[0]
        match = re.search(r"noise_(\d\.\d)", row["eval_dir"])
        if match:
            noise = float(match.groups()[0])
        else:
            noise = 0
        match = re.search(r"miss_(\d\.\d)", row["eval_dir"])
        if match:
            miss = float(match.groups()[0])
        else:
            miss = 0
        row["noise"] = noise
        row["miss"] = miss
        data.append(row)

    df = pandas.DataFrame(data)

    methods = [
        "Naive",
        "RRTConnect",
        "Heuristic",
        "20210709_005731-fusion_net-noise",
        "20210706_194543-conv_net",
        "20210709_005731-openloop_pose_net-noise",
    ]
    data = []
    for method in methods:
        row = df[(df["log_dir"] == method) & (df.noise + df.miss == 0)].mean()
        row = row.drop(["target_object_visibility", "noise", "miss"])
        row["method"] = method
        row["noise"] = False
        data.append(dict(row))

        if method in ["Naive", "Heuristic", "20210706_194543-conv_net"]:
            continue
        row = df[(df["log_dir"] == method) & (df.noise + df.miss != 0)].mean()
        row = row.drop(["target_object_visibility", "noise", "miss"])
        row["method"] = method
        row["noise"] = True
        data.append(dict(row))
    df = pandas.DataFrame(data)
    df = df.set_index(["method", "noise"])
    print(df)


if __name__ == "__main__":
    main()
