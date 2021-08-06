#!/usr/bin/env python

import json

import pandas
import path


def main():
    logs_dir = path.Path("logs")

    data = []
    for log_dir in sorted(logs_dir.listdir()):
        if not log_dir.isdir():
            continue

        for eval_dir in log_dir.glob("eval*"):
            if "noise_" in eval_dir or "miss_" in eval_dir:
                continue
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
    pandas.set_option("display.float_format", "{:.3f}".format)

    df = pandas.DataFrame(data)
    df2 = df.sort_values(["scene_id", "eval_dir"]).set_index(
        ["scene_id", "eval_dir"]
    )
    df3 = df2.count(level=0)
    valid_scene_ids = df3[
        df3["target_object_visibility"] == df["eval_dir"].unique().size
    ].index.values[:600]

    print(f"Support: {len(valid_scene_ids)}")
    print()

    df = df[df["scene_id"].isin(valid_scene_ids)]
    df = (
        df.sort_values(["scene_id", "eval_dir"])
        .set_index(["scene_id", "eval_dir"])
        .mean(level=1)
    )

    df = df.reset_index()
    df = df.drop(columns=["target_object_visibility"])
    df = df.sort_values("eval_dir")
    print(df)


if __name__ == "__main__":
    main()
