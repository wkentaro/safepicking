#!/usr/bin/env python

import json

import numpy as np
import pandas
import path
import sklearn.metrics


def main():
    logs_dir = path.Path("logs")

    data = []
    for log_dir in sorted(logs_dir.listdir()):
        if not log_dir.isdir():
            continue

        for eval_dir in log_dir.glob("eval-*"):
            for json_file in eval_dir.walk("*.json"):
                with open(json_file) as f:
                    json_data = json.load(f)

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
    df2 = df.sort_values(["scene_id", "eval_dir"]).set_index(
        ["scene_id", "eval_dir"]
    )
    print("# Mean over all")
    print(df2.mean(level=1).sort_values("sum_of_translations"))
    print()

    data = []
    for eval_dir in df["eval_dir"].unique():
        df_eval_dir = df[df["eval_dir"] == eval_dir]
        x = np.linspace(0, 2.0)
        y = [(df_eval_dir["sum_of_translations"] < xi).mean() for xi in x]
        auc = sklearn.metrics.auc(x, y) / (x.max() - x.min())
        data.append(
            {
                "eval_dir": eval_dir,
                "auc": auc,
            }
        )
    df = pandas.DataFrame(data)
    df = df.set_index("eval_dir")
    print(df.sort_values("auc").iloc[::-1])


if __name__ == "__main__":
    main()
