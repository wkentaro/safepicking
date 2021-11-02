#!/usr/bin/env python

import json
import re

import IPython
import pandas
import path


def main():
    logs_dir = path.Path("logs")

    data = []
    for log_dir in sorted(logs_dir.listdir()):
        if not log_dir.isdir():
            continue

        # if "RRTConnect" in log_dir:
        #     continue

        for eval_dir in log_dir.glob("eval*"):
            m = re.match(r"^eval-noise_(.*)-miss_(.*)$", eval_dir.basename())
            noise, miss = [float(x) for x in m.groups()]

            for json_file in eval_dir.walk("*.json"):
                with open(json_file) as f:
                    try:
                        json_data = json.load(f)
                    except json.decoder.JSONDecodeError:
                        continue

                assert json_file.stem == json_data["scene_id"]
                assert json_data["seed"] == 0
                data.append(
                    {
                        "log_dir": str(log_dir.stem),
                        "scene_id": json_data["scene_id"],
                        "noise": noise,
                        "miss": miss,
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

    N = 3 * 10 + 3

    count = df.groupby(["scene_id"]).count()
    valid_scene_ids = (
        count[count == N].dropna().index.get_level_values("scene_id")
    )

    valid_scene_ids = valid_scene_ids[:600]
    print("# of valid scene_ids:", len(valid_scene_ids))

    df_valid = df[df["scene_id"].isin(valid_scene_ids)]

    if 0:
        for scene_id in df["scene_id"]:
            assert len(df_valid[df_valid["scene_id"] == scene_id]) in [0, N]

    # summarize
    print(df_valid.groupby(["log_dir", "noise", "miss"]).mean())

    a = df_valid.groupby(["log_dir", "noise", "miss"]).mean().reset_index()
    print(
        a[(a["noise"] == 0) & (a["miss"] == 0)].set_index(
            ["log_dir", "noise", "miss"]
        )
    )

    b = a[(a["noise"] == 0.3) & (a["miss"] != 0)]
    print(b.groupby("log_dir").mean())

    if 0:
        IPython.embed()


if __name__ == "__main__":
    main()
