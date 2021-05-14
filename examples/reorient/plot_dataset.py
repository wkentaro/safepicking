#!/usr/bin/env python

import imgviz
import matplotlib.pyplot as plt
import numpy as np
import path
import sklearn.decomposition


home = path.Path("~").expanduser()


def main():
    root_dir = home / "data/mercury/reorient/00001000"
    grasp_pose = []
    reorient_pose = []
    js_place_length = []
    auc = []
    for npz_file in sorted(root_dir.listdir()):
        result = np.load(npz_file)
        grasp_pose.append(result["grasp_pose"])
        reorient_pose.append(result["reorient_pose"])
        js_place_length.append(result["js_place_length"])
        auc.append(result["auc"])
    grasp_pose = np.array(grasp_pose)
    reorient_pose = np.array(reorient_pose)
    js_place_length = np.array(js_place_length)
    auc = np.array(auc)

    x = sklearn.decomposition.PCA(n_components=1).fit_transform(grasp_pose)[
        :, 0
    ]
    y = sklearn.decomposition.PCA(n_components=1).fit_transform(reorient_pose)[
        :, 0
    ]

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    c = (
        imgviz.depth2rgb(js_place_length[None, :], min_value=0, max_value=10)[
            0
        ]
        / 255
    )
    plt.scatter(x, y, c=c)
    plt.title("js_place_length")
    plt.xlabel("Grasp pose")
    plt.ylabel("Reorient pose")

    plt.subplot(122)
    c = imgviz.depth2rgb(auc[None, :], min_value=0, max_value=1)[0] / 255
    plt.scatter(x, y, c=c)
    plt.title("auc")
    plt.xlabel("Grasp pose")
    plt.ylabel("Reorient pose")
    plt.show()


if __name__ == "__main__":
    main()
