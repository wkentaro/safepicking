#!/usr/bin/env python

import pickle

import imgviz
import numpy as np


data = pickle.load(open("00000000.pkl", "rb"))

rgb = data["rgb"]

yx = np.array(data["yx"])
auc = np.array(data["auc"])

label = np.full(rgb.shape[:2], 0.5, dtype=np.float32)
for i in range(len(yx)):
    y, x = yx[i][0], yx[i][1]
    label[y - 1 : y + 1, x - 1 : x + 1] = auc[i] > 0.9

label = imgviz.depth2rgb(label.astype(np.float32), min_value=0, max_value=1)
label = np.uint8(imgviz.gray2rgb(imgviz.rgb2gray(rgb)) * 0.3 + label * 0.7)
viz = imgviz.tile([rgb, label])
viz = imgviz.resize(viz, width=1200)

imgviz.io.pyglet_imshow(viz)
imgviz.io.pyglet_run()
