#!/usr/bin/env python

import argparse
import time

import imgviz
import numpy as np
import pybullet

import mercury


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npz_file", help="npz file")
    args = parser.parse_args()

    mercury.pybullet.init_world()

    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=-60,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )

    data = np.load(args.npz_file)
    class_ids = data["class_ids"]
    positions = data["positions"]
    quaternions = data["quaternions"]

    for (
        class_id,
        position,
        quaternion,
    ) in zip(class_ids, positions, quaternions):
        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)
        mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            collision_file=collision_file,
            position=position,
            quaternion=quaternion,
            mass=0.1,
            rgba_color=imgviz.label_colormap()[class_id] / 255,
        )

    while True:
        time.sleep(1 / 240)
        pybullet.stepSimulation()


if __name__ == "__main__":
    main()
