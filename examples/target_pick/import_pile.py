#!/usr/bin/env python

import argparse
import time

import imgviz
from loguru import logger
import numpy as np
import path
import pybullet as p

import mercury

import utils


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("export_file", type=path.Path, help="export file")
    parser.add_argument(
        "--enable-visual", action="store_true", help="enable visual"
    )
    args = parser.parse_args()

    utils.init_world(camera_distance=1)

    data = np.load(args.export_file)
    num_instances = len(data["class_id"])
    object_ids = []
    for i in range(num_instances):
        class_id = data["class_id"][i]
        position = data["position"][i]
        quaternion = data["quaternion"][i]

        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)

        if args.enable_visual:
            rgba_color = None
        else:
            visual_file = collision_file
            rgba_color = imgviz.label_colormap()[class_id] / 255

        class_name = mercury.datasets.ycb.class_names[class_id]
        visibility = data["visibility"][i]
        logger.info(
            f"class_id={class_id:02d}, "
            f"class_name={class_name}, "
            f"visibility={visibility:.1%}"
        )

        object_id = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            collision_file=collision_file,
            mass=mercury.datasets.ycb.masses[class_id],
            position=position,
            quaternion=quaternion,
            rgba_color=rgba_color,
        )
        object_ids.append(object_id)

    while True:
        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()
