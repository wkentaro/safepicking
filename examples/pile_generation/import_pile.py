#!/usr/bin/env python

import argparse
import time

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

import mercury

import common_utils


here = path.Path(__file__).abspath().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("export_file", type=path.Path, help="export file")
    args = parser.parse_args()

    pp.connect()
    common_utils.init_simulation(camera_distance=1)

    data = np.load(args.export_file)
    num_instances = len(data["class_id"])
    object_ids = []
    for i in range(num_instances):
        class_id = data["class_id"][i]
        position = data["position"][i]
        quaternion = data["quaternion"][i]

        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)

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
        )
        object_ids.append(object_id)

    while True:
        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()