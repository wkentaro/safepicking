#!/usr/bin/env python

import argparse

import numpy as np
import pybullet as p
import pybullet_planning

import mercury

from create_bin import create_bin


def get_place_pose(object_id, class_id, bin_aabb_min, bin_aabb_max):
    position_org, quaternion_org = p.getBasePositionAndOrientation(object_id)

    if class_id == 15:
        quaternion = mercury.geometry.quaternion_from_euler(
            [np.deg2rad(90), 0, 0]
        )
    else:
        quaternion = [0, 0, 0, 1]
    p.resetBasePositionAndOrientation(object_id, position_org, quaternion)

    aabb_min, aabb_max = np.array(p.getAABB(object_id))
    position_lt = bin_aabb_min - (aabb_min - position_org)
    position_rb = bin_aabb_max + (aabb_min - position_org)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    z = position_lt[2]
    for x in np.linspace(position_lt[0], position_rb[0]):
        for y in np.linspace(position_lt[1], position_rb[1]):
            position = (x, y, z)
            p.resetBasePositionAndOrientation(object_id, position, quaternion)
            if not mercury.pybullet.is_colliding(object_id):
                break
        else:
            continue
        break
    else:
        position, quaternion = None, None

    p.resetBasePositionAndOrientation(object_id, position_org, quaternion_org)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    return position, quaternion


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pause", action="store_true", help="pause")
    args = parser.parse_args()

    pybullet_planning.connect()
    pybullet_planning.add_data_path()
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,
        cameraPitch=-50,
        cameraTargetPosition=(0, 0, 0),
    )

    with pybullet_planning.LockRenderer():
        p.loadURDF("plane.urdf")

        mercury.pybullet.PandaRobotInterface()

        data = np.load("assets/pile_001.npz")
        object_ids = []
        for class_id, position, quaternion in zip(
            data["class_ids"], data["positions"], data["quaternions"]
        ):
            coord = mercury.geometry.Coordinate(
                position=position,
                quaternion=quaternion,
            )
            coord.translate([0.5, -0.5, 0], wrt="world")

            visual_file = mercury.datasets.ycb.get_visual_file(class_id)
            collision_file = mercury.pybullet.get_collision_file(visual_file)
            object_id = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=collision_file,
                mass=0.1,
                position=coord.position,
                quaternion=coord.quaternion,
            )
            object_ids.append(object_id)

        bin_id = create_bin(0.4, 0.35, 0.2)
        p.resetBasePositionAndOrientation(
            bin_id, posObj=[0.5, 0.5, 0.1], ornObj=[0, 0, 0, 1]
        )

    if args.pause:
        print("Please press 'n' to start")
        while True:
            if ord("n") in p.getKeyboardEvents():
                break

    bin_aabb_min, bin_aabb_max = mercury.pybullet.get_aabb(bin_id)
    bin_aabb_min += 0.01
    bin_aabb_max -= 0.01

    for object_id, class_id in zip(object_ids, data["class_ids"]):
        position, quaternion = get_place_pose(
            object_id, class_id, bin_aabb_min, bin_aabb_max
        )
        if position is None or quaternion is None:
            continue
        p.resetBasePositionAndOrientation(object_id, position, quaternion)
        mercury.pybullet.step_and_sleep(0.3)

    while True:
        p.stepSimulation()


if __name__ == "__main__":
    main()
