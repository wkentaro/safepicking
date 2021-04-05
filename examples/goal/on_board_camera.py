#!/usr/bin/env python

import argparse
import time

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning

import mercury


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

        ri = mercury.pybullet.PandaRobotInterface()

        data = np.load("assets/pile_001.npz")
        object_ids = []
        for class_id, position, quaternion in zip(
            data["class_ids"], data["positions"], data["quaternions"]
        ):
            coord = mercury.geometry.Coordinate(
                position=position,
                quaternion=quaternion,
            )
            coord.translate([0.4, -0.4, 0], wrt="world")

            visual_file = mercury.datasets.ycb.get_visual_file(class_id)
            collision_file = mercury.pybullet.get_collision_file(visual_file)
            object_id = mercury.pybullet.create_mesh_body(
                # visual_file=visual_file,
                visual_file=collision_file,
                rgba_color=imgviz.label_colormap()[class_id] / 255,
                collision_file=collision_file,
                mass=0.1,
                position=coord.position,
                quaternion=coord.quaternion,
            )
            object_ids.append(object_id)

    c_cam_to_ee = mercury.geometry.Coordinate()
    c_cam_to_ee.rotate([0, 0, np.deg2rad(45)])
    c_cam_to_ee.translate([0.0, -0.05, -0.1])

    ri.add_camera(
        pose=c_cam_to_ee.pose,
        fovy=np.deg2rad(42),
        height=480,
        width=640,
    )

    if args.pause:
        print("Please press 'n' to start")
        while True:
            if ord("n") in p.getKeyboardEvents():
                break

    c = mercury.geometry.Coordinate(*ri.get_pose("camera_link"))
    c.position = [0.4, -0.4, 0.7]
    j = ri.solve_ik(c.pose, move_target=ri.robot_model.camera_link)
    for _ in ri.movej(j):
        p.stepSimulation()
        time.sleep(1 / 240)

    rgb, _, _ = ri.get_camera_image()
    imgviz.io.pyglet_imshow(rgb)
    imgviz.io.pyglet_run()

    while True:
        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()
