#!/usr/bin/env python

import time

import imgviz
import numpy as np
import pybullet as p

import mercury

import utils


def main():
    parser = utils.get_parser()
    parser.add_argument("--imshow", action="store_true", help="imshow")
    args = parser.parse_args()

    plane = utils.init_world()

    ri = mercury.pybullet.PandaRobotInterface()
    ri.add_camera(
        pose=utils.get_camera_pose(args.camera_config),
        height=240,
        width=320,
    )

    object_ids = utils.load_pile(
        base_pose=([0.4, -0.4, 0], [0, 0, 0, 1]),
        npz_file="assets/pile_001.npz",
        enable_visual=args.enable_visual,
        mass=0.1,
    )

    class StepSimulation:
        def __init__(self):
            self.i = 0

        def __call__(self):
            p.stepSimulation()
            ri.step_simulation()
            if args.imshow and self.i % 8 == 0:
                rgb, depth, _ = ri.get_camera_image()
                depth[(depth < 0.3) | (depth > 2)] = np.nan
                tiled = imgviz.tile(
                    [
                        rgb,
                        imgviz.depth2rgb(depth, min_value=0.3, max_value=0.6),
                    ],
                    border=(255, 255, 255),
                )
                imgviz.io.cv_imshow(tiled)
                imgviz.io.cv_waitkey(1)
            time.sleep(1 / 240)
            self.i += 1

    step_simulation = StepSimulation()
    step_simulation()

    utils.pause(args.pause)

    while True:
        c = mercury.geometry.Coordinate(*ri.get_pose("camera_link"))
        c.position = [0.4, -0.4, 0.7]
        c.quaternion = mercury.geometry.quaternion_from_euler(
            [np.pi, 0, np.pi / 2]
        )
        j = ri.solve_ik(c.pose, move_target=ri.robot_model.camera_link)
        for _ in ri.movej(j):
            step_simulation()

        for _ in ri.random_grasp(plane, object_ids):
            step_simulation()

        if ri.gripper.grasped_object:
            p.removeBody(ri.gripper.grasped_object)
        ri.ungrasp()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
