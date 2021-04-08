#!/usr/bin/env python

import numpy as np
import pybullet as p

import mercury

import utils


def main():
    parser = utils.get_parser()
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

    step_simulation = utils.StepSimulation(ri=ri)
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

        for _ in ri.random_grasp([plane], object_ids):
            step_simulation()

        # ---------------------------------------------------------------------

        if ri.gripper.grasped_object:
            p.removeBody(ri.gripper.grasped_object)
        ri.ungrasp()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
