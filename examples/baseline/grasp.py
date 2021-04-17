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

    step_simulation = utils.StepSimulation(ri=ri, imshow=args.imshow)
    step_simulation()

    utils.pause(args.pause)

    np.random.seed(args.seed)

    while True:
        c = mercury.geometry.Coordinate(*ri.get_pose("camera_link"))
        c.position = [0.4, -0.4, 0.7]
        c.quaternion = mercury.geometry.quaternion_from_euler(
            [np.pi, 0, np.pi / 2]
        )
        j = ri.solve_ik(c.pose, move_target=ri.robot_model.camera_link)
        for _ in ri.movej(j):
            step_simulation()

        i = 0
        _, depth, segm = ri.get_camera_image()
        for _ in ri.random_grasp(
            depth, segm, bg_object_ids=[plane], object_ids=object_ids
        ):
            step_simulation()
            i += 1
        if i == 0:
            print("Completed the task")
            break

        for _ in ri.move_to_homej([plane], object_ids):
            step_simulation()

        utils.draw_grasped_object(ri)

        for _ in range(240):
            step_simulation()

        # ---------------------------------------------------------------------

        if ri.gripper.grasped_object:
            p.removeBody(ri.gripper.grasped_object)
        ri.ungrasp()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
