#!/usr/bin/env python

import itertools

import numpy as np
import pybullet as p
import pybullet_planning

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

    for obj in object_ids:
        pybullet_planning.draw_pose(
            ([0, 0, 0], [0, 0, 0, 1]), length=0.15, width=3, parent=obj
        )

    step_simulation = utils.StepSimulation(ri=ri)
    step_simulation()

    utils.pause(args.pause)

    table = pybullet_planning.create_box(
        0.4, 0.4, 0.2, color=[150 / 255, 111 / 255, 51 / 255, 1]
    )
    pybullet_planning.set_pose(table, ([0, 0.6, 0.1], [0, 0, 0, 1]))
    aabb = pybullet_planning.get_aabb(table)
    regrasp_aabb = (
        [aabb[0][0] + 0.1, aabb[0][1] + 0.1, aabb[1][2]],
        [aabb[1][0] - 0.1, aabb[1][1] - 0.1, aabb[1][2] + 0.001],
    )
    pybullet_planning.draw_aabb(regrasp_aabb)

    np.random.seed(2)

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

        if not ri.gripper.check_grasp():
            ri.ungrasp()
            continue

        done = False
        for i in itertools.count():
            if i > 0:
                while True:
                    c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
                    c.position = [0, 0.5, 0.7]
                    c.rotate([0, 0, np.deg2rad(-90)])
                    c.rotate([np.deg2rad(10), 0, 0])
                    j = ri.solve_ik(
                        c.pose, move_target=ri.robot_model.camera_link
                    )
                    for _ in ri.movej(j):
                        step_simulation()
                    for _ in ri.random_grasp([plane, table], object_ids):
                        step_simulation()
                    if ri.gripper.check_grasp():
                        break
                    ri.ungrasp()

                if done:
                    break

            done = utils.place_to_regrasp(
                ri,
                regrasp_aabb,
                bg_object_ids=[plane, table],
                object_ids=object_ids,
                step_simulation=step_simulation,
            )

        p.removeBody(ri.gripper.grasped_object)
        ri.ungrasp()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
