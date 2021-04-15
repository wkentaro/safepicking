#!/usr/bin/env python

import itertools

from loguru import logger
import numpy as np
import pybullet_planning

import mercury

import utils


def main():
    parser = utils.get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

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

    table = pybullet_planning.create_box(
        0.4, 0.4, 0.2, color=[150 / 255, 111 / 255, 51 / 255, 1]
    )
    pybullet_planning.set_pose(table, ([-0.4, 0.4, 0.1], [0, 0, 0, 1]))
    aabb = pybullet_planning.get_aabb(table)
    regrasp_aabb = (
        [aabb[0][0] + 0.1, aabb[0][1] + 0.1, aabb[1][2]],
        [aabb[1][0] - 0.1, aabb[1][1] - 0.1, aabb[1][2] + 0.001],
    )
    pybullet_planning.draw_aabb(regrasp_aabb)

    place_aabb = ((0.2, 0.2, 0), (0.7, 0.6, 0.2))
    pybullet_planning.draw_aabb(place_aabb, width=2)

    step_simulation = utils.StepSimulation(
        ri=ri, imshow=args.imshow, retime=args.retime
    )
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

        i = 0
        for _ in ri.random_grasp([plane], object_ids):
            step_simulation()
            i += 1
        for _ in ri.move_to_homej([plane, table], object_ids):
            step_simulation()
        if i == 0:
            for _ in ri.movej(ri.homej):
                step_simulation()
            logger.success("Completed the task")
            break

        if not ri.gripper.check_grasp():
            ri.ungrasp()
            continue

        ri.homej[0] = np.pi / 2
        for _ in ri.movej(ri.homej):
            step_simulation()

        done = False
        for i in itertools.count():
            if i > 0:
                while True:
                    c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
                    c.position = np.mean(regrasp_aabb, axis=0)
                    c.position[2] = 0.7
                    j = ri.solve_ik(
                        c.pose,
                        move_target=ri.robot_model.camera_link,
                        rotation_axis="z",
                    )
                    for _ in ri.movej(j):
                        step_simulation()
                    for _ in ri.random_grasp([plane, table], object_ids):
                        step_simulation()
                    for _ in ri.move_to_homej([plane, table], object_ids):
                        step_simulation()
                    if not ri.gripper.check_grasp():
                        ri.ungrasp()
                        continue

                    place_pose, path = utils.plan_placement(
                        ri, place_aabb, [plane, table], object_ids
                    )
                    if path is None:
                        done = False
                        break

                    break

                if done:
                    break

            done = utils.place_to_regrasp(
                ri,
                regrasp_aabb,
                bg_object_ids=[plane, table],
                object_ids=object_ids,
                step_simulation=step_simulation,
            )

        object_id = ri.attachments[0].child

        utils.place(
            ri,
            object_id,
            place_pose,
            path,
            bg_object_ids=[plane, table],
            object_ids=object_ids,
            step_simulation=step_simulation,
        )

        ri.homej[0] = 0
        for _ in ri.movej(ri.homej):
            step_simulation()

        c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
        c.position = place_pose[0]
        c.position[2] = 0.7
        j = ri.solve_ik(
            c.pose,
            move_target=ri.robot_model.camera_link,
            rotation_axis="z",
        )
        if j is None:
            continue
        for _ in ri.movej(j):
            step_simulation()

        while True:
            for _ in ri.random_grasp(
                [plane, table],
                object_ids,
                target_object_ids=[object_id],
            ):
                step_simulation()
            if not ri.gripper.check_grasp():
                ri.ungrasp()
                for _ in ri.movej(j):
                    step_simulation()
                continue
            break

        with ri.enabling_attachments():
            j = ri.solve_ik(
                place_pose, move_target=ri.robot_model.attachment_link0
            )
        for _ in ri.movej(j, speed=0.001):
            step_simulation()

        for _ in range(120):
            step_simulation()

        ri.ungrasp()

        for _ in range(120):
            step_simulation()

        c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
        c.translate([0, 0, -0.05])
        j = ri.solve_ik(c.pose, rotation_axis=None)
        for _ in ri.movej(j):
            step_simulation()

        for _ in ri.move_to_homej([plane, table], object_ids):
            step_simulation()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
