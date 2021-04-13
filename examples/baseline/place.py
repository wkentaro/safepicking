#!/usr/bin/env python

import itertools

import numpy as np
import pybullet_planning
from loguru import logger

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

    table = pybullet_planning.create_box(
        0.4, 0.4, 0.2, color=[150 / 255, 111 / 255, 51 / 255, 1]
    )
    pybullet_planning.set_pose(table, ([0, 0.5, 0.1], [0, 0, 0, 1]))
    aabb = pybullet_planning.get_aabb(table)
    regrasp_aabb = (
        [aabb[0][0] + 0.1, aabb[0][1] + 0.1, aabb[1][2]],
        [aabb[1][0] - 0.1, aabb[1][1] - 0.1, aabb[1][2] + 0.001],
    )
    pybullet_planning.draw_aabb(regrasp_aabb)

    place_aabb = ((0.4, 0.3, 0), (0.8, 0.7, 0.2))
    pybullet_planning.draw_aabb(place_aabb, width=2)

    step_simulation = utils.StepSimulation(
        ri=ri, imshow=args.imshow, retime=args.retime
    )
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
        for _ in ri.random_grasp([plane], object_ids):
            step_simulation()
            i += 1
        if i == 0:
            logger.success("Completed the task")
            break

        if not ri.gripper.check_grasp():
            ri.ungrasp()
            continue

        done = False
        for i in itertools.count():
            if i > 0:
                while True:
                    c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
                    c.position = [0, 0.4, 0.7]
                    c.rotate([0, 0, np.deg2rad(-90)])
                    c.rotate([np.deg2rad(10), 0, 0])
                    j = ri.solve_ik(
                        c.pose, move_target=ri.robot_model.camera_link
                    )
                    for _ in ri.movej(j):
                        step_simulation()
                    for _ in ri.random_grasp([plane, table], object_ids):
                        step_simulation()
                    if not ri.gripper.check_grasp():
                        ri.ungrasp()
                        continue

                    object_id = ri.attachments[0].child
                    place_pose = utils.get_place_pose(
                        object_id=object_id,
                        bin_aabb_min=place_aabb[0],
                        bin_aabb_max=place_aabb[1],
                    )
                    with ri.enabling_attachments():
                        j = ri.solve_ik(
                            place_pose,
                            move_target=ri.robot_model.attachment_link0,
                        )
                    if j is None:
                        logger.warning("j is None")
                        done = False
                        break

                    obstacles = [plane, table] + object_ids
                    obstacles.remove(object_id)
                    path = ri.planj(
                        j, obstacles=obstacles, attachments=ri.attachments
                    )
                    if path is None:
                        logger.warning("path is None")
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

        obj_v = mercury.pybullet.duplicate(
            object_id,
            collision=False,
            rgba_color=[0, 1, 0, 0.5],
            position=place_pose[0],
            quaternion=place_pose[1],
        )
        step_simulation.append_obj_v(obj_v)

        for _ in (_ for j in path for _ in ri.movej(j, speed=0.005)):
            step_simulation()

        for _ in range(240):
            step_simulation()

        ri.ungrasp()

        for _ in range(240):
            step_simulation()

        c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
        c.translate([0, 0, -0.05])
        j = ri.solve_ik(c.pose, rotation_axis=None)
        for _ in ri.movej(j, speed=0.005):
            step_simulation()

        max_distance = 0
        path = None
        while path is None:
            path = ri.planj(
                ri.homej,
                obstacles=[plane, table] + object_ids,
                max_distance=max_distance,
            )
            max_distance -= 0.01
        for _ in (_ for j in path for _ in ri.movej(j)):
            step_simulation()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
