#!/usr/bin/env python

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury

import baseline_utils
import common_utils


here = path.Path(__file__).abspath().parent


def main():
    parser = baseline_utils.get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    pp.connect()
    plane = common_utils.init_simulation(camera_distance=1.2)

    ri = mercury.pybullet.PandaRobotInterface()
    ri.add_camera(
        pose=baseline_utils.get_camera_pose(args.camera_config),
        height=240,
        width=320,
    )

    pile_pose = ([0, -0.5, 0], [0, 0, 0, 1])
    object_ids = baseline_utils.load_pile(
        base_pose=pile_pose,
        npz_file="assets/pile_001.npz",
        enable_visual=args.enable_visual,
        mass=0.1,
    )
    for obj in object_ids:
        pp.draw_pose(
            ([0, 0, 0], [0, 0, 0, 1]), length=0.15, width=3, parent=obj
        )

    table = pp.create_box(
        0.4, 0.4, 0.1, color=[150 / 255, 111 / 255, 51 / 255, 1]
    )
    pp.set_pose(table, ([0.5, 0, 0.1], [0, 0, 0, 1]))
    aabb = pp.get_aabb(table)
    regrasp_aabb = (
        [aabb[0][0] + 0.1, aabb[0][1] + 0.1, aabb[1][2]],
        [aabb[1][0] - 0.1, aabb[1][1] - 0.1, aabb[1][2] + 0.001],
    )
    pp.draw_aabb(regrasp_aabb)

    place_aabb = ((-0.3, 0.3, 0), (0.3, 0.6, 0.2))
    pp.draw_aabb(place_aabb, width=2)

    step_simulation = baseline_utils.StepSimulation(
        ri=ri,
        retime=args.retime,
        video_dir=here / "logs/grasp/video" if args.video else None,
    )
    step_simulation()

    baseline_utils.pause(args.pause)

    while True:
        ri.homej[0] = -np.pi / 2
        for _ in ri.movej(ri.homej):
            step_simulation()

        c = mercury.geometry.Coordinate(*ri.get_pose("camera_link"))
        c.position = pile_pose[0]
        c.position[2] = 0.7
        c.quaternion = mercury.geometry.quaternion_from_euler(
            [np.pi, 0, np.pi / 2]
        )
        j = ri.solve_ik(c.pose, move_target=ri.robot_model.camera_link)
        for _ in ri.movej(j):
            step_simulation()

        i = 0
        _, depth, segm = ri.get_camera_image()
        for _ in ri.random_grasp(depth, segm, [plane], object_ids):
            step_simulation()
            i += 1
        for _ in ri.move_to_homej([plane, table], object_ids):
            step_simulation()
        if i == 0:
            logger.success("Completed the task")
            break

        if not ri.gripper.check_grasp():
            ri.ungrasp()
            continue

        ri.homej[0] = 0
        for _ in ri.move_to_homej([plane, table], object_ids):
            step_simulation()

        object_id = ri.attachments[0].child

        pp.remove_body(object_id)
        ri.ungrasp()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
