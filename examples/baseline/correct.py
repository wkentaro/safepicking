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

    plane = utils.init_world(camera_distance=1.2)

    ri = mercury.pybullet.PandaRobotInterface()
    ri.add_camera(
        pose=utils.get_camera_pose(args.camera_config),
        height=240,
        width=320,
    )

    pile_pose = ([0, -0.5, 0], [0, 0, 0, 1])
    object_ids = utils.load_pile(
        base_pose=pile_pose,
        npz_file="assets/pile_001.npz",
        enable_visual=args.enable_visual,
        mass=0.1,
    )
    for obj in object_ids:
        pybullet_planning.draw_pose(
            ([0, 0, 0], [0, 0, 0, 1]), length=0.15, width=3, parent=obj
        )

    table = pybullet_planning.create_box(
        0.4, 0.4, 0.1, color=[150 / 255, 111 / 255, 51 / 255, 1]
    )
    pybullet_planning.set_pose(table, ([0.5, 0, 0.1], [0, 0, 0, 1]))
    aabb = pybullet_planning.get_aabb(table)
    regrasp_aabb = (
        [aabb[0][0] + 0.1, aabb[0][1] + 0.1, aabb[1][2]],
        [aabb[1][0] - 0.1, aabb[1][1] - 0.1, aabb[1][2] + 0.001],
    )
    pybullet_planning.draw_aabb(regrasp_aabb)

    place_aabb = ((-0.3, 0.3, 0), (0.3, 0.6, 0.2))
    pybullet_planning.draw_aabb(place_aabb, width=2)

    step_simulation = utils.StepSimulation(
        ri=ri, imshow=args.imshow, retime=args.retime
    )
    step_simulation()

    utils.pause(args.pause)

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
            for _ in ri.movej(ri.homej):
                step_simulation()
            logger.success("Completed the task")
            break

        if not ri.gripper.check_grasp():
            ri.ungrasp()
            continue

        ri.homej[0] = 0
        for _ in ri.move_to_homej([plane, table], object_ids):
            step_simulation()

        regrasp_pose = np.mean(regrasp_aabb, axis=0)
        done = False
        for i in itertools.count():
            if i > 0:
                while True:
                    c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
                    c.position = regrasp_pose[0]
                    c.position[2] = 0.7
                    j = ri.solve_ik(
                        c.pose,
                        move_target=ri.robot_model.camera_link,
                        rotation_axis="z",
                    )
                    for _ in ri.movej(j):
                        step_simulation()
                    _, depth, segm = ri.get_camera_image()
                    for _ in ri.random_grasp(
                        depth,
                        segm,
                        [plane, table],
                        object_ids,
                        max_angle=np.deg2rad(10),
                    ):
                        step_simulation()
                    for _ in ri.move_to_homej([plane, table], object_ids):
                        step_simulation()
                    if not ri.gripper.check_grasp():
                        ri.ungrasp()
                        continue

                    ri.homej[0] = np.pi / 2
                    with pybullet_planning.LockRenderer(), pybullet_planning.WorldSaver():  # NOQA
                        ri.setj(ri.homej)
                        place_pose, path = utils.plan_placement(
                            ri, place_aabb, [plane, table], object_ids
                        )
                    if path is None:
                        done = False
                        break

                    break

                if done:
                    break

            ri.homej[0] = 0
            regrasp_pose, done = utils.place_to_regrasp(
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
        j_camera = j

        while True:
            obj_to_world = pybullet_planning.get_pose(object_id)

            class_id = utils.get_class_id(object_id)
            pcd_file = mercury.datasets.ycb.get_pcd_file(class_id=class_id)
            pcd = np.loadtxt(pcd_file)
            pcd_target = mercury.geometry.transform_points(
                pcd, mercury.geometry.transformation_matrix(*place_pose)
            )
            pcd_source = mercury.geometry.transform_points(
                pcd, mercury.geometry.transformation_matrix(*obj_to_world)
            )
            auc = mercury.geometry.average_distance_auc(pcd_target, pcd_source)
            logger.info(auc)
            if auc >= 0.5:
                logger.success("auc >= 0.5")
                break

            while True:
                with utils.stash_objects(utils.virtual_objects):
                    _, depth, segm = ri.get_camera_image()
                for _ in ri.random_grasp(
                    depth=depth,
                    segm=segm,
                    bg_object_ids=[plane, table],
                    object_ids=object_ids,
                    target_object_ids=[object_id],
                    max_angle=np.deg2rad(10),
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
            obstacles = [plane, table] + object_ids
            obstacles.remove(object_id)
            path = ri.planj(j, attachments=ri.attachments, obstacles=obstacles)
            if path is None:
                logger.warning("path is None")
                path = [j]
            for _ in (_ for j in path for _ in ri.movej(j, speed=0.001)):
                step_simulation()

            for _ in range(240):
                step_simulation()

            ri.ungrasp()

            for _ in range(240):
                step_simulation()

            c = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
            c.translate([0, 0, -0.05])
            j = ri.solve_ik(c.pose, rotation_axis=None)
            for _ in ri.movej(j):
                step_simulation()

            for _ in ri.movej(j_camera):
                step_simulation()

        for _ in ri.move_to_homej([plane, table], object_ids):
            step_simulation()

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
