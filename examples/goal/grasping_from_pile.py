#!/usr/bin/env python

import time

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning

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

        rgb, depth, segm = ri.get_camera_image()
        K = ri.get_opengl_intrinsic_matrix()

        mask = ~np.isnan(depth) & ~np.isin(segm, [plane, ri.robot])

        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )

        camera_to_world = ri.get_pose("camera_link")
        ee_to_world = ri.get_pose("tipLink")
        camera_to_ee = pybullet_planning.multiply(
            pybullet_planning.invert(ee_to_world), camera_to_world
        )
        pcd_in_ee = mercury.geometry.transform_points(
            pcd_in_camera,
            mercury.geometry.transformation_matrix(*camera_to_ee),
        )

        normals = mercury.geometry.normals_from_pointcloud(pcd_in_ee)

        mask = mask.reshape(-1)
        pcd_in_ee = pcd_in_ee.reshape(-1, 3)
        normals = normals.reshape(-1, 3)

        indices = np.where(mask)[0]
        np.random.shuffle(indices)

        if indices.size == 0:
            break

        for index in indices:
            position = pcd_in_ee[index]
            quaternion = mercury.geometry.quaternion_from_vec2vec(
                [0, 0, 1], normals[index]
            )
            T_ee_to_ee_af_in_ee = mercury.geometry.transformation_matrix(
                position, quaternion
            )

            T_ee_to_world = mercury.geometry.transformation_matrix(
                *mercury.pybullet.get_pose(ri.robot, ri.ee)
            )
            T_ee_to_ee = np.eye(4)
            T_ee_af_to_ee = T_ee_to_ee_af_in_ee @ T_ee_to_ee
            T_ee_af_to_world = T_ee_to_world @ T_ee_af_to_ee

            c = mercury.geometry.Coordinate(
                *mercury.geometry.pose_from_matrix(T_ee_af_to_world)
            )
            c.translate([0, 0, -0.1])

            j = ri.solve_ik(c.pose, rotation_axis="z")
            if j is None:
                continue

            path = ri.planj(j, obstacles=[plane] + object_ids)
            if path is None:
                continue

            break
        for _ in (_ for j in path for _ in ri.movej(j)):
            step_simulation()

        for i in ri.grasp(dz=None, speed=0.005):
            step_simulation()
            if i > 5 * 240:
                print("Warning: grasping is timeout")
                break
        mercury.pybullet.step_and_sleep(1)

        obstacles = [plane] + object_ids
        if ri.gripper.grasped_object:
            obstacles.remove(ri.gripper.grasped_object)
            obj_to_world = pybullet_planning.get_pose(
                ri.gripper.grasped_object
            )
            ee_to_world = ri.get_pose("tipLink")
            obj_to_ee = pybullet_planning.multiply(
                pybullet_planning.invert(ee_to_world), obj_to_world
            )
            attachments = [
                pybullet_planning.Attachment(
                    ri.robot, ri.ee, obj_to_ee, ri.gripper.grasped_object
                )
            ]
        else:
            attachments = None
        path = None
        max_distance = 0
        while path is None:
            path = ri.planj(
                ri.homej,
                obstacles=obstacles,
                attachments=attachments,
                max_distance=max_distance,
            )
            max_distance -= 0.01
        speed = 0.005 if ri.gripper.check_grasp() else 0.01
        for _ in (_ for j in path for _ in ri.movej(j, speed=speed)):
            step_simulation()
        mercury.pybullet.step_and_sleep(1)

        if ri.gripper.check_grasp():
            p.removeBody(ri.gripper.grasped_object)
        ri.ungrasp()

        mercury.pybullet.step_and_sleep(1)

    while True:
        step_simulation()


if __name__ == "__main__":
    main()
