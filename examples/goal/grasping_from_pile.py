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

    if args.pause:
        print("Please press 'n' to start")
        while True:
            if ord("n") in p.getKeyboardEvents():
                break

    c_camera_to_world = mercury.geometry.Coordinate()
    c_camera_to_world.rotate([0, 0, np.deg2rad(-90)])
    c_camera_to_world.rotate([np.deg2rad(-180), 0, 0])
    c_camera_to_world.translate([0.5, -0.5, 0.7], wrt="world")

    fovy = np.deg2rad(60)
    height = 480
    width = 640
    pybullet_planning.draw_pose(c_camera_to_world.pose)
    mercury.pybullet.draw_camera(fovy, width, height, c_camera_to_world.pose)

    while True:
        rgb, depth, segm = mercury.pybullet.get_camera_image(
            c_camera_to_world.matrix, fovy=fovy, height=height, width=width
        )
        K = mercury.geometry.opengl_intrinsic_matrix(fovy, height, width)

        mask = ~np.isnan(depth) & (segm != 0)

        if 0:
            imgviz.io.cv_imshow(imgviz.tile([rgb, np.uint8(mask) * 255]))
            imgviz.io.cv_waitkey(10)

        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )

        T_camera_to_world = c_camera_to_world.matrix
        ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
        T_ee_to_world = mercury.geometry.transformation_matrix(*ee_to_world)
        T_camera_to_ee = np.linalg.inv(T_ee_to_world) @ T_camera_to_world
        pcd_in_ee = mercury.geometry.transform_points(
            pcd_in_camera, T_camera_to_ee
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
            break

        for _ in ri.movej(j):
            p.stepSimulation()
            time.sleep(1 / 240)

        for i in ri.grasp(dz=0.1):
            p.stepSimulation()
            time.sleep(1 / 240)
            if i > 5 * 240:
                print("Warning: grasping is timeout")
                break
        mercury.pybullet.step_and_sleep(1)

        for _ in ri.movej(ri.homej):
            p.stepSimulation()
            time.sleep(1 / 240)
        mercury.pybullet.step_and_sleep(1)

        if ri.gripper.check_grasp():
            p.removeBody(ri.gripper.grasped_object)
        ri.ungrasp()

        mercury.pybullet.step_and_sleep(1)

    while True:
        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()
