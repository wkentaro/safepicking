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
    parser.add_argument(
        "--enable-visual", action="store_true", help="enable visual"
    )
    parser.add_argument(
        "--camera-config", type=int, default=0, help="camera config"
    )
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
        plane = p.loadURDF("plane.urdf")

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
            if args.enable_visual:
                visual_file = visual_file
                rgba_color = None
            else:
                visual_file = collision_file
                rgba_color = imgviz.label_colormap()[class_id] / 255
            object_id = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                rgba_color=rgba_color,
                collision_file=collision_file,
                mass=0.1,
                position=coord.position,
                quaternion=coord.quaternion,
            )
            object_ids.append(object_id)

    c_cam_to_ee = mercury.geometry.Coordinate()
    c_cam_to_ee.rotate([0, 0, np.deg2rad(45)])

    if args.camera_config == 0:
        c_cam_to_ee.translate([0, -0.05, -0.1])
    elif args.camera_config == 1:
        c_cam_to_ee.rotate([np.deg2rad(-15), 0, 0])
        c_cam_to_ee.translate([0, -0.08, -0.2])
    elif args.camera_config == 2:
        c_cam_to_ee.rotate([np.deg2rad(-15), 0, 0])
        c_cam_to_ee.translate([0, -0.08, -0.35])
    else:
        raise ValueError

    fovy = np.deg2rad(42)
    height = 480
    width = 640
    ri.add_camera(
        pose=c_cam_to_ee.pose,
        fovy=fovy,
        height=height,
        width=width,
    )

    class StepSimulation:
        def __init__(self):
            self.i = 0

        def __call__(self):
            p.stepSimulation()
            if self.i % 8 == 0:
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

    if args.pause:
        print("Please press 'n' to start")
        while True:
            if ord("n") in p.getKeyboardEvents():
                break

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
        K = mercury.geometry.opengl_intrinsic_matrix(fovy, height, width)

        mask = ~np.isnan(depth) & ~np.isin(segm, [0, 1])

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

        for _ in ri.movej(ri.homej):
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
