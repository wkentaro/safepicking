#!/usr/bin/env python

import argparse
import time

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning

import mercury

from bin_packing_no_act import get_place_pose


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pause", action="store_true", help="pause")
    parser.add_argument(
        "--enable-visual", action="store_true", help="enable visual"
    )
    parser.add_argument("--class-id", type=int, default=3, help="class id")
    args = parser.parse_args()

    pybullet_planning.connect()
    pybullet_planning.add_data_path()
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.3,
        cameraYaw=120,
        cameraPitch=-40,
        cameraTargetPosition=(0, 0, 0.2),
    )

    with pybullet_planning.LockRenderer():
        plane = p.loadURDF("plane.urdf")

        bin = mercury.pybullet.create_bin(0.4, 0.4, 0.2)
        p.resetBasePositionAndOrientation(bin, [0.5, 0.5, 0.11], [0, 0, 0, 1])
        bin_aabb = np.array(mercury.pybullet.get_aabb(bin))
        bin_aabb[0] += 0.01
        bin_aabb[1] -= 0.01

        ri = mercury.pybullet.PandaRobotInterface()

        class_id = args.class_id
        visual_file = mercury.datasets.ycb.get_visual_file(class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)
        spawn_space = ((0.2, -0.1, 0), (0.4, -0.4, 0))
        position = np.random.uniform(*spawn_space)
        quaternion = np.random.random((4,))
        quaternion /= np.linalg.norm(quaternion)
        if args.enable_visual:
            rgba_color = None
        else:
            visual_file = collision_file
            rgba_color = imgviz.label_colormap()[class_id] / 255
        obj = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            rgba_color=rgba_color,
            collision_file=collision_file,
            position=position,
            quaternion=quaternion,
            mass=0.1,
        )
        aabb_min, aabb_max = pybullet_planning.get_aabb(obj)
        p.resetBasePositionAndOrientation(
            obj, position + (position - aabb_min), quaternion
        )

    for _ in range(240 * 3):
        p.stepSimulation()

    if args.pause:
        print("Please press 'n' to start")
        while True:
            if ord("n") in p.getKeyboardEvents():
                break

    # XXX: estimate object pose
    obj_to_world = mercury.pybullet.get_pose(obj)
    pybullet_planning.draw_pose(obj_to_world)

    # pre-grasp-pose
    c = mercury.geometry.Coordinate(
        *mercury.pybullet.get_pose(ri.robot, ri.ee)
    )
    c.position = obj_to_world[0]
    c.position[2] = pybullet_planning.get_aabb(obj)[1][2] + 0.1
    path = ri.planj(ri.solve_ik(c.pose, rotation_axis="z"), obstacles=[plane])
    for _ in (_ for j in path for _ in ri.movej(j)):
        p.stepSimulation()
        time.sleep(1 / 240)

    for _ in ri.grasp():
        p.stepSimulation()
        time.sleep(1 / 240)

    ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
    world_to_obj = pybullet_planning.invert(obj_to_world)
    ee_to_obj = pybullet_planning.multiply(world_to_obj, ee_to_world)
    obj_to_ee = pybullet_planning.invert(ee_to_obj)
    attachments = [
        pybullet_planning.Attachment(ri.robot, ri.ee, obj_to_ee, obj)
    ]

    robot_model = ri.get_skrobot(attachments)

    for j in ri.movej(ri.homej):
        p.stepSimulation()
        time.sleep(1 / 240)

    while True:
        with pybullet_planning.WorldSaver():
            p.resetBasePositionAndOrientation(obj, [0, 0, 0], [0, 0, 0, 1])
            aabb_min, _ = pybullet_planning.get_aabb(obj)
            obj_to_world_target = (
                np.random.uniform([0.0, -0.2, 0.01], [0.4, -0.6, 0.01])
                + -aabb_min,
                [0, 0, 0, 1],
            )
        joint_positions = robot_model.inverse_kinematics(
            mercury.geometry.Coordinate(*obj_to_world_target).skrobot_coords,
            move_target=robot_model.attachment_link0,
            rotation_axis="z",
        )
        if joint_positions is False:
            continue

        for max_distance in [0, -0.01, -0.02]:
            path = ri.planj(
                joint_positions[:-1],
                obstacles=[plane, bin],
                attachments=attachments,
                max_distance=max_distance,
            )
            if path:
                break
        if path is None:
            continue

        break
    for _ in (_ for j in path for _ in ri.movej(j)):
        p.stepSimulation()
        time.sleep(1 / 240)

    mercury.pybullet.step_and_sleep(1)

    ri.ungrasp()

    obj_to_world = mercury.geometry.Coordinate.from_matrix(
        robot_model.attachment_link0.worldcoords().T()
    ).pose
    pybullet_planning.draw_pose(obj_to_world)

    # reset-pose
    c = mercury.geometry.Coordinate(
        *mercury.pybullet.get_pose(ri.robot, ri.ee)
    )
    c.translate([0, 0, -0.01])
    for _ in ri.movej(ri.solve_ik(c.pose, rotation_axis="z")):
        p.stepSimulation()
        time.sleep(1 / 240)
    path = None
    while path is None:
        path = ri.planj(ri.homej, obstacles=[plane, bin, obj])
    for _ in (_ for j in path for _ in ri.movej(j)):
        p.stepSimulation()
        time.sleep(1 / 240)

    c = mercury.geometry.Coordinate(
        *mercury.pybullet.get_pose(ri.robot, ri.ee)
    )
    c.position = obj_to_world[0]
    c.position[2] = pybullet_planning.get_aabb(obj)[1][2] + 0.1

    # pre-grasp-pose
    for _ in ri.movej(ri.solve_ik(c.pose)):
        p.stepSimulation()
        time.sleep(1 / 240)

    for _ in ri.grasp(dz=None):
        p.stepSimulation()
        time.sleep(1 / 240)

    ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
    obj_to_ee = pybullet_planning.multiply(
        pybullet_planning.invert(ee_to_world), obj_to_world
    )
    obj_to_ee = mercury.pybullet.get_pose(
        obj, parent_body_id=ri.robot, parent_link_id=ri.ee
    )
    attachments = [
        pybullet_planning.Attachment(ri.robot, ri.ee, obj_to_ee, obj)
    ]

    # reset-pose
    path = ri.planj(ri.homej, obstacles=[plane])
    for _ in (_ for j in path for _ in ri.movej(j)):
        p.stepSimulation()
        time.sleep(1 / 240)

    # place-pose
    place_pose = get_place_pose(obj, class_id, bin_aabb[0], bin_aabb[1])
    while True:
        robot_model = ri.get_skrobot(attachments)
        j = robot_model.inverse_kinematics(
            mercury.geometry.Coordinate(*place_pose).skrobot_coords,
            move_target=robot_model.attachment_link0,
        )
        if j is False:
            continue
        path = ri.planj(
            j[:-1],
            obstacles=[plane, bin],
            attachments=attachments,
        )
        if path is None:
            continue
        break
    for _ in (_ for j in path for _ in ri.movej(j)):
        p.stepSimulation()
        time.sleep(1 / 240)

    mercury.pybullet.step_and_sleep(1)

    ri.ungrasp()

    ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
    obj_to_world = pybullet_planning.multiply(ee_to_world, obj_to_ee)
    pybullet_planning.draw_pose(obj_to_world)

    path = ri.planj(ri.homej, obstacles=[plane, bin])
    for _ in (_ for j in path for _ in ri.movej(j)):
        p.stepSimulation()
        time.sleep(1 / 240)

    mercury.pybullet.step_and_sleep()


if __name__ == "__main__":
    main()
