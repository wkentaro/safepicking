#!/usr/bin/env python

import argparse
import time

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning

import mercury

from bin_packing_no_act import get_place_pose
from create_bin import create_bin


def get_pre_place_joint_positions(
    ri, place_pose, obstacles=None, attachments=None, distance=0.02
):
    obstacles = obstacles or []
    attachments = attachments or []

    collision_fn = pybullet_planning.get_collision_fn(
        ri.robot, ri.joints, obstacles, attachments
    )

    with pybullet_planning.LockRenderer():
        for dx in [-distance, 0, distance]:
            for dy in [-distance, 0, distance]:
                for dz in [-distance, 0, distance]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    c = mercury.geometry.Coordinate(*place_pose)
                    c.translate([dx, dy, dz], wrt="local")
                    targj = ri.solve_ik(c.pose)
                    with pybullet_planning.WorldSaver():
                        is_colliding = collision_fn(targj)
                    if not is_colliding:
                        return targj


def spawn_object_in_hand(ri, class_id, noise=False):
    if class_id == 15:
        quaternion = mercury.geometry.quaternion_from_euler(
            [np.deg2rad(90), 0, 0]
        )
    else:
        quaternion = [0, 0, 0, 1]
    with pybullet_planning.LockRenderer():
        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        collision_file = mercury.pybullet.get_collision_file(visual_file)
        position, _ = pybullet_planning.get_link_pose(ri.robot, ri.ee)
        obj = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            # visual_file=collision_file,
            # rgba_color=imgviz.label_colormap()[class_id] / 255,
            collision_file=collision_file,
            position=(0, 0, 0),
            quaternion=quaternion,
            mass=0.1,
        )
        aabb_min, aabb_max = pybullet_planning.get_aabb(obj)
        obj_mind_to_world = (
            position + np.array([0, 0, -aabb_max[2] - 0.01]),
            quaternion,
        )

        pos = obj_mind_to_world[0]
        qua = obj_mind_to_world[1]
        if noise:
            pos += np.random.normal(0, [0.02 / 3, 0.02 / 3, 0], 3)
            qua += np.random.normal(0, 0.1 / 3, 4)
        obj_to_world = (pos, qua / np.linalg.norm(qua))
        pybullet_planning.set_pose(obj, obj_to_world)

        ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
        obj_to_ee = pybullet_planning.multiply(
            pybullet_planning.invert(ee_to_world), obj_to_world
        )
        constraint_id = p.createConstraint(
            parentBodyUniqueId=ri.robot,
            parentLinkIndex=ri.ee,
            childBodyUniqueId=obj,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=obj_to_ee[0],
            parentFrameOrientation=obj_to_ee[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0, 1),
        )
        p.changeConstraint(constraint_id, maxForce=50)

    ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
    world_to_ee = pybullet_planning.invert(ee_to_world)
    obj_mind_to_ee = pybullet_planning.multiply(world_to_ee, obj_mind_to_world)

    return obj, obj_mind_to_ee, constraint_id


def real_to_virtual(T_obj_to_world):
    real_to_virtual = ([0, 2, 0], [0, 0, 0, 1])
    return pybullet_planning.multiply(real_to_virtual, T_obj_to_world)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pause", action="store_true", help="pause")
    parser.add_argument("--perfect", action="store_true", help="noise")
    args = parser.parse_args()

    pybullet_planning.connect()
    pybullet_planning.add_data_path()
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraYaw=90,
        cameraPitch=-40,
        cameraTargetPosition=(0, 1, 0),
    )

    ri = mercury.pybullet.PandaRobotInterface()
    ri_v = mercury.pybullet.PandaRobotInterface(
        pose=real_to_virtual(pybullet_planning.get_pose(ri.robot))
    )

    with pybullet_planning.LockRenderer():
        plane = p.loadURDF("plane.urdf")
        bin = create_bin(0.4, 0.38, 0.2)
        bin_to_world = ([0.4, 0.4, 0.11], [0, 0, 0, 1])
        pybullet_planning.set_pose(bin, bin_to_world)
        bin_aabb = np.array(p.getAABB(bin))
        bin_aabb[0] += 0.01
        bin_aabb[1] -= 0.01

    with pybullet_planning.LockRenderer():
        bin2 = create_bin(0.4, 0.38, 0.2)
        pybullet_planning.set_pose(bin2, real_to_virtual(bin_to_world))

    if args.pause:
        print("Please press 'n' to start")
        while True:
            if ord("n") in p.getKeyboardEvents():
                break

    T_camera_to_world = mercury.geometry.Coordinate()
    T_camera_to_world.translate([-0.3, 0.4, 0.8], wrt="world")
    T_camera_to_world.rotate([0, 0, np.deg2rad(-90)], wrt="local")
    T_camera_to_world.rotate([np.deg2rad(-130), 0, 0], wrt="local")

    fovy = np.deg2rad(45)
    height = 480
    width = 640
    mercury.pybullet.draw_camera(
        fovy=fovy,
        height=height,
        width=width,
        pose=T_camera_to_world.pose,
    )
    pybullet_planning.draw_pose(T_camera_to_world.pose)
    mercury.pybullet.draw_camera(
        fovy=fovy,
        height=height,
        width=width,
        pose=real_to_virtual(T_camera_to_world.pose),
    )
    pybullet_planning.draw_pose(real_to_virtual(T_camera_to_world.pose))

    obj = None
    obj_v = None
    constraint_id = None

    class StepSimulation:
        def __init__(self):
            self.i = 0

        def __call__(self):
            p.stepSimulation()
            ri_v.setj(ri.getj())
            if constraint_id:
                pybullet_planning.set_pose(
                    obj_v,
                    pybullet_planning.multiply(
                        mercury.pybullet.get_pose(ri_v.robot, ri_v.ee),
                        obj_to_ee,
                    ),
                )
            if self.i % 8 == 0:
                rgb, _, segm = mercury.pybullet.get_camera_image(
                    T_camera_to_world.matrix,
                    fovy=fovy,
                    height=height,
                    width=width,
                )
                rgb[segm == plane] = [222, 184, 135]
                rgb_v, _, segm_v = mercury.pybullet.get_camera_image(
                    mercury.geometry.transformation_matrix(
                        *real_to_virtual(T_camera_to_world.pose)
                    ),
                    fovy=fovy,
                    height=height,
                    width=width,
                )
                imgviz.io.cv_imshow(
                    imgviz.tile(
                        [rgb, rgb_v, np.uint8(rgb * 0.5 + rgb_v * 0.5)],
                        shape=(1, 3),
                        border=(255, 255, 255),
                    ),
                    "shoulder_camera",
                )
                imgviz.io.cv_waitkey(1)
            else:
                time.sleep(1 / 240)
            self.i += 1

    step_simulation = StepSimulation()

    np.random.seed(1)

    placed_objects = []
    placed_objects_v = []
    for _ in range(2):
        class_id = 2
        obj, obj_to_ee, constraint_id = spawn_object_in_hand(
            ri, class_id=class_id, noise=not args.perfect
        )

        visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
        with pybullet_planning.LockRenderer():
            obj_v = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
            )

        obj_to_world = get_place_pose(obj, class_id, bin_aabb[0], bin_aabb[1])
        if obj_to_world[0] is None or obj_to_world[1] is None:
            print("Warning: cannot find place pose")
            break

        ee_to_world = pybullet_planning.multiply(
            obj_to_world, pybullet_planning.invert(obj_to_ee)
        )

        obstacles = [plane, bin] + placed_objects
        attachments = [
            pybullet_planning.Attachment(ri.robot, ri.ee, obj_to_ee, obj)
        ]
        targj = get_pre_place_joint_positions(
            ri,
            place_pose=ee_to_world,
            obstacles=obstacles,
            attachments=attachments,
        )

        path = ri.planj(targj, obstacles=obstacles, attachments=attachments)
        if path is None:
            print("Warning: failed to find collision-free path")
            break
        for j in path:
            for _ in ri.movej(j):
                step_simulation()

        [step_simulation() for _ in range(120)]

        for _ in ri.movej(ri.solve_ik(ee_to_world), speed=0.001):
            step_simulation()

        [step_simulation() for _ in range(120)]

        p.removeConstraint(constraint_id)
        constraint_id = None

        [step_simulation() for _ in range(120)]

        placed_objects.append(obj)
        placed_objects_v.append(obj_v)

        for _ in ri.movej(ri.homej):
            step_simulation()

        for obj, obj_v in zip(placed_objects, placed_objects_v):
            pybullet_planning.set_pose(
                obj_v, real_to_virtual(pybullet_planning.get_pose(obj))
            )

        [step_simulation() for _ in range(120)]

    mercury.pybullet.step_and_sleep()

    pybullet_planning.disconnect()


if __name__ == "__main__":
    main()
