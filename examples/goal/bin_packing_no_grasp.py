#!/usr/bin/env python

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


def spawn_object_in_hand(ri, class_id):
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
            visual_file=collision_file,
            collision_file=collision_file,
            position=(0, 0, 0),
            quaternion=quaternion,
            mass=0.1,
            rgba_color=imgviz.label_colormap()[class_id] / 255,
        )
        aabb_min, aabb_max = pybullet_planning.get_aabb(obj)
        obj_mind_to_world = (
            position + np.array([0, 0, -aabb_max[2] - 0.01]),
            quaternion,
        )
        pos = obj_mind_to_world[0]
        qua = obj_mind_to_world[1]
        if 0:  # XXX: noise
            pos += np.random.normal(0, [0.02 / 3, 0.02 / 3, 0], 3)
            qua += np.random.normal(0, 0.1 / 3, 4)
        obj_to_world = (pos, qua / np.linalg.norm(qua))
        p.resetBasePositionAndOrientation(
            obj, obj_to_world[0], obj_to_world[1]
        )

        obj_to_ee = mercury.pybullet.get_pose(
            obj, parent_body_id=ri.robot, parent_link_id=ri.ee
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


def main():
    pybullet_planning.connect()
    pybullet_planning.add_data_path()
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.3,
        cameraYaw=120,
        cameraPitch=-40,
        cameraTargetPosition=(0, 0, 0.2),
    )

    plane = p.loadURDF("plane.urdf")
    with pybullet_planning.LockRenderer():
        bin = create_bin(0.4, 0.35, 0.2)
        p.resetBasePositionAndOrientation(bin, [0.4, 0.4, 0.11], [0, 0, 0, 1])
        bin_aabb = np.array(p.getAABB(bin))
        bin_aabb[0] += 0.01
        bin_aabb[1] -= 0.01

    ri = mercury.pybullet.PandaRobotInterface()

    np.random.seed(1)

    placed_objects = []
    for _ in range(10):
        class_id = 3
        obj, obj_mind_to_ee, constraint_id = spawn_object_in_hand(
            ri, class_id=class_id
        )

        obj_mind_to_world = get_place_pose(
            obj, class_id, bin_aabb[0], bin_aabb[1]
        )
        if obj_mind_to_world[0] is None or obj_mind_to_world[1] is None:
            print("Warning: cannot find place pose")
            break
        ee_to_obj_mind = pybullet_planning.invert(obj_mind_to_ee)
        ee_to_world = pybullet_planning.multiply(
            obj_mind_to_world, ee_to_obj_mind
        )

        obstacles = [plane, bin] + placed_objects
        attachments = [
            pybullet_planning.Attachment(ri.robot, ri.ee, obj_mind_to_ee, obj)
        ]
        targj = get_pre_place_joint_positions(
            ri,
            ee_to_world,
            obstacles=obstacles,
            attachments=attachments,
        )

        path = ri.planj(targj, obstacles=obstacles, attachments=attachments)
        if path is None:
            print("Warning: failed to find collision-free path")
            break
        [ri.movej(j) for j in path]

        mercury.pybullet.step_and_sleep(0.5)

        ri.movep(ee_to_world)

        mercury.pybullet.step_and_sleep(0.5)

        p.removeConstraint(constraint_id)

        mercury.pybullet.step_and_sleep(0.5)

        ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
        obj_mind_to_world = p.multiplyTransforms(
            ee_to_world[0],
            ee_to_world[1],
            obj_mind_to_ee[0],
            obj_mind_to_ee[1],
        )
        mercury.pybullet.create_mesh_body(
            visual_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            position=obj_mind_to_world[0],
            quaternion=obj_mind_to_world[1],
            rgba_color=(1, 0, 0, 0.5),
        )
        placed_objects.append(obj)

        ri.movej(ri.homej)

    mercury.pybullet.step_and_sleep()

    pybullet_planning.disconnect()


if __name__ == "__main__":
    main()
