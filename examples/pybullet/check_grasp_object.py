#!/usr/bin/env python

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning

import mercury


def main():
    pybullet_planning.connect()
    pybullet_planning.add_data_path()
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=120,
        cameraPitch=-30,
        cameraTargetPosition=(0, 0, 0.3),
    )

    p.loadURDF("plane.urdf")

    ri = mercury.pybullet.PandaRobotInterface()

    class_id = 3
    visual_file = mercury.datasets.ycb.get_visual_file(class_id=class_id)
    collision_file = mercury.pybullet.get_collision_file(visual_file)
    with pybullet_planning.LockRenderer():
        obj = mercury.pybullet.create_mesh_body(
            visual_file=collision_file,
            collision_file=collision_file,
            position=(0, 0, 0),
            quaternion=(0, 0, 0, 1),
            mass=0.1,
            rgba_color=imgviz.label_colormap()[class_id] / 255,
        )
    aabb_min, _ = mercury.pybullet.get_aabb(obj)
    ri.gripper.graspable_objects = [obj]

    spawn_aabb = [0.3, -0.3, 0], [0.5, 0.3, 0]
    pybullet_planning.draw_aabb(spawn_aabb)

    while True:
        p.resetBasePositionAndOrientation(
            obj,
            -aabb_min + np.random.uniform(*spawn_aabb),
            mercury.geometry.quaternion_from_euler(
                [0, 0, np.random.uniform(-np.pi, np.pi)]
            ),
        )

        c = mercury.geometry.Coordinate(
            *mercury.pybullet.get_pose(ri.robot, ri.ee)
        )
        c.position = pybullet_planning.get_pose(obj)[0]
        c.position[2] = pybullet_planning.get_aabb(obj)[1][2] + 0.05

        ri.movep(c.pose)

        ri.grasp()

        c = mercury.geometry.Coordinate(
            *mercury.pybullet.get_pose(ri.robot, ri.ee)
        )

        ri.movej(ri.homej)

        ri.movep(c.pose)

        mercury.pybullet.step_and_sleep(0.5)

        ri.ungrasp()

        ri.movej(ri.homej)


if __name__ == "__main__":
    main()
