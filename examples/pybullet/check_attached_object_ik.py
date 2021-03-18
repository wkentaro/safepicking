#!/usr/bin/env python

import numpy as np
import pybullet as p
import pybullet_planning

import mercury


def main():
    pybullet_planning.connect(use_gui=True)
    pybullet_planning.add_data_path()

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraPitch=-20,
        cameraYaw=80,
        cameraTargetPosition=[0, 0, 0.2],
    )

    p.loadURDF("plane.urdf")
    ri = mercury.pybullet.PandaRobotInterface()

    # -------------------------------------------------------------------------

    cube = pybullet_planning.create_box(
        0.03, 0.05, 0.1, mass=0.1, color=(0, 1, 0, 1)
    )
    ee_to_world = mercury.pybullet.get_pose(ri.robot, ri.ee)
    pybullet_planning.draw_pose(ee_to_world)
    p.resetBasePositionAndOrientation(
        cube, ee_to_world[0] + np.array([0, 0, -0.015 - 0.05]), (0, 0, 0, 1)
    )
    cube_to_world = mercury.pybullet.get_pose(cube)
    pybullet_planning.draw_pose(cube_to_world)
    cube_to_ee = pybullet_planning.multiply(
        pybullet_planning.invert(ee_to_world), cube_to_world
    )
    ri.gripper.contact_constraint = p.createConstraint(
        parentBodyUniqueId=ri.robot,
        parentLinkIndex=ri.ee,
        childBodyUniqueId=cube,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=cube_to_ee[0],
        parentFrameOrientation=cube_to_ee[1],
        childFramePosition=(0, 0, 0),
        childFrameOrientation=(0, 0, 0, 1),
    )
    ri.gripper.activated = True

    attachments = [
        pybullet_planning.Attachment(ri.robot, ri.ee, cube_to_ee, cube)
    ]

    # -------------------------------------------------------------------------

    c_init = mercury.geometry.Coordinate(*pybullet_planning.get_pose(cube))

    c1 = c_init.copy()
    c1.translate([0.3, -0.3, -0.3], wrt="world")
    c1.rotate([np.deg2rad(45), 0, 0], wrt="local")
    pybullet_planning.draw_pose(c1.pose)

    c2 = c_init.copy()
    c2.translate([0.3, 0.3, -0.3], wrt="world")
    c2.rotate([np.deg2rad(-45), 0, 0], wrt="local")
    pybullet_planning.draw_pose(c2.pose)

    robot_model = ri.get_skrobot(attachments=attachments)
    while True:
        joint_positions = robot_model.inverse_kinematics(
            c1.skrobot_coords,
            move_target=robot_model.attachment_link0,
        )
        ri.movej(joint_positions[:-1])

        joint_positions = robot_model.inverse_kinematics(
            c2.skrobot_coords,
            move_target=robot_model.attachment_link0,
        )
        ri.movej(joint_positions[:-1])

    pybullet_planning.disconnect()


if __name__ == "__main__":
    main()
