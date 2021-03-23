#!/usr/bin/env python

import time

import pybullet as p
import pybullet_planning

import mercury


def main():
    pybullet_planning.connect(use_gui=True)
    pybullet_planning.add_data_path()

    p.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraPitch=-20,
        cameraYaw=80,
        cameraTargetPosition=[0, 0, 0],
    )

    plane = p.loadURDF("plane.urdf")
    ri = mercury.pybullet.PandaRobotInterface()

    # -------------------------------------------------------------------------

    box = pybullet_planning.create_box(0.7, 0.1, 0.4)
    p.resetBasePositionAndOrientation(box, [0.5, 0, 0.2], [0, 0, 0, 1])

    # -------------------------------------------------------------------------

    cube = pybullet_planning.create_box(
        0.05, 0.05, 0.05, mass=0.1, color=(0, 1, 0, 1)
    )
    p.resetBasePositionAndOrientation(cube, (0, 0, 0.7), (0, 0, 0, 1))
    p.createConstraint(
        parentBodyUniqueId=ri.robot,
        parentLinkIndex=ri.ee,
        childBodyUniqueId=cube,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0.04),
        parentFrameOrientation=(0, 0, 0, 1),
        childFramePosition=(0, 0, 0),
    )
    attachments = [
        pybullet_planning.Attachment(
            ri.robot, ri.ee, [(0, 0, 0.04), (0, 0, 0, 1)], cube
        )
    ]

    # -------------------------------------------------------------------------

    c1 = mercury.geometry.Coordinate(
        *pybullet_planning.get_link_pose(ri.robot, ri.ee)
    )
    c1.translate([0.4, -0.4, -0.4], wrt="world")

    c2 = c1.copy()
    c2.translate([0, 0.8, 0], wrt="world")

    obstacles = [plane, box]

    while True:
        traj = ri.planj(
            ri.solve_ik(c1.pose),
            obstacles=obstacles,
            attachments=attachments,
        )
        for j in traj:
            for _ in ri.movej(j):
                p.stepSimulation()
                time.sleep(1 / 240)

        traj = ri.planj(
            ri.solve_ik(c2.pose),
            obstacles=obstacles,
            attachments=attachments,
        )
        for j in traj:
            for _ in ri.movej(j):
                p.stepSimulation()
                time.sleep(1 / 240)

    pybullet_planning.disconnect()


if __name__ == "__main__":
    main()
