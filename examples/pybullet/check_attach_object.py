#!/usr/bin/env python

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

    p.loadURDF("plane.urdf")
    ri = mercury.pybullet.PandaRobotInterface()

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

    # -------------------------------------------------------------------------

    coord = mercury.geometry.Coordinate(
        *pybullet_planning.get_link_pose(ri.robot, ri.ee)
    )
    coord.translate([0.5, 0, -0.5], wrt="world")

    while True:
        ri.movep(coord.pose)
        ri.movej(ri.homej)

    pybullet_planning.disconnect()


if __name__ == "__main__":
    main()
