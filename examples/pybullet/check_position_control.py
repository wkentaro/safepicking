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

    p.loadURDF("plane.urdf")
    ri = mercury.pybullet.PandaRobotInterface()

    pose = pybullet_planning.get_link_pose(ri.robot, ri.ee)
    c_reset = mercury.geometry.Coordinate(*pose)

    c = c_reset.copy()
    c.translate([0.2, 0, -0.5], wrt="world")

    while True:
        for _ in ri.movej(ri.solve_ik(c.pose)):
            p.stepSimulation()
            time.sleep(1 / 240)
        for _ in ri.movej(ri.homej):
            p.stepSimulation()
            time.sleep(1 / 240)

    pybullet_planning.disconnect()


if __name__ == "__main__":
    main()
