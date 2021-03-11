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
    with pybullet_planning.HideOutput():
        with pybullet_planning.LockRenderer():
            robot = pybullet_planning.load_pybullet(
                "franka_panda/panda.urdf", fixed_base=True
            )
    pybullet_planning.dump_body(robot)

    ri = mercury.pybullet.PandaRobotInterface(robot)

    pose = pybullet_planning.get_link_pose(ri.robot, ri.ee)
    coord_reset = mercury.geometry.Coordinate(*pose)

    coord = coord_reset.copy()
    coord.translate([0.2, 0, -0.5], wrt="world")

    while True:
        targj = ri.solve_ik((coord.position, coord.quaternion))
        ri.movej(targj)
        targj = ri.solve_ik((coord_reset.position, coord_reset.quaternion))
        ri.movej(targj)

    pybullet_planning.disconnect()


if __name__ == "__main__":
    main()
