#!/usr/bin/env python

import time

import imgviz
import pybullet as p
import pybullet_planning

import mercury

import utils


def main():
    parser = utils.get_parser()
    args = parser.parse_args()

    utils.init_world()

    ri = mercury.pybullet.PandaRobotInterface()
    ri.add_camera(
        pose=utils.get_camera_pose(args.camera_config),
        height=240,
        width=320,
    )

    utils.load_pile(
        base_pose=([0.4, -0.4, 0], [0, 0, 0, 1]),
        npz_file="assets/pile_001.npz",
        enable_visual=args.enable_visual,
        mass=0.1,
    )

    pybullet_planning.draw_pose(
        ([0, 0, 0], [0, 0, 0, 1]), parent=ri.robot, parent_link=ri.ee
    )

    utils.pause(args.pause)

    c = mercury.geometry.Coordinate(*ri.get_pose("camera_link"))
    c.position = [0.4, -0.4, 0.7]
    j = ri.solve_ik(c.pose, move_target=ri.robot_model.camera_link)
    for _ in ri.movej(j):
        p.stepSimulation()
        time.sleep(1 / 240)

    rgb, _, _ = ri.get_camera_image()
    imgviz.io.pyglet_imshow(rgb)
    imgviz.io.pyglet_run()

    while True:
        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()
