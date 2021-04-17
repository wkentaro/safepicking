#!/usr/bin/env python

import time

import imgviz
import numpy as np
import pybullet_planning as pp

import mercury

import utils


def main():
    utils.init_world()

    ri = mercury.pybullet.PandaRobotInterface()
    homej_quaternion = ri.get_pose("tipLink")[1]

    eulers = [
        [0, 0, 0],
        [np.deg2rad(30), 0, 0],
        [np.deg2rad(-30), 0, 0],
        [0, np.deg2rad(30), 0],
        [0, np.deg2rad(-30), 0],
    ]

    if 0:
        c_homej = mercury.geometry.Coordinate(*ri.get_pose("tipLink"))
        for euler in eulers:
            c = c_homej.copy()
            c.rotate(euler)
            j = ri.solve_ik(c.pose)
            ri.setj(j)
            time.sleep(1)

    aabb = ((-0.8, -0.8, 0), (0.8, 0.8, 0.5))
    pp.draw_aabb(aabb)

    x, y, z = np.meshgrid(
        np.linspace(aabb[0][0], aabb[1][0], num=10),
        np.linspace(aabb[0][1], aabb[1][1], num=10),
        np.linspace(aabb[0][2], aabb[1][2], num=5),
    )
    xyz = np.stack([x, y, z], axis=3)

    points = xyz.reshape(-1, 3)
    for point in np.random.permutation(points):
        c_cano = mercury.geometry.Coordinate(
            point, quaternion=homej_quaternion
        )

        num_solutions = 0
        for euler in eulers:
            c = c_cano.copy()
            c.rotate(euler)
            j = ri.solve_ik(c.pose, rotation_axis="z")
            if j is not None:
                num_solutions += 1
        num_solutions /= len(eulers)

        color = (
            imgviz.depth2rgb(
                np.array([num_solutions]).reshape(1, 1),
                min_value=0,
                max_value=1,
            )[0, 0]
            / 255
        )

        box = pp.create_box(0.1, 0.1, 0.1, color=color.tolist() + [0.8])
        pp.set_pose(box, (point, (0, 0, 0, 1)))

    while True:
        pass


if __name__ == "__main__":
    main()
