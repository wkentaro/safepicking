#!/usr/bin/env python

import time

import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury

import box_placing
import shelf_placing


if __name__ == "__main__":
    if 1:
        env = box_placing.Env()
    else:
        env = shelf_placing.Env()

    if 1:
        # target_obj = env.objects["tomato_can"]
        target_obj = env.objects["sugar_box"]
    else:
        target_obj = env.objects["cracker_box_02"]
        # target_obj = env.objects["mustard_bottle"]

    with pp.WorldSaver():
        waypoints = []
        for _ in range(100):
            for distance in [-0.01, 0, 0.01]:
                normals = []
                for obj in env.objects.values():
                    if obj == target_obj:
                        continue
                    points = p.getClosestPoints(
                        target_obj, obj, distance=distance
                    )
                    for point in points:
                        # mercury.pybullet.draw_points(
                        #     [point[5]], colors=[0, 0, 1], size=3
                        # )
                        normal = np.array(point[7])
                        # pp.add_line(point[5], point[5] + 0.1 * normal)
                        normals.append(normal)
                if normals:
                    break
            position, quaternion = pp.get_pose(target_obj)
            vel = np.sum(normals, axis=0)
            if np.linalg.norm(vel) == 0:
                break
            vel /= np.linalg.norm(vel)
            # pp.add_line(position, position + 0.1 * vel, color=[1, 0, 0])

            position = position + 0.001 * vel
            pp.set_pose(target_obj, (position, quaternion))

            waypoints.append((position, quaternion))

    while True:
        for waypoint in waypoints:
            pp.set_pose(target_obj, waypoint)
            time.sleep(0.01)

    mercury.pybullet.pause()
