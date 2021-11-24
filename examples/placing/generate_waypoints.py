#!/usr/bin/env python

import itertools

# import time

import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury

import box_placing
import shelf_placing


if __name__ == "__main__":
    target_obj = "sugar_box"
    # target_obj = "tomato_can"
    # target_obj = "cracker_box_02"
    # target_obj = "mustard_bottle"
    if 1:
        env = box_placing.Env(
            # mp4=f"box-{target_obj}.mp4"
        )
    else:
        env = shelf_placing.Env(
            # mp4=f"shelf-{target_obj}.mp4"
        )
    target_obj = env.objects[target_obj]

    mercury.pybullet.duplicate(
        target_obj,
        visual=True,
        collision=False,
        mass=0,
        rgba_color=(1, 0, 0, 0.5),
    )

    pp.disable_gravity()

    def step():
        force_max = [0, 0, 0]
        for obj in env.objects.values():
            pp.set_velocity(obj, ((0, 0, 0), (0, 0, 0)))

            if obj == target_obj:
                continue
            points = p.getClosestPoints(target_obj, obj, distance=0.02)
            for point in points:
                normal = np.array(point[7])
                force = np.array(normal) * 100 * (0.03 - point[8])
                force_max = np.maximum(force_max, np.abs(force))
                p.applyExternalForce(
                    objectUniqueId=target_obj,
                    linkIndex=-1,
                    forceObj=force,
                    posObj=point[5],
                    flags=p.WORLD_FRAME,
                )
        pp.step_simulation()
        return force_max

    with pp.WorldSaver():
        waypoints = [np.hstack(pp.get_pose(target_obj))]
        bounds = [pp.get_aabb(target_obj)]
        for i in itertools.count():
            for _ in range(10):
                force_max = step()

            bounds.append(pp.get_aabb(target_obj))

            bound_origin = bounds[0]
            bound_current = bounds[-1]

            bound_diff_lower = bound_origin.lower - bound_current.lower
            bound_diff_upper = bound_origin.upper - bound_current.upper
            bound_diff = np.maximum(
                np.abs(bound_diff_lower), np.abs(bound_diff_upper)
            )

            if i > 5:
                c = mercury.geometry.Coordinate(*pp.get_pose(target_obj))
                flag = 0
                if force_max[0] >= 2.9 and bound_diff[0] < 0.02:
                    flag = 1
                    c.rotate([0, np.deg2rad(1), 0], wrt="world")
                if force_max[1] >= 2.9 and bound_diff[1] < 0.02:
                    flag = 1
                    c.rotate([np.deg2rad(1), 0, 0], wrt="world")
                if flag:
                    pp.set_pose(target_obj, c.pose)
                    for _ in range(10):
                        step()

            waypoints.append(np.hstack(pp.get_pose(target_obj)))

            if not mercury.pybullet.is_colliding(target_obj, distance=0.02):
                break
            if i > 100:
                break

    np.save("sugar_box.npy", waypoints)

    pp.enable_gravity()

    # while True:
    #     for waypoint in waypoints:
    #         c = mercury.geometry.Coordinate(*np.hsplit(waypoint, [3]))
    #         c.translate([-0.01, -0.01, 0], wrt="world")
    #         pp.set_pose(target_obj, c.pose)
    #         for obj in env.objects.values():
    #             if obj == target_obj:
    #                 continue
    #             points = p.getClosestPoints(target_obj, obj, distance=0)
    #             if points:
    #                 for point in points:
    #                     pp.draw_point(point[5], color=[0, 0, 1])
    #         time.sleep(1 / 1000)

    mercury.pybullet.pause()
