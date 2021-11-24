#!/usr/bin/env python

import itertools

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

    with pp.WorldSaver():
        waypoints = [np.hstack(pp.get_pose(target_obj))]
        bounds = [pp.get_aabb(target_obj)]
        for i in itertools.count():
            force_max = [0, 0, 0]

            pp.remove_all_debug()
            for obj in env.objects.values():
                pp.set_velocity(obj, ((0, 0, 0), (0, 0, 0)))

                for distance in [-0.01, 0, 0.01, 0.02]:
                    if obj == target_obj:
                        continue
                    points = p.getClosestPoints(
                        target_obj, obj, distance=distance
                    )
                    for point in points:
                        normal = np.array(point[7])

                        force = np.array(normal) * 100 * (0.03 - distance)
                        force_max = np.maximum(force_max, np.abs(force))

                        p.applyExternalForce(
                            objectUniqueId=target_obj,
                            linkIndex=-1,
                            forceObj=force,
                            posObj=point[5],
                            flags=p.WORLD_FRAME,
                        )
            pp.step_simulation()

            bounds.append(pp.get_aabb(target_obj))

            bound_origin = bounds[0]
            bound_current = bounds[-1]

            bound_diff_lower = bound_origin.lower - bound_current.lower
            bound_diff_upper = bound_origin.upper - bound_current.upper
            bound_diff = np.maximum(
                np.abs(bound_diff_lower), np.abs(bound_diff_upper)
            )

            # if i > 100:
            #     torque = [0, 0, 0]
            #     if force_max[0] >= 2.9 and bound_diff[0] < 0.02:
            #         torque[1] = np.pi * ((0.02 - bound_diff[0]) * 10)
            #     if force_max[1] >= 2.9 and bound_diff[1] < 0.02:
            #         torque[0] = np.pi * ((0.02 - bound_diff[1]) * 10)
            #     p.applyExternalTorque(
            #         objectUniqueId=target_obj,
            #         linkIndex=-1,
            #         torqueObj=torque,
            #         flags=p.WORLD_FRAME,
            #     )
            #     print(bound_diff, torque)

            waypoints.append(np.hstack(pp.get_pose(target_obj)))

            if not mercury.pybullet.is_colliding(target_obj, distance=0.02):
                break
            if i > 1000:
                break
            # time.sleep(1 / 240)

    # np.save("waypoints.npy", np.array(waypoints))

    mercury.pybullet.pause()
