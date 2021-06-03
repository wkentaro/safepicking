#!/usr/bin/env python

import argparse
import itertools
import time

from loguru import logger
import numpy as np
import pybullet_planning as pp

import mercury

from pick_and_place_env import PickAndPlaceEnv
from planned import get_query_ocs


def get_reorient_poses2(env, threshold=np.deg2rad(95)):
    query_ocs, query_ocs_normal_ends = get_query_ocs(env)
    index = np.argmin(
        np.linalg.norm(query_ocs - query_ocs.mean(axis=0), axis=1)
    )
    query_ocs = query_ocs[index]
    query_ocs_normal_end = query_ocs_normal_ends[index]

    pose_init = pp.get_pose(env.fg_object_id)

    world_saver = pp.WorldSaver()
    lock_renderer = pp.LockRenderer()

    # XYZ validation
    sphere = pp.create_sphere(radius=0.075)
    XYZ = []
    for dx, dy, z in itertools.product(
        np.linspace(-0.15, 0.15, num=5),
        np.linspace(-0.15, 0.15, num=5),
        np.linspace(0.075, 0.15, num=3),
    ):
        c = mercury.geometry.Coordinate(
            position=(pose_init[0][0] + dx, pose_init[0][1] + dy, z)
        )
        pp.set_pose(sphere, c.pose)

        if not mercury.pybullet.is_colliding(sphere):
            pp.draw_pose(c.pose)
            XYZ.append(c.position)
    pp.remove_body(sphere)

    # XYZ, ABG validation
    ABG = itertools.product(
        np.linspace(-np.pi, np.pi, num=9, endpoint=False),
        np.linspace(-np.pi, np.pi, num=9, endpoint=False),
        np.linspace(-np.pi, np.pi, num=9, endpoint=False),
    )
    poses = []
    for (x, y, z), (a, b, g) in itertools.product(XYZ, ABG):
        c = mercury.geometry.Coordinate(position=(x, y, z))
        c.quaternion = mercury.geometry.quaternion_from_euler((a, b, g))

        pp.set_pose(env.fg_object_id, c.pose)
        if mercury.pybullet.is_colliding(env.fg_object_id):
            continue

        ocs, ocs_normal_end = mercury.geometry.transform_points(
            [query_ocs, query_ocs_normal_end],
            mercury.geometry.transformation_matrix(*c.pose),
        )
        normal = -(ocs_normal_end - ocs)  # flip normal
        normal /= np.linalg.norm(normal)
        angle = np.arccos(np.dot([0, 0, 1], normal))
        if threshold is not None and angle >= threshold:
            continue

        poses.append(c.pose)

    poses = np.array(poses, dtype=object)
    np.random.shuffle(poses)

    world_saver.restore()
    lock_renderer.restore()

    return poses, query_ocs, query_ocs_normal_end


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--class-ids", type=int, nargs="+", required=True, help="class ids"
    )
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--on-plane", action="store_true", help="on plane")
    parser.add_argument("--simulate", action="store_true", help="simulate")
    parser.add_argument(
        "--threshold", type=float, default=10, help="threshold [deg]"
    )
    args = parser.parse_args()

    env = PickAndPlaceEnv(class_ids=args.class_ids, mp4=args.mp4)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.reset()

    if args.on_plane:
        with pp.LockRenderer():
            for object_id in env.object_ids:
                if object_id != env.fg_object_id:
                    pp.remove_body(object_id)

            for _ in range(2400):
                pp.step_simulation()

    poses, query_ocs, query_ocs_normal_end = get_reorient_poses2(
        env, threshold=np.deg2rad(args.threshold)
    )

    logger.info(f"Generated reorient poses: {len(poses)}")

    while True:
        for pose in poses:
            with pp.WorldSaver():
                pp.set_pose(env.fg_object_id, pose)

                debug = pp.add_line(
                    query_ocs,
                    query_ocs - 0.2 * (query_ocs_normal_end - query_ocs),
                    width=2,
                    parent=env.fg_object_id,
                )
                if args.simulate:
                    for _ in range(2400):
                        pp.step_simulation()
                        time.sleep(pp.get_time_step() / 10)
                else:
                    time.sleep(0.1)
                pp.remove_debug(debug)


if __name__ == "__main__":
    main()
