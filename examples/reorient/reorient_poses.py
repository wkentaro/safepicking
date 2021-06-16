#!/usr/bin/env python

import argparse
import itertools
import time

from loguru import logger
import numpy as np
import pybullet_planning as pp

import mercury

from env import Env
from planned import get_query_ocs


def get_reorient_poses2(env):
    query_ocs, query_ocs_normal_ends = get_query_ocs(env)
    index = np.argmin(
        np.linalg.norm(query_ocs - query_ocs.mean(axis=0), axis=1)
    )
    query_ocs = query_ocs[index]
    query_ocs_normal_end = query_ocs_normal_ends[index]

    pose_init = pp.get_pose(env.fg_object_id)

    world_saver = pp.WorldSaver()
    lock_renderer = pp.LockRenderer()

    aabb = pp.get_aabb(env.fg_object_id)
    aabb_extents = aabb[1] - aabb[0]
    max_extent = np.max(aabb_extents)

    # XY validation
    box = pp.create_box(max_extent, max_extent, max_extent)
    XY = []
    for size in [0.2, 0.3, 0.4]:
        for dx, dy in itertools.product(
            np.linspace(-size, size, num=7),
            np.linspace(-size, size, num=7),
        ):
            c = mercury.geometry.Coordinate(
                position=(pose_init[0][0] + dx, pose_init[0][1] + dy, 0.1)
            )
            pp.set_pose(box, c.pose)

            obstacles = [env.ri.robot] + env.object_ids
            obstacles.remove(env.fg_object_id)
            if not mercury.pybullet.is_colliding(box, obstacles):
                XY.append(c.position[:2])
    XY = np.array(XY)
    pp.remove_body(box)

    bounds = ([0.2, -0.4], [0.8, 0.4])

    pp.draw_aabb(
        ((bounds[0][0], bounds[0][1], 0), (bounds[1][0], bounds[1][1], 0.01)),
    )

    keep = (
        (bounds[0][0] <= XY[:, 0])
        & (XY[:, 0] < bounds[1][0])
        & (bounds[0][1] <= XY[:, 1])
        & (XY[:, 1] < bounds[1][1])
    )
    XY = XY[keep]

    indices = np.argsort(np.linalg.norm(XY - pose_init[0][:2], axis=1))[:10]
    XY = XY[indices]
    for x, y in XY:
        pp.draw_point((x, y, 0.01))

    # XY, ABG validation
    ABG = itertools.product(
        np.linspace(-np.pi, np.pi, num=9, endpoint=False),
        np.linspace(-np.pi, np.pi, num=9, endpoint=False),
        np.linspace(-np.pi, np.pi, num=9, endpoint=False),
    )
    poses = []
    angles = []
    for (x, y), (a, b, g) in itertools.product(XY, ABG):
        c = mercury.geometry.Coordinate(position=(x, y, 0))
        c.quaternion = mercury.geometry.quaternion_from_euler((a, b, g))
        pp.set_pose(env.fg_object_id, c.pose)

        c.position[2] = -pp.get_aabb(env.fg_object_id)[0][2]
        pp.set_pose(env.fg_object_id, c.pose)

        points = pp.body_collision_info(
            env.fg_object_id, env.plane, max_distance=0.2
        )
        distance_to_plane = min(point[8] for point in points)
        assert distance_to_plane > 0
        c.position[2] -= distance_to_plane
        c.position[2] += 0.02
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
        assert angle >= 0

        poses.append(np.hstack(c.pose))
        angles.append(angle)
    poses = np.array(poses)
    angles = np.array(angles)

    world_saver.restore()
    lock_renderer.restore()

    logger.info(f"poses: {poses.shape}")

    return poses, angles, query_ocs, query_ocs_normal_end


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
        "--min-angle",
        type=float,
        default=0,
        help="threshold [deg]",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=10,
        help="threshold [deg]",
    )
    args = parser.parse_args()

    args.min_angle = np.deg2rad(args.min_angle)
    args.max_angle = np.deg2rad(args.max_angle)

    env = Env(class_ids=args.class_ids, mp4=args.mp4)
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

    poses, angles, query_ocs, query_ocs_normal_end = get_reorient_poses2(env)

    keep = (args.min_angle <= angles) & (angles < args.max_angle)
    poses = poses[keep]

    logger.info(f"Generated reorient poses: {len(poses)}")

    while True:
        for pose in poses:
            with pp.WorldSaver():
                pp.set_pose(env.fg_object_id, (pose[:3], pose[3:]))

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
                    time.sleep(0.2)
                pp.remove_debug(debug)


if __name__ == "__main__":
    main()
