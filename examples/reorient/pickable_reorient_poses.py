#!/usr/bin/env python

import argparse
import itertools

from loguru import logger
import numpy as np
import path
import pybullet_planning as pp

import mercury

from _env import Env
import _utils


home = path.Path("~").expanduser()


def get_reorient_poses(env):
    bounds = (
        (0.35, -0.55, env.TABLE_OFFSET + 0.001),
        (0.65, -0.35, env.TABLE_OFFSET + 0.001),
    )
    if env.debug:
        pp.draw_aabb(bounds)

    XY = np.array(
        list(
            itertools.product(
                np.linspace(bounds[0][0], bounds[1][0], num=10),
                np.linspace(bounds[0][1], bounds[1][1], num=8),
            )
        )
    )
    ABG = np.array(
        list(
            itertools.product(
                np.linspace(-np.pi, np.pi, num=8, endpoint=False),
                np.linspace(-np.pi, np.pi, num=8, endpoint=False),
                np.linspace(-np.pi, np.pi, num=8, endpoint=False),
            )
        )
    )

    aabb = pp.get_aabb(env.fg_object_id)
    # max_extent = np.sqrt(((aabb[1] - aabb[0]) ** 2).sum())
    max_extent = max(aabb[1] - aabb[0])

    with pp.LockRenderer(), pp.WorldSaver():
        XY_valid = []
        for x, y in XY:
            box = pp.create_box(w=max_extent, l=max_extent, h=0.5)
            pp.set_pose(box, ((x, y, 0), (0, 0, 0, 1)))
            obstacles = env.object_ids[:]
            obstacles.remove(env.fg_object_id)
            if mercury.pybullet.is_colliding(box, ids2=obstacles):
                pp.remove_body(box)
                continue
            pp.remove_body(box)

            if env.debug:
                pp.draw_point(
                    (x, y, env.TABLE_OFFSET + 0.001), color=(0, 1, 0, 1)
                )
            XY_valid.append((x, y))
    XY = XY_valid

    reorient_poses = []
    with pp.LockRenderer(), pp.WorldSaver():
        for a, b, g in ABG:
            x, y = XY[0]
            c = mercury.geometry.Coordinate(
                position=(x, y, 0),
                quaternion=_utils.get_canonical_quaternion(
                    class_id=_utils.get_class_id(env.fg_object_id)
                ),
            )
            c.rotate([a, b, g], wrt="world")
            pp.set_pose(env.fg_object_id, c.pose)

            c.position[2] = (
                -pp.get_aabb(env.fg_object_id)[0][2] + env.TABLE_OFFSET
            )
            pp.set_pose(env.fg_object_id, c.pose)

            points = pp.body_collision_info(
                env.fg_object_id, env.plane, max_distance=0.2
            )
            distance_to_plane = min(point[8] for point in points)
            assert distance_to_plane > 0
            c.position[2] += -distance_to_plane

            for x, y in XY:
                reorient_poses.append([x, y, c.position[2], *c.quaternion])
    return np.array(reorient_poses)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True, help="seed")
    args = parser.parse_args()

    env = Env(class_ids=[2, 3, 5, 11, 12, 15], gui=True)
    env.random_state = np.random.RandomState(args.seed)
    env.eval = True
    env.launch()

    with pp.LockRenderer():
        env.reset()
        for obj in mercury.pybullet.get_body_unique_ids():
            if obj in [env.plane, env.ri.robot] + env.object_ids:
                continue
            pp.remove_body(obj)

    reorient_poses = get_reorient_poses(env)
    logger.info(f"reorient_poses: {reorient_poses.shape}")

    for reorient_pose in reorient_poses:
        pp.set_pose(env.fg_object_id, np.hsplit(reorient_pose, [3]))
        mercury.pybullet.pause()


if __name__ == "__main__":
    main()
